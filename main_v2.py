#!/usr/bin/env python
"""
Math Trading V2 - DOM-Validated Statistical Trading System
===========================================================
Adds Rithmic L2 Order Book validation to v1's statistical engine.
DOM confirms or rejects statistical signals. Stop hunt, iceberg,
absorption, and SL zone detection for enhanced entry/exit timing.

Usage:
  python main_v2.py dashboard
  python main_v2.py backtest --data <file.csv> --tf 5
  python main_v2.py rithmic-test
  python main_v2.py compare --data <file.csv> --tf 5
"""

import argparse, sys, os, pandas as pd, numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from config.settings_v2 import V2Config
from config.settings import GOLD_CONFIG
from data.synthetic import SyntheticDataGenerator
from data.rithmic import SyntheticDOMGenerator
from signals.generator_v2 import SignalGeneratorV2
from backtest.engine import StatisticalBacktester
from backtest.metrics import calculate_metrics
from live.mt5_adapter import MT5Connector
from live.rithmic_adapter import RithmicAdapter
from risk.stops import atr_from_df


def cmd_dashboard(args):
    from dashboard.app_v2 import create_dashboard
    from dashboard.data_provider_v2 import DashboardDataProviderV2

    cfg = V2Config()
    if hasattr(args, "env"):
        cfg.load_env()
    provider = DashboardDataProviderV2(config=cfg)
    print("Connecting MT5 + Rithmic...")
    provider.connect_all()
    if provider._connected_mt5:
        acc = provider.mt5.get_account_info()
        print(f"\n  Account: #{acc.get('login','?')} Balance: ${acc.get('balance',0):.2f}")
    print(f"  Rithmic: {'LIVE' if provider._connected_rithmic else 'SYNTHETIC'}")
    provider.refresh()
    app = create_dashboard(provider, refresh_interval_ms=args.interval)
    print(f"\n  V2 DASHBOARD: http://{args.host}:{args.port}")
    app.run(debug=False, host=args.host, port=args.port)


def cmd_rithmic_test(args):
    cfg = V2Config()
    if hasattr(args, "env"):
        cfg.load_env()
    adapter = RithmicAdapter(config=cfg)
    print("Testing Rithmic connection...")
    if adapter.connect():
        print(f"  Connected: {'RITHMIC LIVE' if not adapter.is_synthetic else 'SYNTHETIC (simulated)'}")
        print("  Streaming 5 DOM snapshots...")
        for i in range(5):
            snap = adapter.get_snapshot(mid_price=4800, bar_direction=0)
            ofs = adapter.get_orderflow_state()
            print(f"  [{i+1}] Bid/Ask: {ofs['top5_bid_vol']:.0f}/{ofs['top5_ask_vol']:.0f} "
                  f"OFI: {ofs['ofi']:+.0f} Delta: {ofs['cum_delta']:+.0f} "
                  f"B/A Ratio: {ofs['bid_ask_ratio']:.2f}")
        adapter.disconnect()
    else:
        print("  FAILED. Check credentials in .env or RITHMIC_* environment variables.")


def cmd_backtest(args):
    df, dom_snaps = _load_data(args)
    cfg = V2Config()
    cfg.timeframe.primary = args.tf or 5

    print(f"Running V2 backtest on {len(df)} bars...")
    gen = SignalGeneratorV2(config=cfg)
    dom_gen = SyntheticDOMGenerator(seed=42)

    close = df["close"].values.astype(np.float64)
    snapshots = dom_snaps or [dom_gen.generate_snapshot(float(p), 0) for p in close]

    atr = float(np.std(df["high"] - df["low"]))

    gen.load_dom_snapshots(snapshots)
    orderflow_df = gen.compute_orderflow_features(snapshots, close, atr)
    sl_zones = gen.compute_sl_zones(close, atr)
    stop_hunt = gen.compute_stop_hunt(close, sl_zones, atr)

    features = gen.compute_features(df)
    signals = gen.generate_signals_v2(features, orderflow_df, stop_hunt, sl_zones)

    v1_count = int(signals["signal_v1"].abs().sum())
    v2_count = int(signals["signal"].abs().sum())
    rejected = int(signals.get("dom_rejected", pd.Series(False)).sum())

    print(f"  V1 signals: {v1_count}")
    print(f"  V2 signals (DOM-validated): {v2_count}")
    print(f"  DOM rejected: {rejected} ({(rejected/max(1,v1_count)*100):.0f}%)")

    from risk.exits import apply_exits_to_df
    cfg_v1 = GOLD_CONFIG
    cfg_v1.timeframe.primary = args.tf or 5
    signs_df = apply_exits_to_df(signals, max_bars=cfg_v1.threshold.time_stop_bars,
                                 zscore_stop_long=cfg_v1.threshold.zscore_stop_long,
                                 zscore_stop_short=cfg_v1.threshold.zscore_stop_short)

    entries = (signs_df["signal"] == 1).astype(bool)
    exits = (signs_df["exit_signal"] == -1).astype(bool)
    short_entries = (signs_df["signal"] == -1).astype(bool)
    short_exits = (signs_df["exit_signal"] == 1).astype(bool)

    import vectorbt as vbt
    try:
        pf = vbt.Portfolio.from_signals(close=close, entries=entries, exits=exits,
                                        short_entries=short_entries, short_exits=short_exits,
                                        init_cash=10000.0, freq=f"{args.tf or 5}min")
        metrics = calculate_metrics(pf, cfg_v1)
        print(f"\n  V2 BACKTEST RESULTS:")
        print(f"  Return: {metrics['total_return_pct']:.2f}% | Sharpe: {metrics['sharpe_ratio']:.3f} | "
              f"Trades: {metrics['num_trades']} | WR: {metrics['win_rate_pct']:.0f}% | DD: {metrics['max_drawdown_pct']:.2f}%")
    except Exception as e:
        print(f"  Backtest error: {e}")


def cmd_compare(args):
    df, dom_snaps = _load_data(args)
    cfg_v2 = V2Config()
    cfg_v2.timeframe.primary = args.tf or 5
    cfg_v1 = GOLD_CONFIG
    cfg_v1.timeframe.primary = args.tf or 5

    print(f"V1 vs V2 comparison on {len(df)} bars ({args.tf or 5}-min)")
    print(f"{'='*70}")

    close = df["close"].values.astype(np.float64)

    dom_gen = SyntheticDOMGenerator(seed=42)
    snapshots = dom_snaps or [dom_gen.generate_snapshot(float(p), 0) for p in close]
    atr = float(np.std(df["high"] - df["low"]))

    import vectorbt as vbt
    from risk.exits import apply_exits_to_df

    s_v1 = SignalGeneratorV2(config=cfg_v1)
    f_v1 = s_v1.compute_and_generate(df)
    signs_v1 = apply_exits_to_df(f_v1, max_bars=cfg_v1.threshold.time_stop_bars,
                                 zscore_stop_long=cfg_v1.threshold.zscore_stop_long,
                                 zscore_stop_short=cfg_v1.threshold.zscore_stop_short)

    try:
        pf1 = vbt.Portfolio.from_signals(
            close=close, entries=(signs_v1["signal"] == 1).astype(bool),
            exits=(signs_v1["exit_signal"] == -1).astype(bool),
            short_entries=(signs_v1["signal"] == -1).astype(bool),
            short_exits=(signs_v1["exit_signal"] == 1).astype(bool),
            init_cash=10000.0, freq=f"{args.tf or 5}min")
        m1 = calculate_metrics(pf1, cfg_v1)
    except Exception:
        m1 = dict.fromkeys(["total_return_pct", "sharpe_ratio", "num_trades", "win_rate_pct", "max_drawdown_pct"], 0)

    s_v2 = SignalGeneratorV2(config=cfg_v2)
    s_v2.load_dom_snapshots(snapshots)
    of_df = s_v2.compute_orderflow_features(snapshots, close, atr)
    sl_z = s_v2.compute_sl_zones(close, atr)
    sh = s_v2.compute_stop_hunt(close, sl_z, atr)
    f_v2 = s_v2.compute_features(df)
    signs_v2 = s_v2.generate_signals_v2(f_v2, of_df, sh, sl_z)
    signs_v2 = apply_exits_to_df(signs_v2, max_bars=cfg_v1.threshold.time_stop_bars,
                                 zscore_stop_long=cfg_v1.threshold.zscore_stop_long,
                                 zscore_stop_short=cfg_v1.threshold.zscore_stop_short)

    try:
        pf2 = vbt.Portfolio.from_signals(
            close=close, entries=(signs_v2["signal"] == 1).astype(bool),
            exits=(signs_v2["exit_signal"] == -1).astype(bool),
            short_entries=(signs_v2["signal"] == -1).astype(bool),
            short_exits=(signs_v2["exit_signal"] == 1).astype(bool),
            init_cash=10000.0, freq=f"{args.tf or 5}min")
        m2 = calculate_metrics(pf2, cfg_v1)
    except Exception:
        m2 = dict.fromkeys(["total_return_pct", "sharpe_ratio", "num_trades", "win_rate_pct", "max_drawdown_pct"], 0)

    rows = [
        ["Metric", "V1 (Stat Only)", "V2 (Stat + DOM)", "Delta"],
        ["Return %", f"{m1['total_return_pct']:.2f}%", f"{m2['total_return_pct']:.2f}%",
         f"{m2['total_return_pct'] - m1['total_return_pct']:+.2f}%"],
        ["Sharpe", f"{m1['sharpe_ratio']:.3f}", f"{m2['sharpe_ratio']:.3f}",
         f"{m2['sharpe_ratio'] - m1['sharpe_ratio']:+.3f}"],
        ["Trades", str(m1["num_trades"]), str(m2["num_trades"]),
         f"{m2['num_trades'] - m1['num_trades']:+d}"],
        ["Win Rate", f"{m1['win_rate_pct']:.0f}%", f"{m2['win_rate_pct']:.0f}%",
         f"{m2['win_rate_pct'] - m1['win_rate_pct']:+.0f}%"],
        ["Max DD", f"{m1['max_drawdown_pct']:.2f}%", f"{m2['max_drawdown_pct']:.2f}%",
         f"{m2['max_drawdown_pct'] - m1['max_drawdown_pct']:+.2f}%"],
    ]
    for row in rows:
        print(f"  {row[0]:<12} {row[1]:>18} {row[2]:>18} {row[3]:>10}")


def _load_data(args):
    if args.data and os.path.exists(args.data):
        df = pd.read_csv(args.data, index_col=0, parse_dates=True)
    elif args.mt5:
        mt5 = MT5Connector(symbol="GOLD-Pro")
        mt5.connect()
        try:
            df = mt5.fetch_rates(timeframe=args.tf or 5, count=args.bars or 3000)
        finally:
            mt5.disconnect()
    else:
        gen = SyntheticDataGenerator(seed=42)
        df = gen.generate_regime_data(n_bars=args.bars or 3000, n_ranging=int((args.bars or 3000) * 0.6),
                                      n_trending=int((args.bars or 3000) * 0.4), timeframe=args.tf or 5)
    return df, None


def main():
    parser = argparse.ArgumentParser(description="Math Trading V2 - DOM-Validated Statistical System")
    sub = parser.add_subparsers(dest="command")

    d = sub.add_parser("dashboard")
    d.add_argument("--port", type=int, default=8050)
    d.add_argument("--host", default="127.0.0.1")
    d.add_argument("--interval", type=int, default=5000)
    d.add_argument("--env", action="store_true")

    r = sub.add_parser("rithmic-test")
    r.add_argument("--env", action="store_true")

    b = sub.add_parser("backtest")
    b.add_argument("--data")
    b.add_argument("--mt5", action="store_true")
    b.add_argument("--tf", type=int, default=5)
    b.add_argument("--bars", type=int, default=3000)

    c = sub.add_parser("compare")
    c.add_argument("--data")
    c.add_argument("--mt5", action="store_true")
    c.add_argument("--tf", type=int, default=5)
    c.add_argument("--bars", type=int, default=3000)

    args = parser.parse_args()
    cmds = {"dashboard": cmd_dashboard, "rithmic-test": cmd_rithmic_test,
            "backtest": cmd_backtest, "compare": cmd_compare}
    if args.command in cmds:
        cmds[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
