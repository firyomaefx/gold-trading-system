#!/usr/bin/env python
"""
XAU/USD Statistical Trading System
===================================
High-frequency mean-reversion strategy for Gold using:
  - Hurst Exponent (variance-time method)
  - Rolling Z-Score (statistical anomaly detection)
  - GARCH volatility forecasting
  - Hidden Markov Models (regime detection)
  - Kelly Criterion position sizing

Usage:
  python main.py fetch --tf 5
  python main.py generate --bars 5000 --tf 5
  python main.py backtest --data <file.csv> --tf 5
  python main.py calibrate --tf 5
  python main.py optimize --data <file.csv> --tf 5
  python main.py live --tf 5
"""

import argparse
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from config.settings import GOLD_CONFIG, DEFAULT_PARAM_GRID
from data.synthetic import SyntheticDataGenerator
from data.fetcher import get_source
from backtest.engine import StatisticalBacktester
from backtest.metrics import trade_summary
from backtest.optimizer import Optimizer
from live.mt5_adapter import MT5Connector
from live.trader import LiveTrader
from dashboard.app import create_dashboard
from dashboard.data_provider import DashboardDataProvider


def cmd_fetch(args):
    mt5 = MT5Connector(symbol=GOLD_CONFIG.symbol.symbol)
    if not mt5.connect():
        print("ERROR: Cannot connect to MT5. Ensure MetaTrader 5 terminal is running.")
        return

    try:
        tf = args.tf or 5
        df = mt5.fetch_rates(timeframe=tf, count=args.bars)
        out_path = args.output or f"xauusd_mt5_{tf}m_{args.bars}.csv"
        df.to_csv(out_path)
        print(f"Saved {len(df)} bars to {out_path}")
        print(f"Date range: {df.index[0]} to {df.index[-1]}")
        print(f"Price range: {df['close'].min():.2f} - {df['close'].max():.2f}")
    finally:
        mt5.disconnect()


def cmd_generate(args):
    print(f"Generating {args.bars} synthetic {args.tf}-minute bars for XAU/USD...")
    gen = SyntheticDataGenerator(seed=args.seed)
    df = gen.generate_regime_data(
        n_bars=args.bars,
        n_ranging=int(args.bars * 0.6),
        n_trending=int(args.bars * 0.4),
        timeframe=args.tf,
        start_price=args.price,
    )
    out_path = args.output or f"xauusd_synthetic_{args.tf}m_{args.bars}.csv"
    df.to_csv(out_path)
    print(f"Saved {len(df)} bars to {out_path}")
    print(f"Price range: {df['close'].min():.2f} - {df['close'].max():.2f}")


def cmd_backtest(args):
    if args.data and os.path.exists(args.data):
        print(f"Loading data from {args.data}...")
        df = pd.read_csv(args.data, index_col=0, parse_dates=True)
    elif args.mt5:
        print("Fetching data from MT5...")
        mt5 = MT5Connector(symbol=GOLD_CONFIG.symbol.symbol)
        mt5.connect()
        try:
            df = mt5.fetch_rates(timeframe=args.tf or 5, count=args.bars or 5000)
            if args.save:
                df.to_csv(f"xauusd_mt5_{args.tf}m_{len(df)}.csv")
        finally:
            mt5.disconnect()
    else:
        print(f"No data file found. Generating synthetic {args.bars} bars...")
        gen = SyntheticDataGenerator(seed=42)
        df = gen.generate_regime_data(n_bars=args.bars or 5000, timeframe=args.tf or 5)
        if args.save:
            df.to_csv(f"xauusd_synthetic_{args.tf}m_{args.bars}.csv")

    config = GOLD_CONFIG
    cfg_tf = args.tf or config.timeframe.primary
    config.timeframe.primary = cfg_tf
    config.threshold.hmm_ranging_prob = 0.0

    print(f"\nRunning backtest on {len(df)} bars ({cfg_tf}-minute)...")
    bt = StatisticalBacktester(config)
    results = bt.run(df, use_kelly=not args.no_kelly)

    print(trade_summary(results["metrics"]))

    if args.export:
        eq = bt.equity_curve()
        eq.to_csv(f"equity_{cfg_tf}m_{datetime.now().strftime('%Y%m%d')}.csv")
        print(f"Equity curve exported.")

    if args.details and "trades" in results and results["trades"] is not None:
        print("\nLast 10 trades:")
        print(results["trades"].tail(10).to_string())

    return results


def cmd_calibrate(args):
    mt5 = MT5Connector(symbol=GOLD_CONFIG.symbol.symbol)
    if not mt5.connect():
        print("ERROR: Cannot connect to MT5.")
        return

    try:
        tf = args.tf or 5
        count = args.bars or 2000
        print(f"Fetching {count} bars of {tf}-minute XAU/USD from MT5...")
        df = mt5.fetch_rates(timeframe=tf, count=count)

        if args.save:
            df.to_csv(f"xauusd_calibration_{tf}m.csv")

        close = df["close"].values.astype(np.float64)
        returns = np.diff(close) / close[:-1]
        returns = returns[np.isfinite(returns)]

        from stats.hmm import HMMRegimeDetector
        print("\nCalibrating HMM (2-state regime detector)...")
        hmm = HMMRegimeDetector(n_states=2, random_state=42)
        hmm.fit(returns)
        print(f"  Ranging state ID: {hmm._ranging_state}")
        print(f"  State labels: {hmm._state_labels}")

        probing_probs = hmm.predict_proba_series(returns)
        print(f"  Ranging probability stats:")
        print(f"    Mean: {probing_probs.mean():.4f}")
        print(f"    Min:  {probing_probs.min():.4f}")
        print(f"    Max:  {probing_probs.max():.4f}")
        print(f"    Pct > 0.80: {(probing_probs > 0.80).mean() * 100:.1f}%")
        print(f"    Pct > 0.90: {(probing_probs > 0.90).mean() * 100:.1f}%")

        from stats.garch import GARCHForecaster
        print("\nCalibrating GARCH(1,1) model...")
        forecaster = GARCHForecaster(p=1, q=1)
        forecaster.fit(returns)
        vol_forecast = forecaster.forecast_volatility(horizon=5)
        print(f"  Latest conditional vol: {forecaster.latest_volatility:.6f}")
        print(f"  Annualized vol: {forecaster.latest_volatility * np.sqrt(252 * 78):.2%}")
        print(f"  Vol forecast (next 5 bars): {vol_forecast}")
        print(f"  Tightening: {forecaster.is_volatility_tightening(horizon=10)}")

        from stats.distributions import kurtosis, is_fat_tailed
        k = kurtosis(returns)
        print(f"\nDistribution analysis:")
        print(f"  Kurtosis (excess): {k:.4f}")
        print(f"  Fat-tailed (k > 3): {is_fat_tailed(returns, threshold=3)}")

        from stats.hurst import hurst_exponent
        h = hurst_exponent(close)
        print(f"  Hurst Exponent: {h:.4f}")
        print(f"  Interpretation: ", end="")
        if h < 0.45:
            print("Mean-reverting (good for scalping)")
        elif h < 0.55:
            print("Random walk (~0.5)")
        else:
            print("Trending (momentum dominant)")

    finally:
        mt5.disconnect()


def cmd_optimize(args):
    if args.data and os.path.exists(args.data):
        print(f"Loading data from {args.data}...")
        df = pd.read_csv(args.data, index_col=0, parse_dates=True)
    elif args.mt5:
        print("Fetching data from MT5 for optimization...")
        mt5 = MT5Connector(symbol="XAUUSD")
        mt5.connect()
        try:
            df = mt5.fetch_rates(timeframe=args.tf or 5, count=args.bars or 3000)
        finally:
            mt5.disconnect()
    else:
        print(f"Generating synthetic data for optimization...")
        gen = SyntheticDataGenerator(seed=42)
        df = gen.generate_regime_data(n_bars=min(args.bars or 3000, 3000), timeframe=args.tf or 5)

    config = GOLD_CONFIG
    cfg_tf = args.tf or config.timeframe.primary
    config.timeframe.primary = cfg_tf
    config.threshold.hmm_ranging_prob = 0.0

    param_grid = {
        "threshold__hurst_mean_revert": [0.30, 0.35, 0.40, 0.45],
        "threshold__zscore_entry_long": [-3.0, -2.5, -2.0],
        "threshold__zscore_entry_short": [2.0, 2.5, 3.0],
        "threshold__velocity_epsilon": [0.5, 1.0, 2.0, 3.0],
        "window__rolling_zscore": [50, 100, 150],
        "threshold__time_stop_bars": [2, 3, 5, 7],
    }

    opt = Optimizer(config)
    results_df = opt.grid_search(df, param_grid=param_grid)

    print("\n" + "=" * 60)
    print("  TOP 5 CONFIGURATIONS (by Sharpe Ratio)")
    print("=" * 60)
    top5 = results_df.head(5)
    for i, (_, row) in enumerate(top5.iterrows()):
        print(f"  #{i + 1}: Sharpe={row['sharpe_ratio']:.3f}  Return={row['total_return_pct']:.1f}%  "
              f"Trades={int(row['num_trades'])}  WR={row['win_rate_pct']:.0f}%  "
              f"DD={row['max_drawdown_pct']:.1f}%")

    best = opt.best_params()
    print(f"\n  Best params saved to best_params.csv")
    print(f"  Optimal Hurst threshold: {best.get('hurst_mean_revert', 'N/A')}")
    print(f"  Optimal Z-score long:    {best.get('zscore_entry_long', 'N/A')}")
    print(f"  Optimal Z-score short:   {best.get('zscore_entry_short', 'N/A')}")
    print(f"  Optimal velocity eps:    {best.get('velocity_epsilon', 'N/A')}")
    print(f"  Optimal rolling window:  {best.get('rolling_zscore', 'N/A')}")
    print(f"  Optimal time stop:       {best.get('time_stop_bars', 'N/A')}")

    results_df.to_csv("optimization_results.csv")
    pd.Series(best).to_csv("best_params.csv")

    return results_df


def cmd_live(args):
    trader = LiveTrader()

    if not trader.connect():
        return

    try:
        tf = args.tf or trader.config.timeframe.primary

        print(f"\nLoading historical data...")
        trader.load_history(timeframe=tf, bars=args.bars or 2000)

        print(f"\nCalibrating HMM on real data...")
        trader.calibrate_hmm()

        trader.config.threshold.hmm_ranging_prob = args.hmm_gate or 0.70

        interval = args.interval or 60
        trader.run_loop(sleep_seconds=interval)

    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        print(trader.trade_summary())
        trader.stop()


def cmd_dashboard(args):
    from dashboard.app import create_dashboard
    from dashboard.data_provider import DashboardDataProvider

    provider = DashboardDataProvider(GOLD_CONFIG)

    print("Connecting to MT5...")
    if not provider.connect():
        print("ERROR: Cannot connect to MT5.")
        print("Ensure MetaTrader 5 terminal is running with GOLD-Pro chart open.")
        print("Starting dashboard in offline mode for layout preview...")

    provider.refresh()
    if provider.connected:
        acc = provider.mt5.get_account_info()
        print(f"\n  Account: #{acc.get('login', '?')}")
        print(f"  Balance: ${acc.get('balance', 0):.2f}")
        print(f"  Equity:  ${acc.get('equity', 0):.2f}")

    app = create_dashboard(provider, refresh_interval_ms=args.interval)

    print(f"\n{'=' * 55}")
    print(f"  DASHBOARD STARTING")
    print(f"  Open http://{args.host}:{args.port} in your browser")
    print(f"{'=' * 55}")

    app.run(debug=False, host=args.host, port=args.port)


def main():
    parser = argparse.ArgumentParser(description="XAU/USD Statistical Trading System")
    subparsers = parser.add_subparsers(dest="command", help="Command")

    fetch_parser = subparsers.add_parser("fetch", help="Fetch real XAU/USD data from MT5")
    fetch_parser.add_argument("--tf", type=int, default=5, help="Timeframe (minutes)")
    fetch_parser.add_argument("--bars", type=int, default=5000, help="Number of bars")
    fetch_parser.add_argument("--output", type=str, default=None, help="Output CSV")

    gen_parser = subparsers.add_parser("generate", help="Generate synthetic data")
    gen_parser.add_argument("--bars", type=int, default=5000)
    gen_parser.add_argument("--tf", type=int, default=5)
    gen_parser.add_argument("--price", type=float, default=2000.0)
    gen_parser.add_argument("--seed", type=int, default=42)
    gen_parser.add_argument("--output", type=str, default=None)

    bt_parser = subparsers.add_parser("backtest", help="Run backtest")
    bt_parser.add_argument("--data", type=str, default=None, help="CSV data file")
    bt_parser.add_argument("--mt5", action="store_true", help="Fetch data from MT5")
    bt_parser.add_argument("--tf", type=int, default=5)
    bt_parser.add_argument("--bars", type=int, default=5000)
    bt_parser.add_argument("--no-kelly", action="store_true")
    bt_parser.add_argument("--export", action="store_true", help="Export equity curve")
    bt_parser.add_argument("--save", action="store_true", help="Save data to CSV")
    bt_parser.add_argument("--details", action="store_true", help="Show trade details")

    cal_parser = subparsers.add_parser("calibrate", help="Calibrate HMM + GARCH on MT5 data")
    cal_parser.add_argument("--tf", type=int, default=5)
    cal_parser.add_argument("--bars", type=int, default=2000)
    cal_parser.add_argument("--save", action="store_true")

    opt_parser = subparsers.add_parser("optimize", help="Grid search optimal parameters")
    opt_parser.add_argument("--data", type=str, default=None)
    opt_parser.add_argument("--mt5", action="store_true")
    opt_parser.add_argument("--tf", type=int, default=5)
    opt_parser.add_argument("--bars", type=int, default=3000)

    live_parser = subparsers.add_parser("live", help="Start live trading on MT5")
    live_parser.add_argument("--tf", type=int, default=5)
    live_parser.add_argument("--bars", type=int, default=2000)
    live_parser.add_argument("--hmm-gate", type=float, default=0.70)
    live_parser.add_argument("--interval", type=int, default=60, help="Update interval (seconds)")

    dash_parser = subparsers.add_parser("dashboard", help="Launch real-time monitoring dashboard")
    dash_parser.add_argument("--port", type=int, default=8050)
    dash_parser.add_argument("--host", type=str, default="127.0.0.1")
    dash_parser.add_argument("--interval", type=int, default=5000, help="Refresh interval (ms)")

    args = parser.parse_args()

    commands = {
        "fetch": cmd_fetch,
        "generate": cmd_generate,
        "backtest": cmd_backtest,
        "calibrate": cmd_calibrate,
        "optimize": cmd_optimize,
        "live": cmd_live,
        "dashboard": cmd_dashboard,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()
        print("\nQuick start:")
        print("  python main.py fetch --tf 5                    # Get real GOLD-Pro from MT5")
        print("  python main.py calibrate --tf 5                # Analyze market stats")
        print("  python main.py backtest --mt5 --tf 5           # Backtest on real data")
        print("  python main.py optimize --mt5 --tf 5           # Find best parameters")
        print("  python main.py dashboard                       # Launch monitoring dashboard")
        print("  python main.py live --tf 5                     # Start live trading")


if __name__ == "__main__":
    main()
