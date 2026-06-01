import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import json
import logging
from typing import Optional, Dict, Tuple

from live.mt5_adapter import MT5Connector
from signals.generator import SignalGenerator
from risk.kelly import kelly_fraction, position_size, calculate_trade_stats, BayesianKelly
from risk.exits import combined_exit
from risk.stops import atr_from_df
from config.settings import GoldConfig, GOLD_CONFIG

logger = logging.getLogger(__name__)


def _get_period_pnl(history: list, start_dt: datetime, end_dt: datetime) -> tuple:
    """Filter trades within a date range and compute stats."""
    period_trades = [
        t for t in history
        if start_dt <= t.get("exit_time", datetime.min) <= end_dt
    ]
    if not period_trades:
        return 0, 0, 0.0, 0.0, 0.0
    wins = [t for t in period_trades if t["pnl"] > 0]
    losses = [t for t in period_trades if t["pnl"] <= 0]
    total_pnl = sum(t["pnl"] for t in period_trades)
    wr = len(wins) / len(period_trades) * 100 if period_trades else 0
    avg_trade = total_pnl / len(period_trades)
    return len(period_trades), len(wins), wr, total_pnl, avg_trade


class LiveTrader:
    def __init__(self, config: GoldConfig = None):
        self.config = config or GOLD_CONFIG
        self.mt5 = MT5Connector(symbol=self.config.symbol.symbol)
        self.signal_gen = SignalGenerator(self.config)
        self._open_position: Optional[Dict] = None
        self._trade_history: list = []
        self._running = False
        self._entry_bar_index = 0
        self._current_bar_index = 0
        self._df: Optional[pd.DataFrame] = None
        self._obsidian = None
        self._trade_counter = 0
        self._bayesian_kelly = BayesianKelly(
            prior_win_rate=getattr(self.config.risk, "bayesian_prior_win_rate", 0.55),
            prior_strength=getattr(self.config.risk, "bayesian_prior_strength", 20.0),
            prior_payoff=getattr(self.config.risk, "bayesian_prior_payoff", 1.2),
            prior_payoff_strength=getattr(self.config.risk, "bayesian_prior_payoff_strength", 20.0),
        )
        try:
            from obsidian_sync import get_writer
            self._obsidian = get_writer(getattr(self.config, "obsidian", None))
        except Exception as e:
            logger.debug("Obsidian writer not initialized: %s", e)

    def connect(self) -> bool:
        if not self.mt5.connect():
            print("Cannot connect to MT5. Ensure MetaTrader 5 terminal is running.")
            return False

        symbol_info = self.mt5.get_symbol_info()
        print(f"Connected to {symbol_info.get('symbol', 'N/A')}")
        print(f"  Spread: {symbol_info.get('spread', 'N/A')} points")
        print(f"  Min Volume: {symbol_info.get('volume_min', 'N/A')}")
        print(f"  Digits: {symbol_info.get('digits', 'N/A')}")
        return True

    def load_history(self, timeframe: int = None, bars: int = None) -> pd.DataFrame:
        tf = timeframe or self.config.timeframe.primary
        count = bars or max(self.config.window.rolling_zscore * 3, self.config.window.hmm_training * 2)
        print(f"Fetching {count} bars of {tf}-minute {self.config.symbol.symbol} data...")
        self._df = self.mt5.fetch_rates(timeframe=tf, count=count)
        print(f"Loaded {len(self._df)} bars. Range: {self._df.index[0]} to {self._df.index[-1]}")
        return self._df

    def calibrate_hmm(self):
        if self._df is None:
            self.load_history()

        close = self._df["close"].values.astype(np.float64)
        returns = np.diff(close) / close[:-1]
        returns = np.where(np.isfinite(returns), returns, 0.0)

        from stats.hmm import HMMRegimeDetector
        self.signal_gen.hmm = HMMRegimeDetector(n_states=2)
        self.signal_gen.hmm.fit(returns)
        print(f"HMM calibrated. Ranging state: {self.signal_gen.hmm._ranging_state}")
        self.signal_gen._returns = returns

    def compute_live_features(self) -> pd.DataFrame:
        if self._df is None or len(self._df) < self.config.window.rolling_zscore + 1:
            print(f"Not enough data for feature computation. Need {self.config.window.rolling_zscore + 1} bars.")
            return pd.DataFrame()

        df_htf = None
        if self.config.threshold.multi_tf_enabled:
            try:
                from data.mtf import aggregate_from_minutes
                df_htf = aggregate_from_minutes(
                    self._df, minutes=self.config.threshold.multi_tf_minutes
                )
            except Exception as e:
                logger.debug("HTF aggregation failed: %s", e)
                df_htf = None

        features = self.signal_gen.compute_and_generate(self._df, df_htf=df_htf)

        self._current_bar_index = len(features) - 1
        self._last_htf_hurst = features.attrs.get("htf_hurst", float("nan"))
        return features

    def get_latest_signal(self, features: pd.DataFrame) -> Tuple[int, float, float, float]:
        if features.empty:
            return 0, 0.0, 0.0, 0.0

        latest = features.iloc[-1]
        return (
            int(latest.get("signal", 0)),
            float(latest.get("zscore", 0.0)),
            float(latest.get("hurst", 0.5)),
            float(latest.get("hmm_ranging_prob", 0.5)),
        )

    def get_entry_remark(self, signal: int, zscore: float, hurst: float,
                         hmm_prob: float = 0.5) -> str:
        """Generate a human-readable remark for why we entered."""
        remarks = []
        if signal == 1:
            remarks.append("Mean-reverting LONG")
        else:
            remarks.append("Mean-reverting SHORT")
        remarks.append(f"H={hurst:.3f}")
        remarks.append(f"Z={zscore:+.2f}")
        if hmm_prob >= 0.5:
            remarks.append(f"HMM ranging={hmm_prob*100:.0f}%")
        else:
            remarks.append(f"HMM trending={100-hmm_prob*100:.0f}%")
        return " | ".join(remarks)

    def get_exit_remark(self, zscore: float, hurst: float, reason: str) -> str:
        """Generate a human-readable remark for why we exited."""
        if "zscore_trail" in reason:
            return f"Z-trail triggered | Z={zscore:+.2f} | H={hurst:.3f}"
        elif "time" in reason:
            return f"Time stop | Z={zscore:+.2f} | H={hurst:.3f}"
        elif "hurst" in reason:
            return f"Hurst regime flip | Z={zscore:+.2f} | H={hurst:.3f}"
        else:
            return f"{reason} | Z={zscore:+.2f} | H={hurst:.3f}"

    def calculate_kelly_size(self) -> float:
        if len(self._trade_history) < 10:
            return 0.0

        wins = [t["pnl"] for t in self._trade_history if t["pnl"] > 0]
        losses = [t["pnl"] for t in self._trade_history if t["pnl"] <= 0]

        if not wins or not losses:
            return 0.0

        win_rate = len(wins) / len(self._trade_history)
        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))

        if avg_loss < 1e-10:
            return 0.0

        kf = kelly_fraction(win_rate, avg_win, avg_loss)
        kf = kf * self.config.risk.kelly_fraction

        return kf

    def calculate_bayesian_kelly_size(self) -> float:
        """Bayesian Kelly — uses Beta-Binomial + log-normal conjugate priors.

        Robust to small samples (shrinks to prior) and adapts as data
        accumulates. Returns fraction scaled by `kelly_fraction`.
        """
        diag = self._bayesian_kelly.diagnostic()
        kf = diag["kelly_shrunk"]
        kf = kf * self.config.risk.kelly_fraction
        return float(max(0.0, kf))

    def _feed_bayesian_kelly(self, pnl: float) -> None:
        try:
            self._bayesian_kelly.update(float(pnl))
        except Exception as e:
            logger.debug("bayesian kelly update failed: %s", e)

    def bayesian_kelly_diagnostic(self) -> dict:
        return self._bayesian_kelly.diagnostic()

    def execute_signal(self, signal: int, zscore: float, hmm_prob: float = 0.5, double_lot: bool = False) -> Optional[Dict]:
        if self._open_position is not None:
            return None

        if signal == 0:
            return None

        account = self.mt5.get_account_info()
        equity = account.get("equity", 0.0)
        if equity <= 0:
            print("Cannot execute: account equity is zero or unavailable")
            return None

        if getattr(self.config.risk, "use_bayesian_kelly", True):
            kf = self.calculate_bayesian_kelly_size()
        else:
            kf = self.calculate_kelly_size()
        if kf <= 0:
            kf = 0.01

        if self._df is not None:
            atr_vals = atr_from_df(self._df, period=14)
            current_atr = atr_vals[-1] if not np.isnan(atr_vals[-1]) else self._df["close"].iloc[-1] * 0.005
        else:
            current_atr = self._df["close"].iloc[-1] * 0.005

        pos_size = position_size(
            account_equity=equity,
            kelly_f=kf,
            atr=current_atr,
            risk_pct=self.config.risk.account_risk_pct,
        )

        # HIGH CONFIDENCE: double the position size
        if double_lot:
            pos_size = pos_size * 2.0
            print(f"  *** HIGH CONFIDENCE: Doubling position size ***")

        symbol_info = self.mt5.get_symbol_info()
        min_vol = symbol_info.get("volume_min", 0.01)
        vol_step = symbol_info.get("volume_step", 0.01)
        pos_size = max(min_vol, round(pos_size / vol_step) * vol_step)
        pos_size = round(pos_size, 2)

        bid, ask = self.mt5.get_current_price()

        zscore_sl = (abs(zscore) - abs(self.config.threshold.zscore_stop_long)) / abs(zscore) if abs(zscore) > 0 else 0

        if signal == 1:
            order_type = "BUY"
            price = ask
            sl = price - (current_atr * self.config.risk.atr_multiplier_sl)
        else:
            order_type = "SELL"
            price = bid
            sl = price + (current_atr * self.config.risk.atr_multiplier_sl)

        print(f"\n{'=' * 50}")
        print(f"  SIGNAL: {order_type}  |  Z-score: {zscore:.2f}")
        if double_lot:
            print(f"  *** HIGH CONFIDENCE DOUBLE LOT ***")
        print(f"  Price: {price:.2f}  |  Size: {pos_size} lots  |  SL: {sl:.2f}")
        print(f"  Kelly f*: {kf:.4f}  |  Account: ${equity:.0f}")
        print(f"{'=' * 50}")

        ticket = self.mt5.place_order(
            order_type=order_type,
            volume=pos_size,
            price=price,
            sl=sl,
        )

        if ticket is None:
            print("Order execution failed!")
            return None

        self._open_position = {
            "ticket": ticket,
            "direction": signal,
            "entry_price": price,
            "entry_zscore": zscore,
            "entry_bar": self._current_bar_index,
            "volume": pos_size,
            "sl": sl,
            "entry_time": datetime.now(),
            "best_zscore": abs(zscore),
            "entry_hurst": self._df.get("hurst", pd.Series([0.5])).iloc[-1] if self._df is not None else 0.5,
            "entry_hmm": self._last_remark,
            "htf_hurst": getattr(self, "_last_htf_hurst", float("nan")),
        }

        self._entry_bar_index = self._current_bar_index
        remark = self.get_entry_remark(signal, zscore, self._open_position["entry_hurst"], hmm_prob=hmm_prob)
        self._last_remark = remark
        print(f"  Remark: {remark}")
        print(f"Order #{ticket} executed.")

        if self._obsidian is not None:
            try:
                self._trade_counter += 1
                atr_ratio = (
                    float(self._df["atr_ratio"].iloc[-1])
                    if self._df is not None and "atr_ratio" in self._df.columns
                    else 0.0
                )
                session = self._current_session_label()
                from obsidian_sync.helpers import signal_from_engine
                signal_from_engine(
                    trade_id=ticket,
                    direction="long" if signal == 1 else "short",
                    symbol=self.config.symbol.symbol,
                    price=price,
                    zscore=zscore,
                    atr_ratio=atr_ratio,
                    session=session,
                    filled=True,
                    notes=remark,
                )
                self._obsidian.append_dashboard_log(
                    "entry filled",
                    f"{'LONG' if signal == 1 else 'SHORT'} {self.config.symbol.symbol} @ {price:.2f} Z={zscore:+.2f}",
                )
            except Exception as e:
                logger.debug("obsidian entry write failed: %s", e)

        return self._open_position

    def _current_session_label(self) -> str:
        try:
            from datetime import datetime, timezone
            h = datetime.now(timezone.utc).hour
            sh = self.config.threshold.session_start_hour
            eh = self.config.threshold.session_end_hour
            if sh <= h < eh:
                if 13 <= h < 17:
                    return "London/NY"
                if h < 13:
                    return "London"
                return "NewYork"
            return "off-session"
        except Exception:
            return "unknown"

    def check_exits(self, zscore: float, hurst: float = 0.5) -> Optional[Dict]:
        if self._open_position is None:
            return None

        pos = self._open_position
        direction = pos["direction"]
        entry_z = pos["entry_zscore"]
        current_abs_z = abs(zscore)

        # ── 1. Z-Score trailing stop (statistical) ──
        # Update best Z reached
        if current_abs_z > pos.get("best_zscore", abs(entry_z)):
            pos["best_zscore"] = current_abs_z

        best_z = pos.get("best_zscore", abs(entry_z))
        retrace_threshold = best_z * (1 - self.config.threshold.zscore_trail_retrace)

        if direction == 1 and zscore > 0:  # Long, Z flipping positive
            if current_abs_z < retrace_threshold:
                return self._exit_position(zscore, hurst, "zscore_trail_retrace")
        elif direction == -1 and zscore < 0:  # Short, Z flipping negative
            if current_abs_z < retrace_threshold:
                return self._exit_position(zscore, hurst, "zscore_trail_retrace")

        # ── 2. Hurst emergency exit (regime flip) ──
        if hurst > self.config.threshold.hurst_exit_threshold:
            entry_hurst = pos.get("entry_hurst", 0.5)
            if entry_hurst < self.config.threshold.hurst_mean_revert and hurst > self.config.threshold.hurst_exit_threshold:
                return self._exit_position(zscore, hurst, f"hurst_regime_flip_{hurst:.2f}")

        # ── 3. Original static Z-score stop (safety net) ──
        should_exit, reason, exit_z = combined_exit(
            current_zscore=zscore,
            entry_zscore=entry_z,
            bar_index=self._current_bar_index,
            entry_bar=pos["entry_bar"],
            signal_direction=direction,
            max_bars=self.config.threshold.time_stop_bars,
            zscore_stop_long=self.config.threshold.zscore_stop_long,
            zscore_stop_short=self.config.threshold.zscore_stop_short,
        )

        if should_exit:
            return self._exit_position(zscore, hurst, reason)

        return None

    def _exit_position(self, zscore: float, hurst: float = 0.5, reason: str = "") -> Optional[Dict]:
        """Close the current position."""
        remark = self.get_exit_remark(zscore, hurst, reason)
        pos = self.mt5.get_positions()
        matching = [p for p in pos if p["ticket"] == self._open_position["ticket"]]
        if not matching:
            logger.warning(f"Position #{self._open_position['ticket']} not found. Marking as closed.")
            self._record_trade(self._open_position["entry_price"], self.mt5.get_current_price()[0], remark)
            closed = dict(self._open_position)
            self._open_position = None
            return closed

        logger.info(f"EXIT: {remark} | Z-score: {zscore:.2f}")
        success = self.mt5.close_position(self._open_position["ticket"])
        if success:
            pnl = matching[0]["profit"]
            exit_price = matching[0]["current_price"]
            trade_record = {
                "direction": self._open_position["direction"],
                "entry_price": self._open_position["entry_price"],
                "exit_price": exit_price,
                "pnl": pnl,
                "reason": reason,
                "remark": remark,
                "entry_time": self._open_position["entry_time"],
                "exit_time": datetime.now(),
            }
            self._trade_history.append(trade_record)
            self._feed_bayesian_kelly(pnl)
            logger.info(f"Position closed. PnL: ${pnl:.2f}")

            if self._obsidian is not None:
                try:
                    from obsidian_sync.helpers import trade_from_engine
                    direction = self._open_position["direction"]
                    entry = self._open_position["entry_price"]
                    pips = (exit_price - entry) * direction * 100
                    duration = max(0, self._current_bar_index - self._open_position.get("entry_bar", self._current_bar_index))
                    trade_from_engine(
                        trade_id=self._open_position.get("ticket"),
                        direction="long" if direction == 1 else "short",
                        symbol=self.config.symbol.symbol,
                        entry=entry,
                        exit_price=exit_price,
                        stop=self._open_position.get("sl"),
                        pnl_usd=pnl,
                        pnl_pips=pips,
                        zscore_entry=self._open_position.get("entry_zscore", 0.0),
                        zscore_exit=zscore,
                        duration_bars=duration,
                        session=self._current_session_label(),
                        exit_reason=reason,
                        open_time=self._open_position.get("entry_time"),
                        close_time=trade_record["exit_time"],
                        notes=remark,
                    )
                except Exception as e:
                    logger.debug("obsidian trade write failed: %s", e)

        closed_pos = dict(self._open_position)
        self._open_position = None
        return closed_pos

    def _record_trade(self, entry_price: float, exit_price: float, remark: str = ""):
        if self._open_position is None:
            return
        pnl = (exit_price - entry_price) * self._open_position["direction"] * self._open_position["volume"] * 100
        self._trade_history.append({
            "direction": self._open_position["direction"],
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl": pnl,
            "reason": "manual",
            "remark": remark,
            "entry_time": self._open_position["entry_time"],
            "exit_time": datetime.now(),
        })

    def run_once(self, double_lot: bool = False) -> Tuple[int, float, float, float, float]:
        """
        Run one iteration of the live trading loop.
        Returns (signal, zscore, hurst, hmm_prob).
        """
        self._df = self.mt5.fetch_rates(
            timeframe=self.config.timeframe.primary,
            count=max(self.config.window.rolling_zscore * 2, 500),
        )

        features = self.compute_live_features()
        signal, zscore, hurst, hmm_prob = self.get_latest_signal(features)

        self.check_exits(zscore, hurst)
        self.execute_signal(signal, zscore, hmm_prob=hmm_prob, double_lot=double_lot)

        return signal, zscore, hurst, hmm_prob

    def run_loop(self, sleep_seconds: int = 60, double_lot: bool = False):
        self._running = True
        print(f"\n{'=' * 60}")
        print(f"  LIVE TRADING STARTED: {self.config.symbol.symbol}")
        print(f"  Timeframe: {self.config.timeframe.primary}min")
        print(f"  HMM gate: {self.config.threshold.hmm_ranging_prob}")
        print(f"  Update interval: {sleep_seconds}s")
        if double_lot:
            print(f"  *** HIGH CONFIDENCE MODE: Double lot enabled ***")
        print(f"{'=' * 60}\n")

        if self._obsidian is not None:
            try:
                self._obsidian.start()
                self._obsidian.append_dashboard_log(
                    "system started",
                    f"TF={self.config.timeframe.primary}m interval={sleep_seconds}s",
                )
            except Exception as e:
                logger.debug("obsidian start failed: %s", e)

        try:
            while self._running:
                try:
                    signal, zscore, hurst, hmm_prob = self.run_once(double_lot=double_lot)

                    status = "LONG" if signal == 1 else "SHORT" if signal == -1 else "WAIT"
                    pos = "IN TRADE" if self._open_position else "FLAT"
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                          f"Signal: {status:>5} | Z: {zscore:>+.2f} | H: {hurst:.3f} | HMM: {hmm_prob:.0%} | {pos}")

                except Exception as e:
                    print(f"Error in loop iteration: {e}")
                    time.sleep(sleep_seconds)
                    continue

                time.sleep(sleep_seconds)

        except KeyboardInterrupt:
            print("\n\nInterrupted. Closing all positions...")
            self._running = False
            self.mt5.close_all_positions()
            self.mt5.disconnect(force=False)
            if self._obsidian is not None:
                try:
                    self._obsidian.append_dashboard_log("system stopped", "KeyboardInterrupt")
                    self._obsidian.stop()
                except Exception:
                    pass

    def stop(self):
        self._running = False
        if self._open_position:
            self.mt5.close_all_positions()
        # Use soft disconnect so manual account switches in MT5 take effect immediately
        self.mt5.disconnect(force=False)
        if self._obsidian is not None:
            try:
                self._obsidian.append_dashboard_log("system stopped", "stop() called")
                self._obsidian.stop()
            except Exception:
                pass

    def daily_performance(self) -> str:
        """Return today's performance summary."""
        now = datetime.now()
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
        n, nw, wr, pnl, avg = _get_period_pnl(self._trade_history, start, end)
        return f"📅 TODAY: {n} trades | {nw} wins | {wr:.0f}% WR | ${pnl:+.2f}"

    def weekly_performance(self) -> str:
        """Return this week's performance summary."""
        now = datetime.now()
        start = now - timedelta(days=now.weekday())
        start = start.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=7)
        n, nw, wr, pnl, avg = _get_period_pnl(self._trade_history, start, end)
        return f"📊 THIS WEEK: {n} trades | {nw} wins | {wr:.0f}% WR | ${pnl:+.2f}"

    def monthly_performance(self) -> str:
        """Return this month's performance summary."""
        now = datetime.now()
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if start.month < 12:
            end = start.replace(month=start.month + 1)
        else:
            end = start.replace(year=start.year + 1, month=1)
        n, nw, wr, pnl, avg = _get_period_pnl(self._trade_history, start, end)
        return f"📈 THIS MONTH: {n} trades | {nw} wins | {wr:.0f}% WR | ${pnl:+.2f}"

    def trade_summary(self) -> str:
        if not self._trade_history:
            return "No trades recorded."

        trades = self._trade_history
        n = len(trades)
        wins = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] <= 0]
        wr = len(wins) / n if n > 0 else 0
        total_pnl = sum(t["pnl"] for t in trades)
        avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
        avg_loss = np.mean([t["pnl"] for t in losses]) if losses else 0
        pf = abs(sum(t["pnl"] for t in wins) / sum(t["pnl"] for t in losses)) if losses and sum(t["pnl"] for t in losses) != 0 else 0

        lines = []
        lines.append("=" * 50)
        lines.append("  TRADE SUMMARY")
        lines.append("=" * 50)
        lines.append(f"  Total Trades:  {n}")
        lines.append(f"  Win Rate:      {wr:.1%}")
        lines.append(f"  Total PnL:     ${total_pnl:.2f}")
        lines.append(f"  Avg Win:       ${avg_win:.2f}")
        lines.append(f"  Avg Loss:      ${avg_loss:.2f}")
        lines.append(f"  Profit Factor: {pf:.2f}")
        lines.append(self.daily_performance())
        lines.append(self.weekly_performance())
        lines.append(self.monthly_performance())
        lines.append("=" * 50)
        return "\n".join(lines)
