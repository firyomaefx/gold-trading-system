import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
from typing import Optional, Dict, Tuple

from live.mt5_adapter import MT5Connector
from signals.generator import SignalGenerator
from risk.kelly import kelly_fraction, position_size, calculate_trade_stats
from risk.exits import combined_exit
from risk.stops import atr_from_df
from config.settings import GoldConfig, GOLD_CONFIG


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

        features = self.signal_gen.compute_and_generate(self._df)

        self._current_bar_index = len(features) - 1
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

    def execute_signal(self, signal: int, zscore: float) -> Optional[Dict]:
        if self._open_position is not None:
            return None

        if signal == 0:
            return None

        account = self.mt5.get_account_info()
        equity = account.get("equity", 0.0)
        if equity <= 0:
            print("Cannot execute: account equity is zero or unavailable")
            return None

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
        }

        self._entry_bar_index = self._current_bar_index

        print(f"Order #{ticket} executed.")
        return self._open_position

    def check_exits(self, zscore: float) -> Optional[Dict]:
        if self._open_position is None:
            return None

        should_exit, reason, exit_z = combined_exit(
            current_zscore=zscore,
            entry_zscore=self._open_position["entry_zscore"],
            bar_index=self._current_bar_index,
            entry_bar=self._open_position["entry_bar"],
            signal_direction=self._open_position["direction"],
            max_bars=self.config.threshold.time_stop_bars,
            zscore_stop_long=self.config.threshold.zscore_stop_long,
            zscore_stop_short=self.config.threshold.zscore_stop_short,
        )

        if not should_exit:
            return None

        pos = self.mt5.get_positions()
        matching = [p for p in pos if p["ticket"] == self._open_position["ticket"]]
        if not matching:
            print(f"Position #{self._open_position['ticket']} not found. Marking as closed.")
            self._record_trade(self._open_position["entry_price"], self.mt5.get_current_price()[0])
            self._open_position = None
            return None

        print(f"\n  EXIT: {reason}  |  Z-score: {zscore:.2f}")
        success = self.mt5.close_position(self._open_position["ticket"])
        if success:
            pnl = matching[0]["profit"]
            self._trade_history.append({
                "direction": self._open_position["direction"],
                "entry_price": self._open_position["entry_price"],
                "exit_price": matching[0]["current_price"],
                "pnl": pnl,
                "reason": reason,
                "entry_time": self._open_position["entry_time"],
                "exit_time": datetime.now(),
            })
            print(f"Position closed. PnL: ${pnl:.2f}")

        closed_pos = dict(self._open_position)
        self._open_position = None
        return closed_pos

    def _record_trade(self, entry_price: float, exit_price: float):
        if self._open_position is None:
            return
        pnl = (exit_price - entry_price) * self._open_position["direction"] * self._open_position["volume"] * 100
        self._trade_history.append({
            "direction": self._open_position["direction"],
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl": pnl,
            "reason": "manual",
            "entry_time": self._open_position["entry_time"],
            "exit_time": datetime.now(),
        })

    def run_once(self) -> Tuple[int, float]:
        """
        Run one iteration of the live trading loop.
        Returns (signal, zscore).
        """
        self._df = self.mt5.fetch_rates(
            timeframe=self.config.timeframe.primary,
            count=max(self.config.window.rolling_zscore * 2, 500),
        )

        features = self.compute_live_features()
        signal, zscore, hurst, hmm_prob = self.get_latest_signal(features)

        self.check_exits(zscore)
        self.execute_signal(signal, zscore)

        return signal, zscore

    def run_loop(self, sleep_seconds: int = 60):
        self._running = True
        print(f"\n{'=' * 60}")
        print(f"  LIVE TRADING STARTED: {self.config.symbol.symbol}")
        print(f"  Timeframe: {self.config.timeframe.primary}min")
        print(f"  Update interval: {sleep_seconds}s")
        print(f"{'=' * 60}\n")

        try:
            while self._running:
                try:
                    signal, zscore = self.run_once()

                    status = "LONG" if signal == 1 else "SHORT" if signal == -1 else "WAIT"
                    pos = "IN TRADE" if self._open_position else "FLAT"
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                          f"Signal: {status:>5} | Z: {zscore:>7.2f} | Status: {pos}")

                except Exception as e:
                    print(f"Error in loop iteration: {e}")
                    time.sleep(sleep_seconds)
                    continue

                time.sleep(sleep_seconds)

        except KeyboardInterrupt:
            print("\n\nInterrupted. Closing all positions...")
            self._running = False
            self.mt5.close_all_positions()
            self.mt5.disconnect()

    def stop(self):
        self._running = False
        if self._open_position:
            self.mt5.close_all_positions()
        self.mt5.disconnect()

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
        lines.append("=" * 50)
        return "\n".join(lines)
