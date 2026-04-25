import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Optional

from live.mt5_adapter import MT5Connector
from signals.generator import SignalGenerator
from config.settings import GOLD_CONFIG


class DashboardDataProvider:
    def __init__(self, config=None):
        self.config = config or GOLD_CONFIG
        self.mt5 = MT5Connector(symbol=self.config.symbol.symbol)
        self.signal_gen = SignalGenerator(self.config)
        self._df: Optional[pd.DataFrame] = None
        self._features: Optional[pd.DataFrame] = None
        self._connected = False
        self._trade_history: list = []
        self._equity_history: list = []
        self._initial_equity = None

    def connect(self) -> bool:
        self._connected = self.mt5.connect()
        if self._connected:
            acc = self.mt5.get_account_info()
            self._initial_equity = acc.get("equity", 10000)
            self._equity_history = [{"time": datetime.now(), "equity": self._initial_equity}]
        return self._connected

    def disconnect(self):
        self.mt5.disconnect()
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    def refresh(self) -> Dict:
        if not self._connected:
            return self._empty_state()

        self._df = self.mt5.fetch_rates(
            timeframe=self.config.timeframe.primary,
            count=max(self.config.window.rolling_zscore * 2, 300),
        )

        close = self._df["close"].values.astype(np.float64)
        n = len(close)

        self._features = self.signal_gen.compute_and_generate(self._df)
        latest = self._features.iloc[-1]

        signal = int(latest.get("signal", 0))
        zscore = float(latest.get("zscore", np.nan))
        hurst = float(latest.get("hurst", np.nan))
        velocity = float(latest.get("velocity", np.nan))
        hmm_prob = float(latest.get("hmm_ranging_prob", 0.0))

        bid, ask = self.mt5.get_current_price()
        spread = self.mt5.get_spread()
        symbol_info = self.mt5.get_symbol_info()
        positions = self.mt5.get_positions()
        account = self.mt5.get_account_info()

        current_equity = account.get("equity", self._initial_equity or 10000)
        if not self._equity_history or self._equity_history[-1]["equity"] != current_equity:
            self._equity_history.append({"time": datetime.now(), "equity": current_equity})

        if len(self._equity_history) > 500:
            self._equity_history = self._equity_history[-500:]

        total_pnl = current_equity - (self._initial_equity or current_equity)

        equity_from_start = self._equity_history[0]["equity"]
        pnl_pct = (current_equity / equity_from_start - 1) * 100 if equity_from_start else 0

        has_position = len(positions) > 0
        pos_info = positions[0] if has_position else None

        signal_text = "BUY" if signal == 1 else "SELL" if signal == -1 else "WAIT"
        hurst_regime = "M-REVERT" if not np.isnan(hurst) and hurst < self.config.threshold.hurst_mean_revert else (
            "TRENDING" if not np.isnan(hurst) and hurst > 0.55 else "RANDOM")
        zscore_signal = "OVERSOLD" if zscore < -2.5 else "OVERBOUGHT" if zscore > 2.5 else "NEUTRAL"

        atr = float(np.std(self._df["high"] - self._df["low"]))

        return {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "connected": True,
            "symbol": self.config.symbol.symbol,
            "timeframe": self.config.timeframe.primary,
            "bid": bid,
            "ask": ask,
            "spread": spread,
            "digits": symbol_info.get("digits", 5),
            "signal": signal,
            "signal_text": signal_text,
            "zscore": zscore,
            "hurst": hurst,
            "hurst_regime": hurst_regime,
            "velocity": velocity,
            "hmm_ranging_prob": hmm_prob,
            "zscore_signal": zscore_signal,
            "account_balance": account.get("balance", 0),
            "account_equity": current_equity,
            "free_margin": account.get("free_margin", 0),
            "total_pnl": total_pnl,
            "pnl_pct": pnl_pct,
            "has_position": has_position,
            "position_type": pos_info["type"] if pos_info else "",
            "position_volume": pos_info["volume"] if pos_info else 0,
            "position_open_price": pos_info["open_price"] if pos_info else 0,
            "position_current_price": pos_info["current_price"] if pos_info else 0,
            "position_pnl": pos_info["profit"] if pos_info else 0,
            "atr": atr,
            "bar_count": n,
            "price_history": {
                "time": self._df.index.strftime("%Y-%m-%d %H:%M:%S").tolist()[-200:],
                "close": close[-200:].tolist(),
            },
            "zscore_history": {
                "time": self._df.index.strftime("%Y-%m-%d %H:%M:%S").tolist()[-200:],
                "zscore": self._features["zscore"].values[-200:].tolist(),
                "hurst": self._features["hurst"].values[-200:].tolist(),
            },
            "signal_markers": {
                "time": self._features.index[self._features["signal"] == 1].strftime("%Y-%m-%d %H:%M:%S").tolist()[-20:],
                "long_price": close[self._features["signal"].values == 1][-20:].tolist(),
            },
            "equity_history": {
                "time": [e["time"].strftime("%H:%M:%S") for e in self._equity_history],
                "equity": [e["equity"] for e in self._equity_history],
            },
        }

    def _empty_state(self) -> Dict:
        return {
            "connected": False,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "symbol": self.config.symbol.symbol,
            "timeframe": self.config.timeframe.primary,
            "signal_text": "N/A",
            "hurst_regime": "N/A",
            "zscore_signal": "N/A",
        }
