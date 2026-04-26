import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Optional

from live.mt5_adapter import MT5Connector
from live.rithmic_adapter import RithmicAdapter
from signals.generator_v2 import SignalGeneratorV2
from orderflow.sl_zones import detect_sl_zones
from orderflow.stop_hunt import detect_stop_hunt
from risk.stops import atr_from_df
from config.settings import GOLD_CONFIG
from config.settings_v2 import V2Config
from dashboard.layout_v2 import POSITIVE_COLOR, NEGATIVE_COLOR, NEUTRAL_COLOR, BUY_COLOR, SELL_COLOR


class DashboardDataProviderV2:
    def __init__(self, config: V2Config = None):
        self.config = config or V2Config()
        self.mt5 = MT5Connector(symbol=self.config.symbol.symbol)
        self.rithmic = RithmicAdapter(config=self.config)
        self.signal_gen = SignalGeneratorV2(config=self.config)
        self._df: Optional[pd.DataFrame] = None
        self._features: Optional[pd.DataFrame] = None
        self._connected_mt5 = False
        self._connected_rithmic = False
        self._equity_history: list = []
        self._initial_equity = None
        self._last_trigger = None

    def connect_all(self) -> bool:
        self._connected_mt5 = self.mt5.connect()
        self._connected_rithmic = self.rithmic.connect()
        if self._connected_mt5:
            acc = self.mt5.get_account_info()
            self._initial_equity = acc.get("equity", 10000)
        return self._connected_mt5 or self._connected_rithmic

    def disconnect_all(self):
        self.mt5.disconnect()
        self.rithmic.disconnect()

    @property
    def connected(self) -> bool:
        return self._connected_mt5

    def refresh(self) -> Dict:
        if not self._connected_mt5:
            return self._empty_state()

        self._df = self.mt5.fetch_rates(
            timeframe=self.config.timeframe.primary,
            count=max(self.config.window.rolling_zscore * 2, 300),
        )

        close = self._df["close"].values.astype(np.float64)
        n = len(close)

        self._features = self.signal_gen.compute_and_generate(self._df)
        latest = self._features.iloc[-1]

        v1_signal = int(latest.get("signal", 0))
        zscore = float(latest.get("zscore", np.nan))
        hurst = float(latest.get("hurst", np.nan))

        returns = np.diff(close) / (close[:-1] + 1e-10)
        atr_val = float(np.std(self._df["high"] - self._df["low"]))
        if atr_val < 0.01:
            atr_val = close[-1] * 0.005

        sl_zones = detect_sl_zones(
            close, lookback=self.config.orderflow.swing_lookback,
            swing_fractal=5, round_digits=self.config.orderflow.sl_round_number_digits,
            atr=atr_val, sl_zone_atr_distance=self.config.orderflow.sl_zone_atr_distance,
            sl_density_threshold=self.config.orderflow.sl_density_threshold,
        )

        stop_hunt = detect_stop_hunt(
            close, sl_zones["sl_zone_below"], sl_zones["sl_zone_above"],
            sl_zones["sl_density_below"], sl_zones["sl_density_above"],
            atr_val, self.config.orderflow.swing_lookback,
            self.config.orderflow.sweep_depth_pct, self.config.orderflow.stop_hunt_min_score,
        )

        price_dir = int(np.sign(close[-1] - close[-2])) if n > 1 else 0
        self.rithmic.get_snapshot(mid_price=float(close[-1]), bar_direction=price_dir)
        of_state = self.rithmic.get_orderflow_state()

        dom_ok = True
        if v1_signal == 1:
            if of_state["ofi"] < 0:
                dom_ok = False
            if of_state["bid_ask_ratio"] < self.config.orderflow.bid_ask_ratio_long:
                dom_ok = False
            if of_state.get("absorption_ask"):
                dom_ok = False
            if of_state.get("iceberg_ask_detected"):
                dom_ok = False
            if stop_hunt.get("sweep_direction") == +1 and stop_hunt.get("high_confidence"):
                dom_ok = True
        elif v1_signal == -1:
            if of_state["ofi"] > 0:
                dom_ok = False
            if of_state["bid_ask_ratio"] > self.config.orderflow.bid_ask_ratio_short:
                dom_ok = False
            if of_state.get("absorption_bid"):
                dom_ok = False
            if of_state.get("iceberg_bid_detected"):
                dom_ok = False
            if stop_hunt.get("sweep_direction") == -1 and stop_hunt.get("high_confidence"):
                dom_ok = True
        elif v1_signal == 0:
            dom_ok = True

        bid, ask = self.mt5.get_current_price()
        account = self.mt5.get_account_info()
        positions = self.mt5.get_positions()

        eq = account.get("equity", self._initial_equity or 10000)
        self._equity_history.append({"time": datetime.now(), "equity": eq})
        if len(self._equity_history) > 200:
            self._equity_history = self._equity_history[-200:]

        total_pnl = eq - (self._initial_equity or eq)
        has_pos = len(positions) > 0
        pos = positions[0] if has_pos else None

        bid_ask_ratio = of_state.get("bid_ask_ratio", 0.5)
        bid_ratio_pct = bid_ask_ratio * 100
        ask_ratio_pct = (1 - bid_ask_ratio) * 100

        dom_ladder = {
            "bids": [(b.price, b.volume) for b in self.rithmic._snapshots[-1].bids[:10]]
                     if self.rithmic._snapshots and self.rithmic._snapshots[-1].bids else [],
            "asks": [(a.price, a.volume) for a in self.rithmic._snapshots[-1].asks[:10]]
                     if self.rithmic._snapshots and self.rithmic._snapshots[-1].asks else [],
        }

        signal_text = "BUY" if v1_signal == 1 and dom_ok else "BUY*" if v1_signal == 1 else \
                      "SELL" if v1_signal == -1 and dom_ok else "SELL*" if v1_signal == -1 else "WAIT"
        sig_color = BUY_COLOR if v1_signal == 1 else SELL_COLOR if v1_signal == -1 else NEUTRAL_COLOR
        dom_validation = "CONFIRMED" if (v1_signal != 0 and dom_ok) else \
                         "REJECTED" if (v1_signal != 0 and not dom_ok) else "IDLE"
        dom_color = POSITIVE_COLOR if dom_validation == "CONFIRMED" else \
                    NEGATIVE_COLOR if dom_validation == "REJECTED" else NEUTRAL_COLOR

        return {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "connected_mt5": self._connected_mt5,
            "connected_rithmic": self._connected_rithmic,
            "using_synthetic": self.rithmic.is_synthetic,
            "symbol": self.config.symbol.symbol,
            "timeframe": self.config.timeframe.primary,
            "bid": bid, "ask": ask,
            "signal": v1_signal, "signal_text": signal_text, "signal_color": sig_color,
            "dom_validation": dom_validation, "dom_color": dom_color, "dom_ok": dom_ok,
            "zscore": zscore, "hurst": hurst,
            "ofi": of_state.get("ofi", 0), "ofi_sma": of_state.get("ofi_sma", 0),
            "cum_delta": of_state.get("cum_delta", 0), "delta_per_bar": of_state.get("delta_per_bar", 0),
            "bid_ask_ratio": bid_ask_ratio, "bid_ratio_pct": bid_ratio_pct, "ask_ratio_pct": ask_ratio_pct,
            "absorption_bid": of_state.get("absorption_bid", False),
            "absorption_ask": of_state.get("absorption_ask", False),
            "iceberg_bid": of_state.get("iceberg_bid", 0),
            "iceberg_ask": of_state.get("iceberg_ask", 0),
            "iceberg_bid_detected": of_state.get("iceberg_bid_detected", False),
            "iceberg_ask_detected": of_state.get("iceberg_ask_detected", False),
            "iceberg_persistence": of_state.get("iceberg_persistence", 0),
            "iceberg_confidence": of_state.get("iceberg_confidence", 0),
            "vpoc": of_state.get("vpoc", 0),
            "equity": eq, "total_pnl": total_pnl,
            "has_position": has_pos,
            "position_type": pos["type"] if pos else "",
            "position_volume": pos["volume"] if pos else 0,
            "position_pnl": pos["profit"] if pos else 0,
            "sl_zone_below": sl_zones["sl_zone_below"],
            "sl_zone_above": sl_zones["sl_zone_above"],
            "sl_dist_below": sl_zones["sl_dist_below"],
            "sl_dist_above": sl_zones["sl_dist_above"],
            "sl_density_below": sl_zones["sl_density_below"],
            "sl_density_above": sl_zones["sl_density_above"],
            "sweep_detected": stop_hunt["sweep_detected"],
            "sweep_direction": stop_hunt["sweep_direction"],
            "sweep_reversal": stop_hunt["sweep_reversal"],
            "stop_hunt_score": stop_hunt["stop_hunt_score"],
            "stop_hunt_high_conf": stop_hunt["high_confidence"],
            "dom_ladder": dom_ladder,
            "atr": atr_val,
            "bar_count": n,
            "swing_highs": sl_zones.get("swing_highs", []),
            "swing_lows": sl_zones.get("swing_lows", []),
            "round_numbers": sl_zones.get("round_numbers", []),
            "price_history": {
                "time": self._df.index.strftime("%Y-%m-%d %H:%M:%S").tolist()[-150:],
                "close": close[-150:].tolist(),
            },
            "zscore_history": {
                "time": self._df.index.strftime("%Y-%m-%d %H:%M:%S").tolist()[-150:],
                "zscore": self._features["zscore"].values[-150:].tolist(),
                "hurst": self._features["hurst"].values[-150:].tolist(),
            },
            "signal_markers": {
                "buy_time": self._features.index[self._features.get("signal", 0) == 1].strftime("%Y-%m-%d %H:%M:%S").tolist()[-15:],
                "buy_price": close[self._features.get("signal", 0) == 1][-15:].tolist(),
            },
            "equity_history": {
                "time": [e["time"].strftime("%H:%M:%S") for e in self._equity_history],
                "equity": [e["equity"] for e in self._equity_history],
            },
        }

    def _empty_state(self) -> Dict:
        return {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "connected_mt5": False, "connected_rithmic": False,
            "using_synthetic": True, "dom_validation": "DISCONNECTED",
            "dom_color": NEGATIVE_COLOR, "signal_color": NEUTRAL_COLOR,
            "signal_text": "N/A",
        }
