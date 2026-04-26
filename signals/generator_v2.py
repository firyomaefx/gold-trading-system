import numpy as np
import pandas as pd
from typing import Optional, Dict, List
from datetime import datetime

from signals.generator import SignalGenerator
from config.settings_v2 import V2Config
from orderflow.dom import compute_dom_features
from orderflow.ofi import compute_ofi
from orderflow.delta import compute_delta
from orderflow.absorption import detect_absorption
from orderflow.footprint import compute_footprint
from orderflow.stop_hunt import detect_stop_hunt
from orderflow.iceberg import detect_iceberg
from orderflow.sl_zones import detect_sl_zones
from data.rithmic import DOMSnapshot
from risk.stops import atr_from_df


class SignalGeneratorV2(SignalGenerator):
    def __init__(self, config: V2Config = None):
        self.v2_config = config or V2Config()
        super().__init__(self.v2_config)
        self._dom_snapshots: List[DOMSnapshot] = []
        self._orderflow_features: Dict = {}
        self._sl_zones: Dict = {}

    def load_dom_snapshots(self, snapshots: List[DOMSnapshot]):
        self._dom_snapshots = snapshots

    def compute_orderflow_features(
        self,
        snapshots: List[DOMSnapshot],
        prices: np.ndarray,
        atr: float = 10.0,
    ) -> pd.DataFrame:

        ofc = self.v2_config.orderflow
        n = min(len(snapshots), len(prices))
        results = []

        for i in range(n):
            snap = snapshots[i] if i < len(snapshots) else None
            window_snaps = snapshots[max(0, i - 5):i + 1]

            if snap is None:
                results.append(self._empty_orderflow_row())
                continue

            dom = compute_dom_features(snap, depth=ofc.dom_depth)

            ofi = compute_ofi(window_snaps, window=ofc.ofi_window)

            delta = compute_delta(window_snaps, lookback=ofc.delta_divergence_lookback)

            absorption = detect_absorption(
                snap, prev_snapshots=window_snaps[:-1],
                multiplier=ofc.absorption_multiplier,
            )

            footprint = compute_footprint(
                window_snaps, lookback=ofc.footprint_lookback,
                current_price=float(prices[i]) if i < len(prices) else None,
            )

            iceberg = detect_iceberg(
                snap, prev_snapshots=window_snaps[:-1],
                volume_ratio=ofc.iceberg_volume_ratio,
                min_persistence=ofc.iceberg_min_persistence,
            )

            results.append({
                **dom,
                "ofi": ofi["ofi"],
                "ofi_sma": ofi["ofi_sma"],
                "cum_delta": delta["cum_delta"],
                "delta_per_bar": delta["delta_per_bar"],
                "delta_divergence": delta["delta_divergence"],
                "absorption_bid": absorption["absorption_bid"],
                "absorption_ask": absorption["absorption_ask"],
                "vpoc": footprint["vpoc"],
                "vpoc_distance": footprint["vpoc_distance"],
                "iceberg_bid": iceberg["iceberg_bid"],
                "iceberg_ask": iceberg["iceberg_ask"],
                "iceberg_bid_detected": iceberg["iceberg_bid_detected"],
                "iceberg_ask_detected": iceberg["iceberg_ask_detected"],
                "iceberg_persistence": iceberg["iceberg_persistence"],
                "iceberg_confidence": iceberg["iceberg_confidence"],
            })

        self._orderflow_features = {"results": results}
        return pd.DataFrame(results)

    def compute_sl_zones(self, prices: np.ndarray, atr: float) -> Dict:
        ofc = self.v2_config.orderflow
        self._sl_zones = detect_sl_zones(
            prices=prices,
            lookback=ofc.swing_lookback,
            swing_fractal=5,
            round_digits=ofc.sl_round_number_digits,
            atr=atr,
            sl_zone_atr_distance=ofc.sl_zone_atr_distance,
            sl_density_threshold=ofc.sl_density_threshold,
        )
        return self._sl_zones

    def compute_stop_hunt(
        self,
        prices: np.ndarray,
        sl_zones: Dict,
        atr: float,
    ) -> Dict:
        ofc = self.v2_config.orderflow
        return detect_stop_hunt(
            prices=prices,
            sl_zone_below=sl_zones.get("sl_zone_below", 0),
            sl_zone_above=sl_zones.get("sl_zone_above", 0),
            sl_density_below=sl_zones.get("sl_density_below", 0),
            sl_density_above=sl_zones.get("sl_density_above", 0),
            atr=atr,
            lookback=ofc.swing_lookback,
            sweep_depth_pct=ofc.sweep_depth_pct,
            min_score=ofc.stop_hunt_min_score,
        )

    def generate_signals_v2(
        self,
        features: pd.DataFrame,
        orderflow_df: pd.DataFrame,
        stop_hunt: Dict,
        sl_zones: Dict,
    ) -> pd.DataFrame:

        sig_cfg = self.v2_config.signals

        v1_features = self.generate_signals(features)

        df = v1_features.copy()
        n = len(df)

        if orderflow_df is not None and len(orderflow_df) >= n:
            of_df = orderflow_df.iloc[:n].reset_index(drop=True)
        else:
            of_df = pd.DataFrame(index=range(n))
            for col in ["ofi", "ofi_sma", "bid_ask_ratio", "absorption_bid", "absorption_ask",
                         "iceberg_bid_detected", "iceberg_ask_detected", "delta_divergence"]:
                of_df[col] = 0.0

        has_dom = "ofi" in of_df.columns and of_df["ofi"].notna().any()

        if not has_dom and sig_cfg.fallback_v1_on_no_dom:
            return df

        if not sig_cfg.dom_confirm_long and not sig_cfg.dom_confirm_short:
            df["signal_v1"] = df["signal"].copy()
            df["dom_validated"] = True
            df["dom_rejected"] = False
            return df

        df["signal_v1"] = df["signal"].copy()
        df["signal"] = 0

        for i in range(n):
            v1_signal = int(df["signal_v1"].iloc[i])

            if v1_signal == 0:
                continue

            ofi_val = float(of_df["ofi"].iloc[i]) if i < len(of_df) else 0.0
            bid_ask_ratio = float(of_df["bid_ask_ratio"].iloc[i]) if i < len(of_df) else 0.5
            abs_bid = bool(of_df["absorption_bid"].iloc[i]) if i < len(of_df) else False
            abs_ask = bool(of_df["absorption_ask"].iloc[i]) if i < len(of_df) else False
            ice_bid = bool(of_df["iceberg_bid_detected"].iloc[i]) if i < len(of_df) else False
            ice_ask = bool(of_df["iceberg_ask_detected"].iloc[i]) if i < len(of_df) else False

            if v1_signal == 1:
                dom_ok = True
                if sig_cfg.ofi_gate_long != 0 and ofi_val < sig_cfg.ofi_gate_long:
                    dom_ok = False
                if sig_cfg.bid_ask_ratio_long_min > 0 and bid_ask_ratio < sig_cfg.bid_ask_ratio_long_min:
                    dom_ok = False
                if sig_cfg.block_absorption and abs_ask:
                    dom_ok = False
                if sig_cfg.block_opposing_iceberg and ice_ask:
                    dom_ok = False

                if sig_cfg.stop_hunt_boost and stop_hunt:
                    if stop_hunt.get("sweep_direction") == +1 and stop_hunt.get("high_confidence", False):
                        dom_ok = True

                if dom_ok:
                    df.loc[df.index[i], "signal"] = 1

            elif v1_signal == -1:
                dom_ok = True
                if sig_cfg.ofi_gate_short != 0 and ofi_val > sig_cfg.ofi_gate_short:
                    dom_ok = False
                if sig_cfg.bid_ask_ratio_short_max > 0 and bid_ask_ratio > sig_cfg.bid_ask_ratio_short_max:
                    dom_ok = False
                if sig_cfg.block_absorption and abs_bid:
                    dom_ok = False
                if sig_cfg.block_opposing_iceberg and ice_bid:
                    dom_ok = False

                if sig_cfg.stop_hunt_boost and stop_hunt:
                    if stop_hunt.get("sweep_direction") == -1 and stop_hunt.get("high_confidence", False):
                        dom_ok = True

                if dom_ok:
                    df.loc[df.index[i], "signal"] = -1

        df["dom_validated"] = df["signal"].abs() > 0
        df["dom_rejected"] = (df["signal_v1"].abs() > 0) & (df["signal"] == 0)

        return df

    def _empty_orderflow_row(self) -> Dict:
        return {
            "top5_bid_vol": 0.0, "top5_ask_vol": 0.0, "bid_ask_ratio": 0.5,
            "bid_ask_total": 0.0, "weighted_mid": 0.0, "spread": 0.0,
            "best_bid": 0.0, "best_ask": 0.0, "last_price": 0.0,
            "last_volume": 0, "last_direction": 0,
            "ofi": 0.0, "ofi_sma": 0.0, "cum_delta": 0.0,
            "delta_per_bar": 0.0, "delta_divergence": False,
            "absorption_bid": False, "absorption_ask": False,
            "vpoc": 0.0, "vpoc_distance": 0.0,
            "iceberg_bid": 0.0, "iceberg_ask": 0.0,
            "iceberg_bid_detected": False, "iceberg_ask_detected": False,
            "iceberg_persistence": 0, "iceberg_confidence": 0.0,
        }
