import threading
import queue
import time
from typing import Optional, List
from datetime import datetime

from data.rithmic import RithmicL2Streamer, SyntheticDOMGenerator, DOMSnapshot
from config.settings_v2 import V2Config, RithmicConfig
from orderflow.dom import compute_dom_features
from orderflow.ofi import OrderFlowImbalance
from orderflow.delta import CumulativeDelta
from orderflow.absorption import detect_absorption
from orderflow.footprint import VolumeFootprint
from orderflow.iceberg import detect_iceberg


class RithmicAdapter:
    def __init__(self, config: V2Config = None):
        self.config = config or V2Config()
        self._streamer: Optional[RithmicL2Streamer] = None
        self._synthetic = SyntheticDOMGenerator(seed=42)
        self._using_synthetic = False
        self._snapshots: List[DOMSnapshot] = []
        self._bar_snapshots: List[DOMSnapshot] = []
        self._ofi = OrderFlowImbalance(window=self.config.orderflow.ofi_window)
        self._delta = CumulativeDelta()
        self._footprint = VolumeFootprint(lookback=self.config.orderflow.footprint_lookback)
        self._connected = False

    def connect(self) -> bool:
        rcfg = self.config.rithmic
        if rcfg and rcfg.host and rcfg.user and rcfg.password:
            self._streamer = RithmicL2Streamer(config=rcfg)
            if self._streamer.connect():
                if self._streamer.start():
                    self._connected = True
                    self._using_synthetic = False
                    time.sleep(2)
                    return True

        self._using_synthetic = True
        self._connected = True
        return True

    def disconnect(self):
        if self._streamer:
            self._streamer.stop()
        self._connected = False

    def get_snapshot(self, mid_price: float = 4800, bar_direction: int = 0) -> DOMSnapshot:
        if self._using_synthetic:
            snap = self._synthetic.generate_snapshot(mid_price, bar_direction)
        elif self._streamer:
            snap = self._streamer.get_latest_snapshot()
            if snap is None:
                snap = self._synthetic.generate_snapshot(mid_price, bar_direction)
        else:
            snap = self._synthetic.generate_snapshot(mid_price, bar_direction)

        self._snapshots.append(snap)
        self._bar_snapshots.append(snap)
        self._ofi.update(snap)
        self._delta.update(snap.last_direction, snap.last_volume)
        self._footprint.update(snap)

        if len(self._snapshots) > 500:
            self._snapshots = self._snapshots[-500:]
        if len(self._bar_snapshots) > 100:
            self._bar_snapshots = self._bar_snapshots[-100:]

        return snap

    def end_bar(self):
        self._delta.reset_bar()
        self._bar_snapshots = []
        if self._streamer:
            self._streamer.drain_queue()

    def get_orderflow_state(self):
        snap = self._snapshots[-1] if self._snapshots else None
        if snap is None:
            return self._empty_state()

        dom = compute_dom_features(snap, depth=self.config.orderflow.dom_depth)
        absorption = detect_absorption(snap, prev_snapshots=list(self._snapshots)[-5:-1])
        iceberg = detect_iceberg(snap, prev_snapshots=list(self._snapshots)[-5:-1])

        return {
            **dom,
            "ofi": self._ofi.latest(),
            "ofi_sma": self._ofi.sma(),
            "cum_delta": self._delta.cum_delta(),
            "delta_per_bar": self._delta.latest_delta(),
            "absorption_bid": absorption["absorption_bid"],
            "absorption_ask": absorption["absorption_ask"],
            "iceberg_bid": iceberg["iceberg_bid"],
            "iceberg_ask": iceberg["iceberg_ask"],
            "iceberg_bid_detected": iceberg["iceberg_bid_detected"],
            "iceberg_ask_detected": iceberg["iceberg_ask_detected"],
            "iceberg_persistence": iceberg["iceberg_persistence"],
            "iceberg_confidence": iceberg["iceberg_confidence"],
            "vpoc": self._footprint._vpoc_history[-1] if self._footprint._vpoc_history else 0.0,
            "connected": self._connected,
            "using_synthetic": self._using_synthetic,
        }

    def _empty_state(self):
        return {
            "ofi": 0.0, "ofi_sma": 0.0, "cum_delta": 0.0, "delta_per_bar": 0.0,
            "bid_ask_ratio": 0.5, "connected": self._connected,
            "using_synthetic": self._using_synthetic,
        }

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def is_synthetic(self) -> bool:
        return self._using_synthetic
