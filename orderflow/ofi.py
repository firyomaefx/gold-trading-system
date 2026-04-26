import numpy as np
from collections import deque


class OrderFlowImbalance:
    def __init__(self, window: int = 5):
        self.window = window
        self._ofi_deque: deque = deque(maxlen=window)
        self._prev_snapshot = None

    def update(self, snapshot) -> float:
        if self._prev_snapshot is None:
            self._prev_snapshot = snapshot
            self._ofi_deque.append(0.0)
            return 0.0

        ofi = 0.0
        prev_bids = {b.price: b.volume for b in self._prev_snapshot.bids}
        prev_asks = {a.price: a.volume for a in self._prev_snapshot.asks}
        curr_bids = {b.price: b.volume for b in snapshot.bids}
        curr_asks = {a.price: a.volume for a in snapshot.asks}

        prev_best_bid = max(prev_bids) if prev_bids else 0
        prev_best_ask = min(prev_asks) if prev_asks else 0
        curr_best_bid = max(curr_bids) if curr_bids else 0
        curr_best_ask = min(curr_asks) if curr_asks else 0

        for price, vol in curr_bids.items():
            old_vol = prev_bids.get(price, 0)
            if price >= prev_best_bid:
                ofi += (vol - old_vol)

        for price, vol in curr_asks.items():
            old_vol = prev_asks.get(price, 0)
            if price <= prev_best_ask:
                ofi -= (vol - old_vol)

        for price in prev_bids:
            if price not in curr_bids and price >= prev_best_bid:
                ofi -= prev_bids[price]

        for price in prev_asks:
            if price not in curr_asks and price <= prev_best_ask:
                ofi += prev_asks[price]

        self._prev_snapshot = snapshot
        self._ofi_deque.append(ofi)

        return float(ofi)

    def sma(self) -> float:
        if not self._ofi_deque:
            return 0.0
        return float(np.mean(list(self._ofi_deque)))

    def latest(self) -> float:
        if not self._ofi_deque:
            return 0.0
        return float(self._ofi_deque[-1])


def compute_ofi(snapshots: list, window: int = 5) -> dict:
    ofi_calc = OrderFlowImbalance(window=window)
    ofi_values = []
    ofi_sma_values = []

    for snap in snapshots:
        v = ofi_calc.update(snap)
        ofi_values.append(v)
        ofi_sma_values.append(ofi_calc.sma())

    return {
        "ofi": ofi_values[-1] if ofi_values else 0.0,
        "ofi_sma": ofi_sma_values[-1] if ofi_sma_values else 0.0,
        "ofi_values": ofi_values,
        "ofi_sma_values": ofi_sma_values,
    }
