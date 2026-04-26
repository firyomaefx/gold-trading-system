import numpy as np
from collections import deque
from typing import Dict


class VolumeFootprint:
    def __init__(self, lookback: int = 50):
        self.lookback = lookback
        self._vpoc_history: deque = deque(maxlen=lookback)
        self._volume_profile: dict = {}

    def update(self, snapshot) -> Dict:
        price = snapshot.weighted_mid
        volume = snapshot.last_volume

        price_rounded = round(price / 0.1) * 0.1
        self._volume_profile[price_rounded] = self._volume_profile.get(price_rounded, 0) + volume

        max_vol = 0
        vpoc = price_rounded
        for p, v in self._volume_profile.items():
            if v > max_vol:
                max_vol = v
                vpoc = p

        self._vpoc_history.append(vpoc)

        return {"vpoc": vpoc, "poc_volume": max_vol, "vpoc_history": list(self._vpoc_history)}

    def distance_from_vpoc(self, current_price: float) -> float:
        if not self._vpoc_history:
            return 0.0
        vpoc = self._vpoc_history[-1]
        return abs(current_price - vpoc)


def compute_footprint(snapshots: list, lookback: int = 50, current_price: float = None) -> dict:
    footprint = VolumeFootprint(lookback=lookback)
    last_result = {}

    for snap in snapshots:
        last_result = footprint.update(snap)

    vpoc = last_result.get("vpoc", 0.0)
    distance = 0.0
    if current_price and vpoc:
        distance = abs(current_price - vpoc)

    return {
        "vpoc": vpoc,
        "poc_volume": last_result.get("poc_volume", 0),
        "vpoc_distance": distance,
        "vpoc_history": last_result.get("vpoc_history", []),
    }
