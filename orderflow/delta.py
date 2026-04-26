import numpy as np
from collections import deque


class CumulativeDelta:
    def __init__(self):
        self._cumulative = 0.0
        self._delta_per_bar: deque = deque(maxlen=200)

    def update(self, last_direction: int, last_volume: int) -> float:
        delta = last_direction * last_volume
        self._cumulative += delta
        self._delta_per_bar.append(delta)
        return float(delta)

    def reset_bar(self):
        self._delta_per_bar.append(0.0)

    def cum_delta(self) -> float:
        return float(self._cumulative)

    def latest_delta(self) -> float:
        if not self._delta_per_bar:
            return 0.0
        return float(self._delta_per_bar[-1])

    def delta_series(self, n: int = None) -> list:
        items = list(self._delta_per_bar)
        return items[-n:] if n else items


def compute_delta(snapshots: list, lookback: int = 3) -> dict:
    delta_calc = CumulativeDelta()
    delta_values = []

    for snap in snapshots:
        d = delta_calc.update(snap.last_direction, snap.last_volume)
        delta_values.append(d)

    cum_delta = delta_calc.cum_delta()
    latest = delta_values[-1] if delta_values else 0.0

    divergence = False
    if len(delta_values) >= lookback + 1:
        recent_delta = sum(delta_values[-lookback:])
        divergence = (latest > 0 and recent_delta / (abs(latest) + 1) < -0.5) or \
                     (latest < 0 and recent_delta / (abs(latest) + 1) > 0.5)

    return {
        "cum_delta": cum_delta,
        "delta_per_bar": latest,
        "delta_series": delta_values,
        "delta_divergence": divergence,
    }
