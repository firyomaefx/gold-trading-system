import numpy as np
from typing import Dict


def detect_stop_hunt(
    prices: np.ndarray,
    sl_zone_below: float,
    sl_zone_above: float,
    sl_density_below: float,
    sl_density_above: float,
    atr: float,
    lookback: int = 5,
    sweep_depth_pct: float = 0.001,
    min_score: float = 0.7,
) -> Dict:

    sweep_detected = False
    sweep_direction = 0
    sweep_reversal = False
    score = 0.0

    current = float(prices[-1]) if len(prices) > 0 else 0.0
    prev_close = float(prices[-2]) if len(prices) > 1 else current

    sweep_threshold = sweep_depth_pct * current

    if sl_zone_below > 0 and current < sl_zone_below - sweep_threshold:
        sweep_detected = True
        sweep_direction = +1

        depth = (sl_zone_below - current) / (atr + 1)
        density = sl_density_below

        if current > prev_close and len(prices) >= 3:
            low_in_window = float(np.min(prices[-3:]))
            if low_in_window < sl_zone_below and current > sl_zone_below - sweep_threshold * 0.3:
                sweep_reversal = True

        score = min(1.0, depth * 0.4 + density * 0.4 + (0.2 if sweep_reversal else 0.0))

    elif sl_zone_above > 0 and current > sl_zone_above + sweep_threshold:
        sweep_detected = True
        sweep_direction = -1

        depth = (current - sl_zone_above) / (atr + 1)
        density = sl_density_above

        if current < prev_close and len(prices) >= 3:
            high_in_window = float(np.max(prices[-3:]))
            if high_in_window > sl_zone_above and current < sl_zone_above + sweep_threshold * 0.3:
                sweep_reversal = True

        score = min(1.0, depth * 0.4 + density * 0.4 + (0.2 if sweep_reversal else 0.0))

    return {
        "sweep_detected": sweep_detected,
        "sweep_direction": sweep_direction,
        "sweep_reversal": sweep_reversal,
        "stop_hunt_score": score,
        "high_confidence": score >= min_score,
    }
