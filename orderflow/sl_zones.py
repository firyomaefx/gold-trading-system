import numpy as np
from collections import deque
from typing import Dict, List, Tuple


def find_swing_levels(prices: np.ndarray, lookback: int = 5) -> Tuple[List[int], List[int], List[float], List[float]]:
    n = len(prices)
    swing_high_idx, swing_low_idx = [], []
    swing_highs, swing_lows = [], []

    for i in range(lookback, n - lookback):
        window = prices[i - lookback:i + lookback + 1]
        center = prices[i]

        is_high = True
        for j, v in enumerate(window):
            if j != lookback and v > center:
                is_high = False
                break
        if is_high:
            swing_high_idx.append(i)
            swing_highs.append(center)

        is_low = True
        for j, v in enumerate(window):
            if j != lookback and v < center:
                is_low = False
                break
        if is_low:
            swing_low_idx.append(i)
            swing_lows.append(center)

    return swing_high_idx, swing_low_idx, swing_highs, swing_lows


def find_round_numbers(price_min: float, price_max: float, digits: int = 2) -> List[float]:
    step = 10 ** digits
    start = int(price_min / step) * step
    end = int(price_max / step + 1) * step
    return [float(x) for x in range(int(start), int(end), int(step))]


def detect_sl_zones(
    prices: np.ndarray,
    lookback: int = 20,
    swing_fractal: int = 5,
    round_digits: int = 2,
    atr: float = 10.0,
    sl_zone_atr_distance: float = 0.5,
    sl_density_threshold: float = 0.6,
) -> Dict:

    _, _, swing_highs, swing_lows = find_swing_levels(prices, swing_fractal)

    price_min = float(np.min(prices[-lookback:]))
    price_max = float(np.max(prices[-lookback:]))

    round_numbers = find_round_numbers(price_min, price_max, round_digits)

    all_zones_above = []
    all_zones_below = []

    for h in swing_highs:
        if len(swing_highs) < 2:
            score = 0.6
        else:
            score = min(1.0, 0.5 + len([x for x in swing_highs if abs(x - h) < atr * 0.3]) * 0.15)
        all_zones_above.append((h, score))

    for l in swing_lows:
        if len(swing_lows) < 2:
            score = 0.6
        else:
            score = min(1.0, 0.5 + len([x for x in swing_lows if abs(x - l) < atr * 0.3]) * 0.15)
        all_zones_below.append((l, score))

    for rn in round_numbers:
        all_zones_above.append((rn, 0.7))
        all_zones_below.append((rn, 0.7))

    current_price = float(prices[-1])

    above = sorted([z for z in all_zones_above if z[0] > current_price], key=lambda x: x[0])
    below = sorted([z for z in all_zones_below if z[0] < current_price], key=lambda x: x[0], reverse=True)

    sl_zone_above = above[0][0] if above else 0.0
    sl_zone_below = below[0][0] if below else 0.0
    sl_density_above = above[0][1] if above else 0.0
    sl_density_below = below[0][1] if below else 0.0
    sl_dist_above = (sl_zone_above - current_price) / atr if atr > 0 and sl_zone_above > 0 else 999
    sl_dist_below = (current_price - sl_zone_below) / atr if atr > 0 and sl_zone_below > 0 else 999

    return {
        "sl_zone_above": sl_zone_above,
        "sl_zone_below": sl_zone_below,
        "sl_dist_above": sl_dist_above,
        "sl_dist_below": sl_dist_below,
        "sl_density_above": sl_density_above,
        "sl_density_below": sl_density_below,
        "swing_highs": swing_highs,
        "swing_lows": swing_lows,
        "round_numbers": round_numbers,
    }
