import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stats.hurst import hurst_exponent, rolling_hurst


def test_hurst_random_walk():
    np.random.seed(42)
    rw = np.cumsum(np.random.randn(500))
    h = hurst_exponent(rw)
    assert 0.30 < h < 0.80, f"Random walk H={h}, expected near 0.5"


def test_hurst_mean_reverting():
    np.random.seed(42)
    theta = 0.3
    mr = np.zeros(500)
    for t in range(1, 500):
        mr[t] = mr[t - 1] - theta * mr[t - 1] + np.random.randn() * 0.1
    h = hurst_exponent(mr)
    assert h < 0.55, f"Mean-reverting H={h}, expected <= 0.5"


def test_hurst_trending():
    np.random.seed(42)
    trend = np.cumsum(np.random.randn(500) + 0.05)
    h = hurst_exponent(trend)
    assert h > 0.45, f"Trending H={h}, expected > 0.45"


def test_rolling_hurst():
    np.random.seed(42)
    series = np.cumsum(np.random.randn(300))
    rh = rolling_hurst(series, window=100)
    assert len(rh) == 300
    assert not np.all(np.isnan(rh))
    assert np.sum(~np.isnan(rh)) > 100
