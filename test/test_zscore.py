import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stats.zscore import rolling_zscore


def test_rolling_zscore_extreme():
    np.random.seed(42)
    prices = np.random.randn(300).cumsum() + 100
    df = rolling_zscore(prices, window=100)

    assert list(df.columns) == ["value", "mean", "std", "zscore"]
    assert len(df) == 300
    assert np.all(np.isnan(df["zscore"].iloc[:99]))
    assert not np.any(np.isnan(df["zscore"].iloc[99:]))

    z_values = df["zscore"].dropna()
    assert abs(z_values.mean()) < 0.5


def test_zscore_at_mean():
    prices = np.ones(200)
    df = rolling_zscore(prices, window=100)
    valid = df["zscore"].dropna()
    assert np.allclose(valid, 0.0, atol=1e-10)


def test_zscore_extreme_values():
    np.random.seed(42)
    n = 200
    prices = np.random.randn(n)
    big_move = np.arange(1, 21) * 0.5
    prices = np.concatenate([prices, 100 + big_move])

    df = rolling_zscore(prices, window=100)
    last_z = df["zscore"].iloc[-1]
    assert last_z > 2.0, f"Expected Z > 2.0 for extreme move, got {last_z}"
