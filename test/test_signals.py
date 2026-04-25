import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from signals.generator import SignalGenerator
from config.settings import GOLD_CONFIG


def test_signal_generator_no_signals_in_random():
    np.random.seed(42)
    n = 200
    prices = 2000 + np.random.randn(n).cumsum() * 2
    df = pd.DataFrame({
        "open": np.roll(prices, 1),
        "high": prices + np.abs(np.random.randn(n)) * 3,
        "low": prices - np.abs(np.random.randn(n)) * 3,
        "close": prices,
        "volume": np.random.lognormal(8, 0.8, n),
    })
    df.iloc[0, 0] = prices[0]

    gen = SignalGenerator(GOLD_CONFIG)
    features = gen.compute_features(df)
    signals = gen.generate_signals(features)

    assert "signal" in signals.columns
    assert signals["signal"].isin([-1, 0, 1]).all()
    assert abs(signals["signal"].sum()) < 10


def test_signal_generator_returns_features():
    np.random.seed(42)
    n = 300
    prices = 2000 + np.random.randn(n).cumsum() * 1.5
    df = pd.DataFrame({
        "open": np.roll(prices, 1),
        "high": prices + np.abs(np.random.randn(n)) * 2,
        "low": prices - np.abs(np.random.randn(n)) * 2,
        "close": prices,
        "volume": np.random.lognormal(8, 0.8, n),
    })
    df.iloc[0, 0] = prices[0]

    gen = SignalGenerator(GOLD_CONFIG)
    features = gen.compute_and_generate(df)

    required = ["zscore", "hurst", "velocity", "signal"]
    for col in required:
        assert col in features.columns, f"Missing column: {col}"

    assert len(features) == n
