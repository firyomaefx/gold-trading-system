"""Multi-timeframe aggregation utilities.

Aggregates a lower timeframe OHLCV dataframe into a higher timeframe.
Used to derive 15-min bars from 5-min bars without re-fetching from MT5.
"""

from __future__ import annotations

import pandas as pd


def aggregate_ohlcv(df: pd.DataFrame, rule: str = "15min") -> pd.DataFrame:
    """Resample OHLCV to a higher timeframe.

    `rule` is a pandas offset alias (e.g. '15min', '30min', '1h').
    """
    if df.empty:
        return df.copy()
    out = df.resample(rule).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna(subset=["open", "close"])
    return out


def aggregate_from_minutes(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    return aggregate_ohlcv(df, rule=f"{int(minutes)}min")
