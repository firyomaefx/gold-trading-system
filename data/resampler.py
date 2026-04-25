import pandas as pd
from typing import Optional

def resample_to_timeframe(
    df: pd.DataFrame,
    target_tf: int,
    source_tf: Optional[int] = None,
) -> pd.DataFrame:
    required_cols = {"open", "high", "low", "close", "volume"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    freq_str = f"{target_tf}min"

    if source_tf is not None and target_tf <= source_tf:
        return df.copy()

    ohlcv = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }

    available_cols = {k: v for k, v in ohlcv.items() if k in df.columns}
    extra_cols = [c for c in df.columns if c not in available_cols]

    resampled = df.resample(freq_str).agg(available_cols)

    if extra_cols:
        extra_agg = df[extra_cols].resample(freq_str).mean()
        resampled = pd.concat([resampled, extra_agg], axis=1)

    resampled.dropna(inplace=True)
    return resampled

def forward_fill_gaps(df: pd.DataFrame, freq: str = "5min") -> pd.DataFrame:
    df = df.sort_index()
    full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
    df = df.reindex(full_idx)
    df.fillna(method="ffill", inplace=True)
    df.fillna(method="bfill", inplace=True)
    return df
