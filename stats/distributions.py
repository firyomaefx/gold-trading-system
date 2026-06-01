import numpy as np
import pandas as pd
from scipy import stats


def rolling_kurtosis(returns: np.ndarray, window: int = 200) -> np.ndarray:
    """Rolling excess (Fisher) kurtosis. Returns 0 for warmup bars."""
    n = len(returns)
    out = np.zeros(n, dtype=np.float64)
    if n < window:
        return out
    s = pd.Series(returns) if False else returns
    try:
        import pandas as _pd
        s = _pd.Series(returns)
        rk = s.rolling(window=window, min_periods=window // 2).kurt()
        out = rk.fillna(0.0).to_numpy()
    except Exception:
        for i in range(window, n):
            seg = returns[i - window:i]
            m = seg.mean()
            d = seg - m
            v = (d * d).mean()
            if v < 1e-16:
                out[i] = 0.0
                continue
            k = (d ** 4).mean() / (v ** 2)
            out[i] = k - 3.0
    return out


def kurtosis(returns: np.ndarray, fisher: bool = True) -> float:
    returns = np.asarray(returns, dtype=np.float64).ravel()
    returns = returns[np.isfinite(returns)]

    if len(returns) < 4:
        return 0.0

    if fisher:
        return float(stats.kurtosis(returns, fisher=True))
    else:
        return float(stats.kurtosis(returns, fisher=False))


def is_fat_tailed(returns: np.ndarray, threshold: float = 3.0, fisher: bool = True) -> bool:
    k = kurtosis(returns, fisher=fisher)

    if fisher:
        return k > threshold
    else:
        return k > (threshold + 3)


def var_historic(returns: np.ndarray, confidence: float = 0.99) -> float:
    returns = np.asarray(returns, dtype=np.float64).ravel()
    returns = returns[np.isfinite(returns)]

    if len(returns) < 10:
        return 0.0

    return float(np.percentile(returns, (1 - confidence) * 100))


def cvar_historic(returns: np.ndarray, confidence: float = 0.99) -> float:
    var = var_historic(returns, confidence)
    cvar = returns[returns <= var]
    if len(cvar) == 0:
        return var
    return float(cvar.mean())
