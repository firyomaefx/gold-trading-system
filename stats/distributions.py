import numpy as np
from scipy import stats


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
