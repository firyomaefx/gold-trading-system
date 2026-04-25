from statsmodels.tsa.stattools import adfuller
import numpy as np


def adf_test(series: np.ndarray, maxlag: int = None, autolag: str = "AIC") -> dict:
    series = np.asarray(series, dtype=np.float64).ravel()
    series = series[np.isfinite(series)]

    if len(series) < 20:
        return {
            "test_stat": np.nan,
            "p_value": 1.0,
            "critical_values": {"1%": np.nan, "5%": np.nan, "10%": np.nan},
            "is_stationary": False,
            "n_obs": len(series),
        }

    result = adfuller(series, maxlag=maxlag, autolag=autolag)

    return {
        "test_stat": result[0],
        "p_value": result[1],
        "critical_values": {
            "1%": result[4]["1%"],
            "5%": result[4]["5%"],
            "10%": result[4]["10%"],
        },
        "is_stationary": result[1] < 0.05,
        "n_obs": result[3],
    }


def is_stationary(series: np.ndarray, significance: float = 0.05) -> bool:
    result = adf_test(series)
    return result["p_value"] < significance
