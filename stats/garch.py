import numpy as np
import pandas as pd
from typing import Optional
from arch import arch_model


class GARCHForecaster:
    def __init__(self, p: int = 1, q: int = 1, mean: str = "Constant"):
        self.p = p
        self.q = q
        self.mean = mean
        self._model = None
        self._result = None
        self._latest_forecast = None

    def fit(self, returns: np.ndarray, rescale: bool = True) -> "GARCHForecaster":
        returns = np.asarray(returns, dtype=np.float64).ravel()
        returns = returns[np.isfinite(returns)]

        if len(returns) < 100:
            raise ValueError(f"Need at least 100 observations, got {len(returns)}")

        if rescale:
            scale = returns.std()
            if scale > 1e-15:
                scaled = returns / scale
            else:
                scale = 1.0
                scaled = returns
        else:
            scale = 1.0
            scaled = returns

        self._model = arch_model(
            scaled * 100,
            vol="GARCH",
            p=self.p,
            q=self.q,
            mean=self.mean,
            dist="normal",
            rescale=False,
        )

        self._result = self._model.fit(disp="off")
        self._scale = scale
        return self

    def forecast_volatility(self, horizon: int = 5) -> np.ndarray:
        if self._result is None:
            raise RuntimeError("Model must be fitted before forecasting")

        forecast = self._result.forecast(horizon=horizon)
        variance_forecast = forecast.variance.values[-1, :]

        vol_forecast = np.sqrt(variance_forecast) * self._scale / 100.0
        return vol_forecast

    @property
    def latest_volatility(self) -> float:
        if self._result is None:
            return np.nan
        cond_vol = self._result.conditional_volatility
        if len(cond_vol) == 0:
            return np.nan
        return float(cond_vol[-1] * self._scale / 100.0)

    def is_volatility_tightening(self, horizon: int = 10) -> bool:
        vol_forecast = self.forecast_volatility(horizon)
        if len(vol_forecast) < 2:
            return False
        return vol_forecast[-1] < vol_forecast[0]


def forecast_volatility(returns: np.ndarray, horizon: int = 5, p: int = 1, q: int = 1) -> np.ndarray:
    forecaster = GARCHForecaster(p=p, q=q)
    forecaster.fit(returns)
    return forecaster.forecast_volatility(horizon)
