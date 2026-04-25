from .hurst import hurst_exponent, rolling_hurst
from .stationarity import adf_test, is_stationary
from .zscore import rolling_zscore, gold_bollinger_bands
from .garch import GARCHForecaster, forecast_volatility
from .hmm import HMMRegimeDetector
from .velocity import ma_velocity, velocity_approaching_zero
from .distributions import kurtosis, is_fat_tailed, var_historic
