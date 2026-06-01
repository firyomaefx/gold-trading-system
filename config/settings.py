from dataclasses import dataclass, field
from typing import List
import os as _os

try:
    from dotenv import load_dotenv as _load_dotenv
    _env_path = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), ".env")
    if _os.path.exists(_env_path):
        _load_dotenv(_env_path, override=False)
except Exception:
    pass

try:
    from obsidian_sync.config import ObsidianConfig as _ObsidianConfig
except Exception:
    _ObsidianConfig = None  # type: ignore

@dataclass
class SymbolConfig:
    symbol: str
    yahoo_ticker: str
    pip_value: float
    typical_spread: float
    min_lot: float
    lot_step: float

@dataclass
class TimeframeConfig:
    available: List[int] = field(default_factory=lambda: [2, 5, 10])
    primary: int = 5
    min_bars_required: int = 100

@dataclass
class WindowConfig:
    rolling_zscore: int = 100
    rolling_ma: int = 20
    hurst_max_lag: int = 20
    hmm_training: int = 500
    garch_lookback: int = 200
    kelly_sample: int = 100

@dataclass
class ThresholdConfig:
    hurst_mean_revert: float = 0.35
    zscore_entry_long: float = -2.5
    zscore_entry_short: float = 3.0
    zscore_stop_long: float = -3.5
    zscore_stop_short: float = 3.5
    zscore_exit_target: float = 0.0
    zscore_exit_upper: float = 0.5
    zscore_partial_exit: float = 0.5
    zscore_trail_retrace: float = 0.30
    hurst_exit_threshold: float = 0.55
    velocity_epsilon: float = 3.0
    hmm_ranging_prob: float = 0.00
    time_stop_bars: int = 5
    max_consecutive_losses: int = 4
    atr_ratio_max: float = 1.6
    session_start_hour: int = 8
    session_end_hour: int = 17
    atr_ratio_max: float = 1.8
    session_start_hour: int = 8
    session_end_hour: int = 21
    adaptive_z_enabled: bool = True
    kurtosis_window: int = 200
    kurtosis_tighten_threshold: float = 4.0
    kurtosis_tighten_factor: float = 1.18
    kurtosis_loosen_threshold: float = 1.5
    kurtosis_loosen_factor: float = 0.90
    multi_tf_enabled: bool = False
    multi_tf_minutes: int = 15
    multi_tf_hurst_max: float = 0.45
    multi_tf_required_bars: int = 200

@dataclass
class RiskConfig:
    account_risk_pct: float = 0.02
    kelly_fraction: float = 0.50
    max_daily_loss_pct: float = 0.05
    atr_multiplier_sl: float = 1.5
    gold_bollinger_mult: float = 3.0

@dataclass
class BacktestConfig:
    initial_capital: float = 10_000.0
    commission_pct: float = 0.01
    slippage_pips: float = 0.5

@dataclass
class GoldConfig:
    symbol: SymbolConfig = field(default_factory=lambda: SymbolConfig(
        symbol="GOLD-Pro",
        yahoo_ticker="GC=F",
        pip_value=0.01,
        typical_spread=18,
        min_lot=0.01,
        lot_step=0.01,
    ))
    timeframe: TimeframeConfig = field(default_factory=TimeframeConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    disable_mt5: bool = False  # Set to True to run without MT5 connection
    obsidian: "object" = field(
        default_factory=lambda: _ObsidianConfig() if _ObsidianConfig else None
    )

GOLD_CONFIG = GoldConfig()

TRADING_SESSIONS = {
    "London": ("08:00", "17:00"),
    "New York": ("13:00", "22:00"),
    "London_NY_Overlap": ("13:00", "17:00"),
}

DEFAULT_PARAM_GRID = {
    "threshold__hurst_mean_revert": [0.30, 0.35, 0.40, 0.45],
    "threshold__zscore_entry_long": [-3.0, -2.5, -2.0],
    "threshold__zscore_entry_short": [2.0, 2.5, 3.0],
    "window__rolling_zscore": [50, 100, 150, 200],
}
