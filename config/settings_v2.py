from dataclasses import dataclass, field
import os
from config.settings import SymbolConfig, TimeframeConfig, WindowConfig, ThresholdConfig, RiskConfig, BacktestConfig, GoldConfig


@dataclass
class OrderflowConfig:
    dom_depth: int = 10
    bid_ask_ratio_long: float = 0.6
    bid_ask_ratio_short: float = 0.4
    ofi_window: int = 5
    delta_divergence_lookback: int = 3
    swing_lookback: int = 20
    sweep_depth_pct: float = 0.001
    stop_hunt_min_score: float = 0.7
    iceberg_volume_ratio: float = 2.0
    iceberg_min_persistence: int = 3
    iceberg_block_signal: bool = True
    sl_zone_atr_distance: float = 0.5
    sl_round_number_digits: int = 2
    sl_density_threshold: float = 0.6
    absorption_multiplier: float = 3.0
    footprint_lookback: int = 50


@dataclass
class RithmicConfig:
    host: str = ""
    port: int = 64100
    user: str = ""
    password: str = ""
    gold_symbol: str = "GC"
    system_name: str = "math_trading_v2"
    app_name: str = "v2"
    app_version: str = "1.0.0"

    @classmethod
    def from_env(cls) -> "RithmicConfig":
        return cls(
            host=os.getenv("RITHMIC_HOST", ""),
            port=int(os.getenv("RITHMIC_PORT", "64100")),
            user=os.getenv("RITHMIC_USER", ""),
            password=os.getenv("RITHMIC_PASS", ""),
            gold_symbol=os.getenv("RITHMIC_GOLD_SYMBOL", "GC"),
            system_name=os.getenv("RITHMIC_SYSTEM_NAME", "math_trading_v2"),
            app_name=os.getenv("RITHMIC_APP_NAME", "v2"),
            app_version=os.getenv("RITHMIC_APP_VERSION", "1.0.0"),
        )


@dataclass
class V2SignalConfig:
    dom_confirm_long: bool = False
    dom_confirm_short: bool = False
    ofi_gate_long: float = -999.0
    ofi_gate_short: float = 999.0
    bid_ask_ratio_long_min: float = 0.0
    bid_ask_ratio_short_max: float = 1.0
    block_opposing_iceberg: bool = False
    block_absorption: bool = False
    stop_hunt_boost: bool = False
    stop_hunt_boost_multiplier: float = 1.5
    delta_divergence_exit: bool = False
    fallback_v1_on_no_dom: bool = True


@dataclass
class V2Config:
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
    orderflow: OrderflowConfig = field(default_factory=OrderflowConfig)
    rithmic: RithmicConfig = field(default_factory=RithmicConfig.from_env)
    signals: V2SignalConfig = field(default_factory=V2SignalConfig)

    def load_env(self):
        from dotenv import load_dotenv
        load_dotenv()
        self.rithmic = RithmicConfig.from_env()


V2_CONFIG = V2Config()
