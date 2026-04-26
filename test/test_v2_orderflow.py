import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings_v2 import V2Config
from signals.generator_v2 import SignalGeneratorV2
from data.rithmic import SyntheticDOMGenerator, DOMSnapshot
from orderflow.dom import compute_dom_features
from orderflow.ofi import OrderFlowImbalance
from orderflow.delta import CumulativeDelta
from orderflow.sl_zones import detect_sl_zones
from orderflow.stop_hunt import detect_stop_hunt
from orderflow.iceberg import detect_iceberg


def test_synthetic_dom_generation():
    gen = SyntheticDOMGenerator(seed=42)
    snap = gen.generate_snapshot(mid_price=4800, bar_direction=1)
    assert len(snap.bids) == 15
    assert len(snap.asks) == 15
    assert snap.bids[0].price <= snap.asks[0].price
    assert snap.top5_bid_volume > 0
    assert snap.top5_ask_volume > 0
    assert 0 < snap.bid_ask_ratio < 1


def test_dom_features():
    gen = SyntheticDOMGenerator(seed=42)
    snap = gen.generate_snapshot(mid_price=4800, bar_direction=0)
    features = compute_dom_features(snap, depth=5)
    assert features["bid_ask_ratio"] > 0
    assert features["spread"] >= 0
    assert features["best_bid"] > 0


def test_ofi_calculation():
    ofi = OrderFlowImbalance(window=3)
    gen = SyntheticDOMGenerator(seed=42)
    s1 = gen.generate_snapshot(4800, 1)
    s2 = gen.generate_snapshot(4801, 1)
    ofi.update(s1)
    v = ofi.update(s2)
    assert isinstance(v, float)


def test_cumulative_delta():
    cd = CumulativeDelta()
    d1 = cd.update(1, 100)
    d2 = cd.update(-1, 50)
    assert cd.cum_delta() == 50.0


def test_sl_zones():
    np.random.seed(42)
    prices = 4800 + np.cumsum(np.random.randn(200) * 5)
    zones = detect_sl_zones(prices, lookback=50, atr=10)
    assert "sl_zone_below" in zones
    assert "sl_zone_above" in zones


def test_stop_hunt():
    np.random.seed(42)
    prices = np.linspace(4800, 4790, 20)
    prices = np.append(prices, [4788, 4789, 4791, 4795, 4798])
    result = detect_stop_hunt(prices, sl_zone_below=4790, sl_zone_above=0,
                              sl_density_below=0.8, sl_density_above=0, atr=10.0)
    assert isinstance(result["stop_hunt_score"], float)


def test_iceberg_detection():
    gen = SyntheticDOMGenerator(seed=42)
    snap = gen.generate_snapshot(4800, 1)
    prev = [gen.generate_snapshot(4799, 1) for _ in range(5)]
    result = detect_iceberg(snap, prev_snapshots=prev)
    assert "iceberg_confidence" in result


def test_v2_signal_generator_passthrough():
    cfg = V2Config()
    cfg.signals.dom_confirm_long = False
    cfg.signals.dom_confirm_short = False

    np.random.seed(42)
    n = 300
    prices = 4800 + np.random.randn(n).cumsum() * 2
    df = pd.DataFrame({
        "open": np.roll(prices, 1), "high": prices + 2, "low": prices - 2,
        "close": prices, "volume": np.random.lognormal(8, 0.8, n),
    })
    df.iloc[0, df.columns.get_loc("open")] = prices[0]

    gen = SignalGeneratorV2(config=cfg)
    gen.load_dom_snapshots([])

    orderflow_df = gen.compute_orderflow_features(
        [SyntheticDOMGenerator(seed=42).generate_snapshot(float(p), 0) for p in prices],
        prices, atr=20.0,
    )
    sl_zones = gen.compute_sl_zones(prices, 20.0)
    stop_hunt = gen.compute_stop_hunt(prices, sl_zones, 20.0)

    features = gen.compute_features(df)
    signals = gen.generate_signals_v2(features, orderflow_df, stop_hunt, sl_zones)

    assert "signal" in signals.columns
    assert "signal_v1" in signals.columns
    assert (signals["signal"] == signals["signal_v1"]).all()


def test_v2_imports():
    from config.settings_v2 import V2Config
    from signals.generator_v2 import SignalGeneratorV2
    from data.rithmic import SyntheticDOMGenerator
    from live.rithmic_adapter import RithmicAdapter
    from orderflow.dom import compute_dom_features
    from orderflow.ofi import OrderFlowImbalance
    from orderflow.delta import CumulativeDelta
    from orderflow.absorption import detect_absorption
    from orderflow.footprint import compute_footprint
    from orderflow.stop_hunt import detect_stop_hunt
    from orderflow.iceberg import detect_iceberg
    from orderflow.sl_zones import detect_sl_zones
    assert True
