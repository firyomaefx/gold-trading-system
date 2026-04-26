import numpy as np
from typing import Dict, List
from data.rithmic import DOMSnapshot


def detect_iceberg(
    snapshot: DOMSnapshot,
    prev_snapshots: List[DOMSnapshot] = None,
    volume_ratio: float = 2.0,
    min_persistence: int = 3,
) -> Dict:

    iceberg_bid_volume = 0
    iceberg_ask_volume = 0
    iceberg_bid_detected = False
    iceberg_ask_detected = False
    iceberg_persistence = 0
    iceberg_confidence = 0.0

    if snapshot is None or not snapshot.bids or not snapshot.asks:
        return {
            "iceberg_bid": 0, "iceberg_ask": 0,
            "iceberg_bid_detected": False, "iceberg_ask_detected": False,
            "iceberg_persistence": 0, "iceberg_confidence": 0.0,
        }

    trade_volume = snapshot.last_volume
    trade_price = snapshot.last_price

    for b in snapshot.bids[:5]:
        if abs(b.price - trade_price) < 1.0 and b.volume > 0:
            iceberg_bid_volume = max(0, trade_volume - b.volume * volume_ratio)
            iceberg_bid_detected = iceberg_bid_volume > 0
            break

    if not iceberg_bid_detected:
        for a in snapshot.asks[:5]:
            if abs(a.price - trade_price) < 1.0 and a.volume > 0:
                iceberg_ask_volume = max(0, trade_volume - a.volume * volume_ratio)
                iceberg_ask_detected = iceberg_ask_volume > 0
                break

    if prev_snapshots and len(prev_snapshots) >= 2:
        persistence_count = 0
        for ps in prev_snapshots[-min_persistence:]:
            if iceberg_bid_detected and ps.bids:
                for pb in ps.bids[:3]:
                    if abs(pb.price - trade_price) < 1.0 and pb.volume > 0:
                        if trade_volume > pb.volume * volume_ratio:
                            persistence_count += 1
                        break
            elif iceberg_ask_detected and ps.asks:
                for pa in ps.asks[:3]:
                    if abs(pa.price - trade_price) < 1.0 and pa.volume > 0:
                        if trade_volume > pa.volume * volume_ratio:
                            persistence_count += 1
                        break
        iceberg_persistence = persistence_count

    est_hidden = max(iceberg_bid_volume, iceberg_ask_volume)
    if est_hidden > 0:
        iceberg_confidence = min(1.0, 0.3 + 0.3 * (iceberg_persistence / max(1, min_persistence)) + 0.4 * (
            est_hidden / (snapshot.last_volume + 1)))

    return {
        "iceberg_bid": iceberg_bid_volume,
        "iceberg_ask": iceberg_ask_volume,
        "iceberg_bid_detected": iceberg_bid_detected,
        "iceberg_ask_detected": iceberg_ask_detected,
        "iceberg_persistence": iceberg_persistence,
        "iceberg_confidence": iceberg_confidence,
    }
