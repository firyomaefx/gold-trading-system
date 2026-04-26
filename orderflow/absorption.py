import numpy as np


def detect_absorption(snapshot, prev_snapshots: list = None, multiplier: float = 3.0) -> dict:
    if snapshot is None or not snapshot.bids or not snapshot.asks:
        return {"absorption_bid": False, "absorption_ask": False}

    bid_vols = [b.volume for b in snapshot.bids[:5]]
    ask_vols = [a.volume for a in snapshot.asks[:5]]

    avg_bid_vol = float(np.mean(bid_vols)) if bid_vols else 1.0
    avg_ask_vol = float(np.mean(ask_vols)) if ask_vols else 1.0

    absorption_bid = False
    for b in snapshot.bids[:3]:
        if b.volume > multiplier * avg_bid_vol:
            absorption_bid = True
            break

    absorption_ask = False
    for a in snapshot.asks[:3]:
        if a.volume > multiplier * avg_ask_vol:
            absorption_ask = True
            break

    if prev_snapshots and len(prev_snapshots) >= 2:
        prev_snap = prev_snapshots[-1]
        if prev_snap and prev_snap.bids and prev_snap.asks:
            bid_increase = snapshot.top5_bid_volume > prev_snap.top5_bid_volume * 1.5
            ask_increase = snapshot.top5_ask_volume > prev_snap.top5_ask_volume * 1.5
            absorption_bid = absorption_bid or bid_increase
            absorption_ask = absorption_ask or ask_increase

    return {
        "absorption_bid": absorption_bid,
        "absorption_ask": absorption_ask,
        "avg_bid_vol": avg_bid_vol,
        "avg_ask_vol": avg_ask_vol,
    }
