import numpy as np
from typing import List, Dict
from data.rithmic import DOMSnapshot


def compute_dom_features(snapshot: DOMSnapshot, depth: int = 5) -> Dict:
    bids = snapshot.bids[:depth]
    asks = snapshot.asks[:depth]

    top5_bid_vol = float(sum(b.volume for b in bids))
    top5_ask_vol = float(sum(a.volume for a in asks))

    total = top5_bid_vol + top5_ask_vol
    bid_ask_ratio = top5_bid_vol / total if total > 0 else 0.5

    top1_bid = float(bids[0].price) if bids else 0.0
    top1_ask = float(asks[0].price) if asks else 0.0

    return {
        "top5_bid_vol": top5_bid_vol,
        "top5_ask_vol": top5_ask_vol,
        "bid_ask_ratio": bid_ask_ratio,
        "bid_ask_total": total,
        "weighted_mid": float(snapshot.weighted_mid),
        "spread": float(snapshot.spread),
        "best_bid": top1_bid,
        "best_ask": top1_ask,
        "last_price": float(snapshot.last_price),
        "last_volume": int(snapshot.last_volume),
        "last_direction": int(snapshot.last_direction),
    }
