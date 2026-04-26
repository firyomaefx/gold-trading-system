import threading
import queue
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import numpy as np
from collections import deque
from datetime import datetime


@dataclass
class DOMLevel:
    price: float
    volume: int
    order_count: int = 0

@dataclass
class DOMSnapshot:
    timestamp: datetime = field(default_factory=datetime.now)
    bids: List[DOMLevel] = field(default_factory=list)
    asks: List[DOMLevel] = field(default_factory=list)
    last_price: float = 0.0
    last_volume: int = 0
    last_direction: int = 0
    total_bid_volume: float = 0.0
    total_ask_volume: float = 0.0
    top5_bid_volume: float = 0.0
    top5_ask_volume: float = 0.0
    bid_ask_ratio: float = 0.5
    weighted_mid: float = 0.0
    spread: float = 0.0

    def compute_derived(self):
        self.total_bid_volume = sum(b.volume for b in self.bids)
        self.total_ask_volume = sum(a.volume for a in self.asks)

        top_n = min(5, len(self.bids))
        self.top5_bid_volume = sum(b.volume for b in self.bids[:top_n])

        top_n = min(5, len(self.asks))
        self.top5_ask_volume = sum(a.volume for a in self.asks[:top_n])

        t = self.top5_bid_volume + self.top5_ask_volume
        if t > 0:
            self.bid_ask_ratio = self.top5_bid_volume / t

        if self.bids and self.asks:
            self.weighted_mid = (self.bids[0].price + self.asks[0].price) / 2
            self.spread = self.asks[0].price - self.bids[0].price
        elif self.last_price > 0:
            self.weighted_mid = self.last_price
            self.spread = 0.0


class RithmicL2Streamer:
    def __init__(self, config=None):
        self.config = config
        self._snapshots: deque = deque(maxlen=1000)
        self._current_snapshot: Optional[DOMSnapshot] = None
        self._snapshot_queue: queue.Queue = queue.Queue(maxsize=500)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._connected = False
        self._client = None

    def connect(self) -> bool:
        try:
            from async_rithmic import RithmicClient, DataType, OrderBookPresenceBits
            from async_rithmic.enums import SearchPattern

            cfg = self.config
            if not cfg or not cfg.host or not cfg.user:
                return False

            self._client = RithmicClient(
                host=cfg.host,
                port=cfg.port,
                user=cfg.user,
                password=cfg.password,
                system_name=cfg.system_name,
                app_name=cfg.app_name,
                app_version=cfg.app_version,
            )
            self._connected = True
            return True
        except ImportError:
            return False
        except Exception as e:
            print(f"Rithmic connection failed: {e}")
            return False

    async def _stream_order_book(self, symbol: str):
        try:
            from async_rithmic import DataType, OrderBookPresenceBits
            async for update in self._client.stream_market_data(
                symbol=symbol,
                data_types=[DataType.ORDER_BOOK, DataType.BBO, DataType.LAST_TRADE],
                order_book_presence=OrderBookPresenceBits.FULL,
            ):
                if not self._running:
                    break
                self._process_update(update)
        except Exception as e:
            print(f"Rithmic stream error: {e}")

    def _process_update(self, update):
        snap = self._current_snapshot
        if snap is None:
            snap = DOMSnapshot(timestamp=datetime.now())

        if hasattr(update, "bids") and hasattr(update, "asks"):
            snap.bids = [
                DOMLevel(price=float(b.price), volume=int(b.quantity))
                for b in update.bids[:20]
            ]
            snap.asks = [
                DOMLevel(price=float(a.price), volume=int(a.quantity))
                for a in update.asks[:20]
            ]
        elif hasattr(update, "bid_price") and hasattr(update, "ask_price"):
            if snap.bids:
                snap.bids[0] = DOMLevel(price=float(update.bid_price), volume=0)
            else:
                snap.bids = [DOMLevel(price=float(update.bid_price), volume=0)]
            if snap.asks:
                snap.asks[0] = DOMLevel(price=float(update.ask_price), volume=0)
            else:
                snap.asks = [DOMLevel(price=float(update.ask_price), volume=0)]

        if hasattr(update, "last_price") and hasattr(update, "last_volume"):
            snap.last_price = float(update.last_price)
            snap.last_volume = int(update.last_volume)
            try:
                snap.last_direction = int(update.direction) if hasattr(update, "direction") else 0
            except Exception:
                snap.last_direction = 0

        snap.compute_derived()
        snap.timestamp = datetime.now()
        self._current_snapshot = snap
        self._snapshots.append(snap)

        try:
            self._snapshot_queue.put_nowait(snap)
        except queue.Full:
            try:
                self._snapshot_queue.get_nowait()
                self._snapshot_queue.put_nowait(snap)
            except queue.Full:
                pass

    def _stream_loop(self):
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._stream_order_book(self.config.gold_symbol))
        finally:
            loop.close()

    def start(self) -> bool:
        if not self._connected:
            return False
        self._running = True
        self._thread = threading.Thread(target=self._stream_loop, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def get_latest_snapshot(self) -> Optional[DOMSnapshot]:
        try:
            return self._snapshot_queue.get_nowait()
        except queue.Empty:
            return self._current_snapshot

    def get_snapshot_at_bar(self, bar_timestamp: datetime) -> Optional[DOMSnapshot]:
        best = self._current_snapshot
        for s in self._snapshots:
            if s.timestamp >= bar_timestamp:
                return s
        return best

    def drain_queue(self):
        while not self._snapshot_queue.empty():
            try:
                self._snapshot_queue.get_nowait()
            except queue.Empty:
                break


class SyntheticDOMGenerator:
    def __init__(self, seed: int = None, price_range: Tuple[float, float] = (4500, 4900), tick_size: float = 0.1):
        self.rng = np.random.default_rng(seed)
        self.price_range = price_range
        self.tick_size = tick_size
        self._last_snap: Optional[DOMSnapshot] = None

    def generate_snapshot(self, mid_price: float, bar_direction: int = 0) -> DOMSnapshot:
        spread = self.rng.integers(1, 4) * self.tick_size
        bid_price = round(mid_price - spread / 2, 1)
        ask_price = round(mid_price + spread / 2, 1)

        bids = []
        asks = []

        for i in range(15):
            b = bid_price - i * self.tick_size
            vol = int(self.rng.lognormal(mean=4.0 + i * 0.15, sigma=0.6))
            if bar_direction >= 0:
                vol = int(vol * (1.0 + 0.3 * (i == 0)))
            bids.append(DOMLevel(price=max(b, self.price_range[0]), volume=vol))

        for i in range(15):
            a = ask_price + i * self.tick_size
            vol = int(self.rng.lognormal(mean=4.0 + i * 0.15, sigma=0.6))
            if bar_direction <= 0:
                vol = int(vol * (1.0 + 0.3 * (i == 0)))
            asks.append(DOMLevel(price=min(a, self.price_range[1]), volume=vol))

        if self.rng.random() < 0.08 and self._last_snap:
            persist_bid_idx = self.rng.integers(0, 8)
            persist_ask_idx = self.rng.integers(0, 8)
            if persist_bid_idx < len(bids):
                bids[persist_bid_idx].volume = int(bids[persist_bid_idx].volume * 3.0)
            if persist_ask_idx < len(asks):
                asks[persist_ask_idx].volume = int(asks[persist_ask_idx].volume * 3.0)

        snap = DOMSnapshot(
            timestamp=datetime.now(),
            bids=bids,
            asks=asks,
            last_price=mid_price,
            last_volume=self.rng.integers(10, 200),
            last_direction=bar_direction if bar_direction != 0 else int(self.rng.choice([-1, 1])),
        )
        snap.compute_derived()
        self._last_snap = snap
        return snap
