import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timedelta
import time
import os
import tempfile


# Cross-process lock file: prevents ANY Python process from calling mt5.initialize()
# more than once, no matter how many processes are spawned.
_MT5_LOCK_FILE = os.path.join(tempfile.gettempdir(), "math_trading_mt5.lock")
_MT5_SESSION_LOCKED = False


class MT5Connector:
    TIMEFRAME_MAP = {
        1: mt5.TIMEFRAME_M1,
        2: mt5.TIMEFRAME_M2,
        3: mt5.TIMEFRAME_M3,
        5: mt5.TIMEFRAME_M5,
        10: mt5.TIMEFRAME_M10,
        15: mt5.TIMEFRAME_M15,
        30: mt5.TIMEFRAME_M30,
        60: mt5.TIMEFRAME_H1,
        240: mt5.TIMEFRAME_H4,
        1440: mt5.TIMEFRAME_D1,
    }

    def __init__(self, symbol: str = "XAUUSD", path: str = None):
        self.symbol = symbol
        self.path = path
        self.connected = False

    def connect(self, max_attempts: int = 3) -> bool:
        """
        Attach to the running MT5 terminal.

        Tries multiple times because MT5 sometimes needs a few seconds after
        login before the Python API can fetch data.
        Uses a cross-process lock so we never disturb the session after first attach.
        """
        global _MT5_SESSION_LOCKED

        if _MT5_SESSION_LOCKED:
            self.connected = True
            return True

        for attempt in range(1, max_attempts + 1):
            if self.path:
                ok = mt5.initialize(path=self.path)
            else:
                ok = mt5.initialize()

            if ok:
                try:
                    ti = mt5.terminal_info()
                    if ti is None:
                        raise RuntimeError("Terminal not responding")
                except Exception:
                    mt5.shutdown()
                    print(f"MT5 init attempt {attempt}/{max_attempts} connected but terminal not responding.")
                    if attempt < max_attempts:
                        time.sleep(3)
                    continue

                self.connected = True
                _MT5_SESSION_LOCKED = True

                acc = mt5.account_info()
                login = acc.login if acc else "?"

                print(f"MT5 attached (attempt {attempt}/{max_attempts})")
                print(f"  Terminal: {ti.name} | Account: #{login}")
                print("  Will follow whatever account you switch to manually in MT5.")

                self._resolve_symbol()
                return True

            err = mt5.last_error()
            print(f"MT5 init attempt {attempt}/{max_attempts} failed: {err}")
            if attempt < max_attempts:
                time.sleep(2)

        print("\nMT5 connection failed after retries.")
        print("  → Make sure only ONE MT5 terminal is open.")
        print("  → Make sure you are logged in and GOLD-Pro is visible.")
        return False

        # Try to connect (with path if specified, otherwise default)
        if self.path:
            ok = mt5.initialize(path=self.path)
        else:
            ok = mt5.initialize()

        if not ok:
            err = mt5.last_error()
            print("MT5 connection failed.")
            print(f"  Error: {err}")
            print("  → Make sure MetaTrader 5 is running and logged in.")
            print("  → Make sure the correct terminal has GOLD-Pro visible.")
            return False

        self.connected = True
        _MT5_SESSION_LOCKED = True

        acc = mt5.account_info()
        login = acc.login if acc else "?"

        print(f"MT5 attached. Current login: {login}")
        print("Will follow whatever account you switch to manually in MT5.")

        self._resolve_symbol()
        return True

        # First time in any process — initialize the MT5 session
        if self.path:
            ok = mt5.initialize(path=self.path)
        else:
            ok = mt5.initialize()

        if not ok:
            print(f"MT5 init failed: {mt5.last_error()}")
            return False

        self.connected = True

        # Create the lock so no future process touches initialize/shutdown
        try:
            with open(_MT5_LOCK_FILE, "w") as f:
                f.write(str(datetime.now()))
        except Exception:
            pass

        acc = mt5.account_info()
        login = acc.login if acc else "?"
        print(f"MT5 attached. Will follow whatever account you log into manually.")
        print(f"Current login: {login}")

        self._resolve_symbol()
        return True

    def get_current_login(self) -> int:
        """Returns the account that is currently logged into MT5 right now."""
        info = mt5.account_info()
        if info is None:
            return 0
        return info.login

    def _resolve_symbol(self):
        if mt5.symbol_info(self.symbol) is not None:
            return

        alternatives = [
            self.symbol + ".",
            self.symbol.rstrip(".") + ".",
            self.symbol + "m",
            self.symbol.rstrip("m") + "m",
        ]

        for alt in alternatives:
            if mt5.symbol_info(alt) is not None:
                print(f"Symbol '{self.symbol}' not found, using '{alt}'")
                self.symbol = alt
                return

        all_symbols = mt5.symbols_get()
        if all_symbols:
            matching = [s.name for s in all_symbols if self.symbol.replace(".", "").replace("m", "") in s.name]
            if matching:
                print(f"Found matching symbols for {self.symbol}: {matching[:10]}")
                self.symbol = matching[0]
                print(f"Using '{self.symbol}'")
                return

        print(f"WARNING: Symbol '{self.symbol}' not found in MT5. Available symbols with similar names not found.")

    def list_symbols(self, pattern: str = None) -> list:
        all_symbols = mt5.symbols_get()
        if all_symbols is None:
            return []
        if pattern:
            return [s.name for s in all_symbols if pattern.lower() in s.name.lower()]
        return [s.name for s in all_symbols]

    def disconnect(self, force: bool = False):
        """
        Release our reference to MT5.

        By default this does ABSOLUTELY NOTHING to the MT5 terminal.
        We never call mt5.shutdown() during normal operation.

        This is the only way to let you freely switch accounts inside the MT5
        terminal without the Python code forcing you back to account #10046026.

        Only pass force=True on full program exit if you really want to close MT5.
        """
        if force:
            mt5.shutdown()
            self.connected = False
            print("MT5 connection fully closed (forced).")
        else:
            self.connected = False
            # Do nothing to the MT5 session.

    def _tf(self, minutes: int):
        if minutes not in self.TIMEFRAME_MAP:
            raise ValueError(f"Timeframe {minutes}min not supported. Valid: {list(self.TIMEFRAME_MAP.keys())}")
        return self.TIMEFRAME_MAP[minutes]

    def fetch_rates(self, timeframe: int = 5, count: int = 2000, start_date: datetime = None) -> pd.DataFrame:
        tf = self._tf(timeframe)

        for attempt in range(3):
            if start_date:
                rates = mt5.copy_rates_from(self.symbol, tf, start_date, count)
            else:
                rates = mt5.copy_rates_from_pos(self.symbol, tf, 0, count)

            if rates is not None and len(rates) > 0:
                break

            if attempt < 2:
                time.sleep(2)
            else:
                raise RuntimeError(f"No rates for {self.symbol} TF={timeframe}min: {mt5.last_error()}")

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        df.rename(columns={"open": "open", "high": "high", "low": "low", "close": "close",
                           "tick_volume": "volume", "real_volume": "real_volume"}, inplace=True)
        df = df[["open", "high", "low", "close", "volume"]]

        return df

    def get_current_price(self) -> Tuple[float, float]:
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            raise RuntimeError(f"Cannot get price for {self.symbol}: {mt5.last_error()}")
        return tick.bid, tick.ask

    def get_spread(self) -> float:
        info = mt5.symbol_info(self.symbol)
        if info is None:
            return 0.0
        return info.spread

    def get_symbol_info(self) -> Dict:
        info = mt5.symbol_info(self.symbol)
        if info is None:
            return {}
        return {
            "symbol": info.name,
            "bid": info.bid,
            "ask": info.ask,
            "spread": info.spread,
            "digits": info.digits,
            "point": info.point,
            "volume_min": info.volume_min,
            "volume_step": info.volume_step,
            "trade_mode": info.trade_mode,
        }

    def place_order(
        self,
        order_type: str,
        volume: float,
        price: float = 0.0,
        sl: float = 0.0,
        tp: float = 0.0,
        deviation: int = 20,
        magic: int = 270426,
        comment: str = "math_trading",
    ) -> Optional[int]:

        if order_type.upper() == "BUY":
            request_type = mt5.ORDER_TYPE_BUY
        elif order_type.upper() == "SELL":
            request_type = mt5.ORDER_TYPE_SELL
        else:
            raise ValueError(f"Invalid order_type: {order_type}")

        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            return None

        point = symbol_info.point
        digits = symbol_info.digits

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": round(volume, 2),
            "type": request_type,
            "price": round(price, digits) if price > 0 else 0.0,
            "sl": round(sl, digits) if sl > 0 else 0.0,
            "tp": round(tp, digits) if tp > 0 else 0.0,
            "deviation": deviation,
            "magic": magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result is None:
            print(f"Order send failed: no result")
            return None

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Order failed: retcode={result.retcode}, {result.comment}")
            return None

        return result.order

    def close_position(self, position_ticket: int, deviation: int = 20) -> bool:
        position = mt5.positions_get(ticket=position_ticket)
        if position is None or len(position) == 0:
            print(f"Position {position_ticket} not found")
            return False

        pos = position[0]
        symbol_info = mt5.symbol_info(pos.symbol)
        point = symbol_info.point

        if pos.type == mt5.POSITION_TYPE_BUY:
            order_type = mt5.ORDER_TYPE_SELL
            close_price = mt5.symbol_info_tick(pos.symbol).bid
        else:
            order_type = mt5.ORDER_TYPE_BUY
            close_price = mt5.symbol_info_tick(pos.symbol).ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": order_type,
            "position": pos.ticket,
            "price": close_price,
            "deviation": deviation,
            "magic": 270426,
            "comment": "math_trading_close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Close failed: {result.comment if result else 'no result'}")
            return False

        return True

    def close_all_positions(self, symbol: str = None) -> int:
        target = symbol or self.symbol
        positions = mt5.positions_get(symbol=target)
        if positions is None:
            return 0

        closed = 0
        for pos in positions:
            if pos.magic == 270426:
                if self.close_position(pos.ticket):
                    closed += 1

        return closed

    def get_positions(self, symbol: str = None) -> List[Dict]:
        target = symbol or self.symbol
        positions = mt5.positions_get(symbol=target)
        if positions is None:
            return []

        result = []
        for pos in positions:
            if pos.magic != 270426:
                continue
            result.append({
                "ticket": pos.ticket,
                "symbol": pos.symbol,
                "type": "buy" if pos.type == mt5.POSITION_TYPE_BUY else "sell",
                "volume": pos.volume,
                "open_price": pos.price_open,
                "current_price": pos.price_current,
                "sl": pos.sl,
                "tp": pos.tp,
                "profit": pos.profit,
                "open_time": pos.time,
            })
        return result

    def get_account_info(self) -> Dict:
        """Always returns the current account info (respects manual account switches)."""
        info = mt5.account_info()
        if info is None:
            return {}
        return {
            "login": info.login,
            "balance": info.balance,
            "equity": info.equity,
            "margin": info.margin,
            "free_margin": info.margin_free,
            "leverage": info.leverage,
            "currency": info.currency,
        }

    def modify_position(self, ticket: int, sl: float = 0.0, tp: float = 0.0) -> bool:
        position = mt5.positions_get(ticket=ticket)
        if position is None or len(position) == 0:
            return False

        pos = position[0]
        symbol_info = mt5.symbol_info(pos.symbol)
        point = symbol_info.point
        digits = symbol_info.digits

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": pos.symbol,
            "position": pos.ticket,
            "sl": round(sl, digits) if sl > 0 else pos.sl,
            "tp": round(tp, digits) if tp > 0 else pos.tp,
        }

        result = mt5.order_send(request)
        return result is not None and result.retcode == mt5.TRADE_RETCODE_DONE
