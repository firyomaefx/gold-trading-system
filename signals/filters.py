import numpy as np
import pandas as pd
from typing import Optional
from datetime import datetime


class SignalFilters:
    def __init__(self, max_consecutive: int = 1, allowed_sessions: Optional[list] = None):
        self.max_consecutive = max_consecutive
        self.allowed_sessions = allowed_sessions or []
        self._last_signal = 0
        self._consecutive_count = 0

    def filter_consecutive(self, signal: int) -> int:
        if signal != 0 and signal == self._last_signal:
            self._consecutive_count += 1
        elif signal != 0:
            self._consecutive_count = 1
        else:
            self._consecutive_count = 0

        self._last_signal = signal

        if signal != 0 and self._consecutive_count > self.max_consecutive:
            return 0

        return signal

    def filter_session(self, dt: datetime, signal: int) -> int:
        if not self.allowed_sessions:
            return signal
        if signal == 0:
            return signal
        time_str = dt.strftime("%H:%M")
        for start, end in self.allowed_sessions:
            if start <= time_str <= end:
                return signal
        return 0

    def filter_spread(self, current_spread: float, avg_spread: float, signal: int, multiplier: float = 2.0) -> int:
        if signal != 0 and current_spread > avg_spread * multiplier:
            return 0
        return signal

    def apply_all(
        self,
        signal: int,
        dt: datetime,
        current_spread: float = 0.0,
        avg_spread: float = 0.0,
    ) -> int:
        signal = self.filter_consecutive(signal)
        signal = self.filter_session(dt, signal)
        signal = self.filter_spread(current_spread, avg_spread, signal)
        return signal
