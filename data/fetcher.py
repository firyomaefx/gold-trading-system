from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

class DataSource(ABC):
    @abstractmethod
    def fetch(self, symbol: str, interval: str, start: str, end: Optional[str] = None) -> pd.DataFrame:
        pass

class YFinanceSource(DataSource):
    VALID_INTERVALS = {"1m", "2m", "5m", "15m", "30m", "1h", "1d"}
    MAX_INTRADAY_LOOKBACK = timedelta(days=59)

    def fetch(self, symbol: str, interval: str = "5m", start: str = None, end: Optional[str] = None) -> pd.DataFrame:
        if interval not in self.VALID_INTERVALS:
            raise ValueError(f"Interval {interval} not supported by yfinance. Valid: {self.VALID_INTERVALS}")

        if "yfinance" not in str(type(yf)):
            pass

        ticker = yf.Ticker(symbol)
        end_dt = pd.Timestamp(end) if end else pd.Timestamp.now()
        start_dt = pd.Timestamp(start) if start else end_dt - self.MAX_INTRADAY_LOOKBACK

        period_len = end_dt - start_dt
        if period_len > self.MAX_INTRADAY_LOOKBACK and interval in {"1m", "2m", "5m", "15m", "30m"}:
            start_dt = end_dt - self.MAX_INTRADAY_LOOKBACK

        df = ticker.history(start=start_dt.strftime("%Y-%m-%d"), end=end_dt.strftime("%Y-%m-%d"), interval=interval)

        if df.empty:
            raise ValueError(f"No data returned for {symbol} at {interval} interval from {start_dt} to {end_dt}")

        df.rename(columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume"
        }, inplace=True)

        return df[["open", "high", "low", "close", "volume"]]

class AlphaVantageSource(DataSource):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def fetch(self, symbol: str, interval: str = "5min", start: str = None, end: Optional[str] = None) -> pd.DataFrame:
        raise NotImplementedError("Alpha Vantage source not yet implemented. Configure API key and implement fetch.")

def get_source(name: str = "yfinance", **kwargs) -> DataSource:
    sources = {
        "yfinance": YFinanceSource(),
        "alpha_vantage": AlphaVantageSource(**kwargs),
    }
    if name not in sources:
        raise ValueError(f"Unknown source '{name}'. Valid: {list(sources.keys())}")
    return sources[name]
