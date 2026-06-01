import numpy as np
import pandas as pd
from typing import Optional, Dict

from stats.hurst import hurst_exponent, rolling_hurst
from stats.zscore import rolling_zscore
from stats.velocity import ma_velocity, velocity_approaching_zero
from stats.hmm import HMMRegimeDetector
from config.settings import ThresholdConfig, WindowConfig


class SignalGenerator:
    def __init__(self, config):
        self.threshold: ThresholdConfig = config.threshold
        self.window: WindowConfig = config.window
        self.hmm: Optional[HMMRegimeDetector] = None
        self._returns: Optional[np.ndarray] = None

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"].values.astype(np.float64)
        high = df["high"].values
        low = df["low"].values
        n = len(close)
        features = pd.DataFrame(index=df.index)

        returns = np.diff(close, prepend=close[0]) / np.where(close > 0, close, 1.0)
        self._returns = np.where(np.isfinite(returns), returns, 0.0)

        zscore_df = rolling_zscore(close, window=self.window.rolling_zscore)
        features["zscore"] = zscore_df["zscore"].values
        features["mean"] = zscore_df["mean"].values
        features["std"] = zscore_df["std"].values

        h_vals = rolling_hurst(close, window=self.window.rolling_zscore, max_lag=self.window.hurst_max_lag)
        features["hurst"] = h_vals

        velocity = ma_velocity(close, ma_period=self.window.rolling_ma)
        features["velocity"] = velocity
        features["velocity_zero"] = velocity_approaching_zero(
            velocity, epsilon=self.threshold.velocity_epsilon
        ).astype(int)

        # ATR ratio (volatility expansion filter)
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
        atr_fast = np.full(n, np.nan)
        atr_slow = np.full(n, np.nan)
        if n >= 14:
            atr_fast[13] = np.mean(tr[:14])
            atr_slow[13] = np.mean(tr[:14])
            for i in range(14, n):
                atr_fast[i] = (atr_fast[i-1] * 13 + tr[i]) / 14
                if i >= 50:
                    atr_slow[i] = (atr_slow[i-1] * 49 + tr[i]) / 50
                else:
                    atr_slow[i] = atr_fast[i]
        atr_ratio = np.where(atr_slow > 0, atr_fast / atr_slow, 1.0)
        features["atr_ratio"] = atr_ratio
        features["volatility_ok"] = (atr_ratio < self.threshold.atr_ratio_max).astype(int)

        # Session filter (London/NY overlap)
        if hasattr(df.index, 'hour'):
            hour = df.index.hour
            features["session_ok"] = ((hour >= self.threshold.session_start_hour) &
                                      (hour <= self.threshold.session_end_hour)).astype(int)
        else:
            features["session_ok"] = np.ones(n, dtype=int)

        try:
            if self.hmm is None:
                self.hmm = HMMRegimeDetector(n_states=2)
                self.hmm.fit(self._returns)
        except Exception:
            pass

        if self.hmm is not None and self.hmm._fitted:
            features["hmm_ranging_prob"] = self.hmm.predict_proba_series(self._returns)
            features["hmm_state"] = 0
        else:
            features["hmm_ranging_prob"] = 0.9
            features["hmm_state"] = 0

        features["returns"] = self._returns
        return features

    def generate_signals(self, features: pd.DataFrame) -> pd.DataFrame:
        df = features.copy()

        is_mean_revert = df["hurst"] < self.threshold.hurst_mean_revert
        is_oversold = df["zscore"] < self.threshold.zscore_entry_long
        is_overbought = df["zscore"] > self.threshold.zscore_entry_short
        velocity_flat = df["velocity_zero"] == 1

        # New filters for higher win rate
        volatility_ok = df["volatility_ok"] == 1
        session_ok = df["session_ok"] == 1

        long_condition = is_mean_revert & is_oversold & velocity_flat & volatility_ok & session_ok
        short_condition = is_mean_revert & is_overbought & velocity_flat & volatility_ok & session_ok

        if "hmm_ranging_prob" in df.columns and self.threshold.hmm_ranging_prob > 0.01:
            hmm_ranging = df["hmm_ranging_prob"] >= self.threshold.hmm_ranging_prob
            long_condition = long_condition & hmm_ranging
            short_condition = short_condition & hmm_ranging

        df["signal"] = 0
        df.loc[long_condition, "signal"] = 1
        df.loc[short_condition, "signal"] = -1

        df["entry_zscore"] = df["zscore"].where(df["signal"] != 0)

        return df

    def compute_and_generate(self, df: pd.DataFrame) -> pd.DataFrame:
        features = self.compute_features(df)
        return self.generate_signals(features)
