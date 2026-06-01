import numpy as np
import pandas as pd
from typing import Optional, Dict

from stats.hurst import hurst_exponent, rolling_hurst
from stats.zscore import rolling_zscore
from stats.velocity import ma_velocity, velocity_approaching_zero
from stats.hmm import HMMRegimeDetector
from stats.distributions import rolling_kurtosis
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

    def compute_features_multi_tf(
        self,
        df: pd.DataFrame,
        df_htf: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """compute_features() + rolling kurtosis + multi-TF Hurst confirmation."""
        features = self.compute_features(df)
        kurt = rolling_kurtosis(self._returns, window=self.threshold.kurtosis_window)
        features["kurtosis"] = kurt

        if self.threshold.adaptive_z_enabled:
            tight = kurt > self.threshold.kurtosis_tighten_threshold
            loose = kurt < self.threshold.kurtosis_loosen_threshold
            adj = np.ones_like(kurt, dtype=np.float64)
            adj[tight] = self.threshold.kurtosis_tighten_factor
            adj[loose] = self.threshold.kurtosis_loosen_factor
            features["zscore_adaptive"] = features["zscore"].values / np.where(adj != 0, adj, 1.0)
        else:
            features["zscore_adaptive"] = features["zscore"].values

        features.attrs["htf_hurst"] = float("nan")
        features["multi_tf_ok"] = 1

        if self.threshold.multi_tf_enabled and df_htf is not None and len(df_htf) >= self.threshold.multi_tf_required_bars:
            try:
                htf_hurst = rolling_hurst(
                    df_htf["close"].values.astype(np.float64),
                    window=min(self.threshold.multi_tf_required_bars, len(df_htf) // 2),
                    max_lag=self.window.hurst_max_lag,
                )
                reindexed = pd.Series(htf_hurst, index=df_htf.index).reindex(
                    df.index, method="ffill"
                )
                reindexed = reindexed.fillna(0.5)
                features["htf_hurst"] = reindexed.values
                features["multi_tf_ok"] = (reindexed.values < self.threshold.multi_tf_hurst_max).astype(int)
                features.attrs["htf_hurst_last"] = float(reindexed.iloc[-1])
            except Exception as e:
                logger = __import__("logging").getLogger(__name__)
                logger.debug("rolling HTF Hurst failed: %s", e)

        return features

    def generate_signals(self, features: pd.DataFrame) -> pd.DataFrame:
        df = features.copy()

        is_mean_revert = df["hurst"] < self.threshold.hurst_mean_revert
        z_entry = df.get("zscore_adaptive", df["zscore"])
        is_oversold = z_entry < self.threshold.zscore_entry_long
        is_overbought = z_entry > self.threshold.zscore_entry_short
        velocity_flat = df["velocity_zero"] == 1

        volatility_ok = df["volatility_ok"] == 1
        session_ok = df["session_ok"] == 1
        multi_tf_ok = df.get("multi_tf_ok", 1) == 1

        long_condition = is_mean_revert & is_oversold & velocity_flat & volatility_ok & session_ok & multi_tf_ok
        short_condition = is_mean_revert & is_overbought & velocity_flat & volatility_ok & session_ok & multi_tf_ok

        if "hmm_ranging_prob" in df.columns and self.threshold.hmm_ranging_prob > 0.01:
            hmm_ranging = df["hmm_ranging_prob"] >= self.threshold.hmm_ranging_prob
            long_condition = long_condition & hmm_ranging
            short_condition = short_condition & hmm_ranging

        df["signal"] = 0
        df.loc[long_condition, "signal"] = 1
        df.loc[short_condition, "signal"] = -1

        df["entry_zscore"] = df["zscore"].where(df["signal"] != 0)

        return df

    def compute_and_generate(
        self,
        df: pd.DataFrame,
        df_htf: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        if self.threshold.adaptive_z_enabled or self.threshold.multi_tf_enabled:
            features = self.compute_features_multi_tf(df, df_htf=df_htf)
        else:
            features = self.compute_features(df)
        return self.generate_signals(features)
