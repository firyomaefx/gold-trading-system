import numpy as np
import pandas as pd
from typing import Optional, Dict
import vectorbt as vbt

from signals.generator import SignalGenerator
from risk.kelly import position_size, calculate_trade_stats, BayesianKelly
from risk.exits import apply_exits_to_df
from backtest.metrics import calculate_metrics


class StatisticalBacktester:
    def __init__(self, config):
        self.config = config
        self.signal_generator = SignalGenerator(config)
        self.results: Dict = {}

    def _generate_exit_signals(self, signals_df: pd.DataFrame) -> pd.DataFrame:
        return apply_exits_to_df(
            signals_df,
            max_bars=self.config.threshold.time_stop_bars,
            zscore_stop_long=self.config.threshold.zscore_stop_long,
            zscore_stop_short=self.config.threshold.zscore_stop_short,
        )

    def run(
        self,
        df: pd.DataFrame,
        use_kelly: bool = True,
        df_htf: Optional[pd.DataFrame] = None,
    ) -> Dict:

        close = df["close"].values.astype(np.float64)

        if df_htf is None and self.config.threshold.multi_tf_enabled:
            try:
                from data.mtf import aggregate_from_minutes
                df_htf = aggregate_from_minutes(
                    df, minutes=self.config.threshold.multi_tf_minutes
                )
            except Exception:
                df_htf = None

        signals_df = self.signal_generator.compute_and_generate(df, df_htf=df_htf)
        signs_df = self._generate_exit_signals(signals_df)

        entries = (signs_df["signal"] == 1).astype(bool)
        exits = (signs_df["exit_signal"] == -1).astype(bool)
        short_entries = (signs_df["signal"] == -1).astype(bool)
        short_exits = (signs_df["exit_signal"] == 1).astype(bool)

        size = np.inf
        size_type = "percent"
        size_value = 1.0

        if use_kelly and entries.any():
            trades_loc = np.where(entries)[0]
            trade_results = []
            for e_idx in trades_loc:
                exit_indices = np.where(exits[e_idx:])[0]
                if len(exit_indices) > 0:
                    x_idx = e_idx + exit_indices[0]
                    pnl = close[x_idx] - close[e_idx]
                    trade_results.append(pnl)
                else:
                    pnl = close[-1] - close[e_idx]
                    trade_results.append(pnl)

            if trade_results:
                if getattr(self.config.risk, "use_bayesian_kelly", True):
                    bk = BayesianKelly(
                        prior_win_rate=self.config.risk.bayesian_prior_win_rate,
                        prior_strength=self.config.risk.bayesian_prior_strength,
                        prior_payoff=self.config.risk.bayesian_prior_payoff,
                        prior_payoff_strength=self.config.risk.bayesian_prior_payoff_strength,
                    )
                    bk.update_batch(trade_results)
                    kf = bk.conservative_kelly(shrink=self.config.risk.bayesian_kelly_shrink)
                else:
                    win_rate, avg_win, avg_loss, kf = calculate_trade_stats(trade_results)
                total_size = position_size(
                    account_equity=self.config.backtest.initial_capital,
                    kelly_f=kf,
                    atr=np.std(close[-50:]) if len(close) >= 50 else close[-1] * 0.005,
                    risk_pct=self.config.risk.account_risk_pct,
                )
                size_value = total_size / self.config.backtest.initial_capital

        try:
            pf = vbt.Portfolio.from_signals(
                close=close,
                entries=entries,
                exits=exits,
                short_entries=short_entries,
                short_exits=short_exits,
                init_cash=self.config.backtest.initial_capital,
                freq=f"{self.config.timeframe.primary}min",
            )
        except Exception:
            pf = vbt.Portfolio.from_signals(
                close=close,
                entries=entries,
                exits=exits,
                short_entries=short_entries,
                short_exits=short_exits,
                init_cash=self.config.backtest.initial_capital,
                freq=f"{self.config.timeframe.primary}min",
            )

        metrics = calculate_metrics(pf, self.config)

        self.results = {
            "portfolio": pf,
            "signals": signs_df,
            "metrics": metrics,
            "trades": pf.trades.records_readable if hasattr(pf, "trades") else None,
        }

        return self.results

    def equity_curve(self) -> pd.Series:
        if "portfolio" not in self.results:
            raise RuntimeError("No backtest results. Call run() first.")
        return self.results["portfolio"].value()

    def trade_log(self) -> Optional[pd.DataFrame]:
        if "trades" not in self.results or self.results["trades"] is None:
            return None
        return self.results["trades"]
