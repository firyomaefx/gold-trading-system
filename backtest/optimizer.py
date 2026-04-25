import numpy as np
import pandas as pd
from typing import Dict, Iterator
from itertools import product

from backtest.engine import StatisticalBacktester
from config.settings import DEFAULT_PARAM_GRID


class Optimizer:
    def __init__(self, base_config):
        self.base_config = base_config
        self.results_list = []

    def _build_configs(self, param_grid: Dict) -> Iterator:
        keys = list(param_grid.keys())
        values = list(param_grid.values())

        for combination in product(*values):
            config = self.base_config.__class__.__new__(self.base_config.__class__)
            config.symbol = self.base_config.symbol
            config.timeframe = self.base_config.timeframe
            config.window = self.base_config.window.__class__.__new__(self.base_config.window.__class__)
            config.threshold = self.base_config.threshold.__class__.__new__(self.base_config.threshold.__class__)
            config.risk = self.base_config.risk
            config.backtest = self.base_config.backtest

            for source_obj in [self.base_config.window, self.base_config.threshold]:
                for field_name in source_obj.__dataclass_fields__:
                    setattr(getattr(config, "window" if hasattr(source_obj, "rolling_zscore") else "threshold"), field_name, getattr(source_obj, field_name))

            for key, val in zip(keys, combination):
                section, param = key.split("__")
                target_obj = getattr(config, section)
                setattr(target_obj, param, val)

            yield config

    def grid_search(
        self,
        df: pd.DataFrame,
        param_grid: Dict = None,
        metric: str = "sharpe_ratio",
    ) -> pd.DataFrame:

        if param_grid is None:
            param_grid = DEFAULT_PARAM_GRID

        results = []
        n_configs = 1
        for v in param_grid.values():
            n_configs *= len(v)

        i = 0
        for config in self._build_configs(param_grid):
            i += 1
            if i % 10 == 0:
                print(f"  Optimizing... {i}/{n_configs}")

            bt = StatisticalBacktester(config)
            result = bt.run(df, use_kelly=True)

            row = {}
            for key in param_grid.keys():
                section, param = key.split("__")
                row[param] = getattr(getattr(config, section), param)

            row.update(result["metrics"])
            results.append(row)

        self.results_list = results
        return pd.DataFrame(results).sort_values(metric, ascending=False)

    def best_params(self, metric: str = "sharpe_ratio") -> Dict:
        if not self.results_list:
            return {}
        df = pd.DataFrame(self.results_list)
        best = df.loc[df[metric].idxmax()].to_dict()
        return best


def run_optimization(config, df: pd.DataFrame, param_grid: Dict = None) -> pd.DataFrame:
    opt = Optimizer(config)
    results = opt.grid_search(df, param_grid)
    print(f"\nOptimization complete. {len(results)} configurations tested.")
    print(f"Best params:\n{opt.best_params()}")
    return results
