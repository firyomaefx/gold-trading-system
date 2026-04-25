import numpy as np
import pandas as pd
from typing import Dict


def calculate_metrics(portfolio, config) -> Dict:
    total_return = float(portfolio.total_return())
    init_cash = float(portfolio.init_cash)
    final_value = init_cash * (1 + total_return)

    try:
        sharpe = float(portfolio.sharpe_ratio(freq=f"{config.timeframe.primary}min"))
    except Exception:
        returns = portfolio.returns()
        sharpe = float(np.sqrt(252 * 78) * returns.mean() / returns.std()) if returns.std() > 0 else 0.0

    try:
        sortino = float(portfolio.sortino_ratio(freq=f"{config.timeframe.primary}min"))
    except Exception:
        sortino = sharpe

    try:
        max_dd = float(portfolio.max_drawdown())
    except Exception:
        max_dd = -abs(1 - final_value / init_cash) if final_value < init_cash else 0.0

    try:
        calmar = float(portfolio.calmar_ratio()) if hasattr(portfolio, "calmar_ratio") else (total_return / abs(max_dd) if abs(max_dd) > 1e-10 else 0.0)
    except Exception:
        calmar = 0.0

    try:
        n_trades = len(portfolio.trades.records) if hasattr(portfolio, "trades") else 0
    except Exception:
        n_trades = 0

    try:
        win_rate = float(portfolio.trades.win_rate()) if n_trades > 0 and hasattr(portfolio.trades, "win_rate") else 0.0
    except Exception:
        win_rate = 0.0

    try:
        profit_factor = float(portfolio.trades.profit_factor()) if n_trades > 0 and hasattr(portfolio.trades, "profit_factor") else 0.0
    except Exception:
        profit_factor = 0.0

    try:
        avg_win = float(portfolio.trades.avg_win()) if n_trades > 0 and hasattr(portfolio.trades, "avg_win") else 0.0
    except Exception:
        avg_win = 0.0

    try:
        avg_loss = float(portfolio.trades.avg_loss()) if n_trades > 0 and hasattr(portfolio.trades, "avg_loss") else 0.0
    except Exception:
        avg_loss = 0.0

    expectancy = (win_rate * avg_win) - ((1.0 - win_rate) * abs(avg_loss)) if n_trades > 0 else 0.0

    metrics = {
        "total_return_pct": round(total_return * 100, 2),
        "final_value": round(final_value, 2),
        "sharpe_ratio": round(sharpe, 3),
        "sortino_ratio": round(sortino, 3),
        "calmar_ratio": round(calmar, 3),
        "max_drawdown_pct": round(abs(max_dd) * 100, 2),
        "num_trades": n_trades,
        "win_rate_pct": round(win_rate * 100, 2),
        "profit_factor": round(profit_factor, 2),
        "avg_win": round(avg_win, 4),
        "avg_loss": round(avg_loss, 4),
        "expectancy": round(expectancy, 4),
    }

    return metrics


def trade_summary(metrics: Dict) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append("  STATISTICAL TRADING BACKTEST REPORT")
    lines.append("=" * 60)
    lines.append(f"  Total Return:       {metrics['total_return_pct']:>10.2f}%")
    lines.append(f"  Final Value:        ${metrics['final_value']:>10,.2f}")
    lines.append(f"  Sharpe Ratio:       {metrics['sharpe_ratio']:>10.3f}")
    lines.append(f"  Sortino Ratio:      {metrics['sortino_ratio']:>10.3f}")
    lines.append(f"  Calmar Ratio:       {metrics['calmar_ratio']:>10.3f}")
    lines.append(f"  Max Drawdown:       {metrics['max_drawdown_pct']:>10.2f}%")
    lines.append(f"  Number of Trades:   {metrics['num_trades']:>10}")
    lines.append(f"  Win Rate:           {metrics['win_rate_pct']:>10.2f}%")
    lines.append(f"  Profit Factor:      {metrics['profit_factor']:>10.2f}")
    lines.append(f"  Expectancy:         ${metrics['expectancy']:>10.4f}")
    lines.append("=" * 60)
    return "\n".join(lines)
