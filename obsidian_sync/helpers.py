"""High-level helpers to convert trading events into Obsidian notes."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .writer import get_writer


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def signal_from_engine(
    *,
    trade_id: Any,
    direction: str,
    symbol: str,
    price: float,
    zscore: float,
    atr_ratio: float,
    session: str,
    filled: bool,
    notes: str = "",
) -> None:
    now = datetime.now(timezone.utc)
    ctx = {
        "id": trade_id,
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "direction": direction,
        "symbol": symbol,
        "price": price,
        "zscore": zscore,
        "atr_ratio": atr_ratio,
        "session": session,
        "filled": filled,
        "notes": notes,
    }
    get_writer().write_signal(ctx)


def trade_from_engine(
    *,
    trade_id: Any,
    direction: str,
    symbol: str,
    entry: float,
    exit_price: float,
    stop: Optional[float],
    pnl_usd: float,
    pnl_pips: float,
    zscore_entry: float,
    zscore_exit: float,
    duration_bars: int,
    session: str,
    exit_reason: str,
    open_time: Optional[datetime] = None,
    close_time: Optional[datetime] = None,
    notes: str = "",
) -> None:
    open_time = open_time or datetime.now(timezone.utc)
    close_time = close_time or datetime.now(timezone.utc)
    ctx = {
        "id": trade_id,
        "date": open_time.strftime("%Y-%m-%d"),
        "open_time": open_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "close_time": close_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "direction": direction,
        "symbol": symbol,
        "entry": entry,
        "exit": exit_price,
        "stop": stop if stop is not None else "",
        "pnl_usd": pnl_usd,
        "pnl_pips": pnl_pips,
        "zscore_entry": zscore_entry,
        "zscore_exit": zscore_exit,
        "duration_bars": duration_bars,
        "session": session,
        "exit_reason": exit_reason,
        "notes": notes,
    }
    get_writer().write_trade(ctx)
    get_writer().append_dashboard_log(
        "trade closed",
        f"{direction} {symbol} pnl=${pnl_usd:.2f} reason={exit_reason}",
    )


def backtest_from_results(
    *,
    strategy: str,
    total_trades: int,
    win_rate: float,
    net_return: float,
    sharpe: float,
    max_drawdown: float,
    notes: str = "",
    date: Optional[str] = None,
) -> None:
    ctx = {
        "date": date or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "strategy": strategy,
        "total_trades": total_trades,
        "win_rate": round(win_rate, 2),
        "net_return": round(net_return, 4),
        "sharpe": round(sharpe, 3),
        "max_drawdown": round(max_drawdown, 4),
        "notes": notes,
    }
    get_writer().write_backtest(ctx)


def daily_summary(
    *,
    date: Optional[str] = None,
    total_trades: int,
    net_pnl: float,
    win_rate: float,
    best_trade: float,
    worst_trade: float,
) -> None:
    body = (
        f"- Trades: **{total_trades}**\n"
        f"- Net P&L: **${net_pnl:.2f}**\n"
        f"- Win rate: **{win_rate:.1f}%**\n"
        f"- Best: **${best_trade:.2f}** | Worst: **${worst_trade:.2f}**\n"
    )
    ctx = {"date": date, "body": body}
    get_writer().append_daily(ctx)


def hourly_snapshot(
    *,
    account: str,
    balance: float,
    equity: float,
    open_pnl: float,
    win_rate: float,
    trades_today: int,
) -> None:
    get_writer().append_dashboard_snapshot(
        {
            "account": account,
            "balance": f"{balance:.2f}",
            "equity": f"{equity:.2f}",
            "open_pnl": f"{open_pnl:.2f}",
            "win_rate": f"{win_rate:.1f}%",
            "trades_today": trades_today,
        }
    )
