"""Obsidian vault bridge for the GOLD trading system.

Exports trades, signals, backtests, and daily summaries as Markdown
notes into the user's Obsidian vault.

Public API:
    from obsidian_sync import ObsidianWriter
    w = ObsidianWriter(GOLD_CONFIG.obsidian)
    w.start()
    w.write_signal({...})
    w.write_trade({...})
    w.write_backtest({...})
    w.append_daily({...})
    w.append_dashboard_log("event", "detail")
    w.stop()
"""

from .config import ObsidianConfig
from .writer import ObsidianWriter, get_writer

__all__ = ["ObsidianConfig", "ObsidianWriter", "get_writer"]
