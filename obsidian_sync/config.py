"""ObsidianConfig dataclass and .env loading."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _default_vault_path() -> str:
    env_vault = os.environ.get("OBSIDIAN_VAULT_PATH")
    if env_vault:
        return env_vault
    home = Path(os.path.expanduser("~"))
    for candidate in [
        home / "Obsidian" / "TTRG",
        home / "Documents" / "Obsidian" / "TTRG",
    ]:
        if candidate.exists():
            return str(candidate)
    return str(home / "Obsidian" / "TTRG")


def _env_enabled() -> bool:
    return os.environ.get("OBSIDIAN_ENABLED", "").lower() in ("1", "true", "yes", "on")


@dataclass
class ObsidianConfig:
    enabled: bool = field(default_factory=_env_enabled)
    vault_path: str = field(default_factory=_default_vault_path)
    project_folder: str = "20-Research/GOLD-Trading"
    trade_folder: str = "Trades"
    signal_folder: str = "Signals"
    backtest_folder: str = "Backtests"
    daily_folder: str = "Daily"
    dashboard_file: str = "Dashboard.md"
    templates_subdir: str = ".templates"
    async_writes: bool = True
    queue_max: int = 1000

    def base(self) -> Path:
        return Path(self.vault_path) / self.project_folder

    def trade_dir(self) -> Path:
        return self.base() / self.trade_folder

    def signal_dir(self) -> Path:
        return self.base() / self.signal_folder

    def backtest_dir(self) -> Path:
        return self.base() / self.backtest_folder

    def daily_dir(self) -> Path:
        return self.base() / self.daily_folder

    def dashboard_path(self) -> Path:
        return self.base() / self.dashboard_file

    def template_dir(self) -> Path:
        return self.base() / self.templates_subdir
