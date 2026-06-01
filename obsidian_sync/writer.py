"""ObsidianWriter: background-thread Markdown writer to an Obsidian vault.

Writes are queued and executed by a single daemon thread so the trading
loop is never blocked by disk I/O. If async_writes is False, writes are
synchronous.
"""

from __future__ import annotations

import logging
import queue
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .config import ObsidianConfig

logger = logging.getLogger("obsidian_sync.writer")


def _slug(value: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-")
    return s or "x"


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _fmt_ts(dt: Optional[datetime] = None) -> str:
    dt = dt or _now_utc()
    return dt.strftime("%Y-%m-%d_%H%M%S")


def _safe(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if v != v:
            return ""
        return f"{v:.4f}".rstrip("0").rstrip(".") or "0"
    return str(v)


def _render_template(text: str, ctx: Dict[str, Any]) -> str:
    def repl(m):
        key = m.group(1).strip()
        return _safe(ctx.get(key, m.group(0)))
    return re.sub(r"\{\{\s*([A-Za-z0-9_]+)\s*\}\}", repl, text)


@dataclass
class _Job:
    kind: str
    payload: Dict[str, Any]


class ObsidianWriter:
    """Threaded Markdown writer to an Obsidian vault."""

    def __init__(self, cfg: ObsidianConfig):
        self.cfg = cfg
        self._q: "queue.Queue[_Job]" = queue.Queue(maxsize=cfg.queue_max)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._enabled = bool(cfg.enabled)
        self._vault_ok = False
        if self._enabled:
            self._ensure_vault_paths()
            self._vault_ok = self.base_ok()
            if not self._vault_ok:
                logger.warning(
                    "ObsidianWriter: vault path missing or unwritable: %s",
                    self.cfg.base(),
                )

    def base_ok(self) -> bool:
        try:
            self.cfg.base().mkdir(parents=True, exist_ok=True)
            test = self.cfg.base() / ".write_test"
            test.write_text("ok", encoding="utf-8")
            test.unlink(missing_ok=True)
            return True
        except Exception as e:
            logger.error("ObsidianWriter: vault check failed: %s", e)
            return False

    def _ensure_vault_paths(self) -> None:
        for p in [
            self.cfg.base(),
            self.cfg.trade_dir(),
            self.cfg.signal_dir(),
            self.cfg.backtest_dir(),
            self.cfg.daily_dir(),
            self.cfg.template_dir(),
        ]:
            try:
                p.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.error("ObsidianWriter: cannot create %s: %s", p, e)

    def start(self) -> None:
        if not self._enabled or not self._vault_ok:
            logger.info("ObsidianWriter disabled or vault unavailable.")
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="ObsidianWriter", daemon=True
        )
        self._thread.start()
        logger.info("ObsidianWriter started (vault=%s)", self.cfg.base())

    def stop(self, timeout: float = 10.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._q.empty():
                break
            time.sleep(0.05)
        self._stop.set()
        try:
            self._q.put_nowait(_Job(kind="__stop__", payload={}))
        except Exception:
            pass
        if self._thread:
            self._thread.join(timeout=timeout)
        logger.info("ObsidianWriter stopped.")

    def _enqueue(self, kind: str, payload: Dict[str, Any]) -> None:
        if not self._enabled or not self._vault_ok:
            return
        if not self.cfg.async_writes:
            self._handle(_Job(kind=kind, payload=payload))
            return
        try:
            self._q.put_nowait(_Job(kind=kind, payload=payload))
        except queue.Full:
            logger.warning("ObsidianWriter queue full, dropping %s", kind)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                job = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            if job.kind == "__stop__":
                break
            try:
                self._handle(job)
            except Exception as e:
                logger.error("ObsidianWriter job %s failed: %s", job.kind, e)

    def _handle(self, job: _Job) -> None:
        kind = job.kind
        if kind == "signal":
            self._do_write_signal(job.payload)
        elif kind == "trade":
            self._do_write_trade(job.payload)
        elif kind == "backtest":
            self._do_write_backtest(job.payload)
        elif kind == "daily":
            self._do_append_daily(job.payload)
        elif kind == "dashboard_log":
            self._do_append_dashboard_log(job.payload)
        elif kind == "dashboard_snapshot":
            self._do_dashboard_snapshot(job.payload)
        else:
            logger.warning("ObsidianWriter: unknown job %s", kind)

    def _read_template(self, name: str) -> Optional[str]:
        p = self.cfg.template_dir() / name
        if p.exists():
            try:
                return p.read_text(encoding="utf-8")
            except Exception as e:
                logger.error("template read failed %s: %s", p, e)
        return None

    def _do_write_signal(self, ctx: Dict[str, Any]) -> None:
        tpl = self._read_template("signal.md")
        if tpl is None:
            tpl = (
                "---\ntype: signal\ndate: {{date}}\ntime: {{time}}\n"
                "direction: {{direction}}\nsymbol: {{symbol}}\n"
                "zscore: {{zscore}}\natr_ratio: {{atr_ratio}}\n"
                "session: {{session}}\nfilled: {{filled}}\n"
                "tags: [signal, {{direction}}]\n---\n\n"
                "# Signal — {{direction}} @ {{price}}\n"
            )
        body = _render_template(tpl, ctx)
        ts = _fmt_ts()
        fname = f"{ts}_ENTRY-{_slug(ctx.get('id', 'x'))}.md"
        path = self.cfg.signal_dir() / fname
        path.write_text(body, encoding="utf-8")
        logger.info("signal note: %s", path)

    def _do_write_trade(self, ctx: Dict[str, Any]) -> None:
        tpl = self._read_template("trade.md")
        if tpl is None:
            tpl = (
                "---\ntype: trade\ndate: {{date}}\ndirection: {{direction}}\n"
                "symbol: {{symbol}}\nentry: {{entry}}\nexit: {{exit}}\n"
                "pnl_usd: {{pnl_usd}}\npnl_pips: {{pnl_pips}}\n"
                "zscore_entry: {{zscore_entry}}\nzscore_exit: {{zscore_exit}}\n"
                "duration_bars: {{duration_bars}}\nsession: {{session}}\n"
                "exit_reason: {{exit_reason}}\ntags: [trade, {{direction}}]\n---\n"
            )
        body = _render_template(tpl, ctx)
        ts = _fmt_ts()
        fname = f"{ts}_TRADE-{_slug(ctx.get('id', 'x'))}.md"
        path = self.cfg.trade_dir() / fname
        path.write_text(body, encoding="utf-8")
        logger.info("trade note: %s", path)

    def _do_write_backtest(self, ctx: Dict[str, Any]) -> None:
        tpl = self._read_template("backtest.md")
        if tpl is None:
            tpl = (
                "---\ntype: backtest\ndate: {{date}}\nstrategy: {{strategy}}\n"
                "total_trades: {{total_trades}}\nwin_rate: {{win_rate}}\n"
                "net_return: {{net_return}}\nsharpe: {{sharpe}}\n"
                "max_drawdown: {{max_drawdown}}\ntags: [backtest]\n---\n"
            )
        body = _render_template(tpl, ctx)
        date = ctx.get("date") or _now_utc().strftime("%Y-%m-%d")
        fname = f"{date}_backtest.md"
        path = self.cfg.backtest_dir() / fname
        existing = path.read_text(encoding="utf-8") if path.exists() else ""
        sep = "\n\n---\n\n"
        path.write_text(existing + (sep if existing else "") + body, encoding="utf-8")
        logger.info("backtest note: %s", path)

    def _do_append_daily(self, ctx: Dict[str, Any]) -> None:
        date = ctx.get("date") or _now_utc().strftime("%Y-%m-%d")
        path = self.cfg.daily_dir() / f"{date}.md"
        if not path.exists():
            header = (
                f"---\ntype: daily\ndate: {date}\n"
                f"total_trades: 0\nnet_pnl: 0\nwin_rate: 0\ntags: [daily]\n---\n\n"
                f"# Daily Summary — {date}\n"
            )
            path.write_text(header, encoding="utf-8")
        ts = _now_utc().strftime("%H:%M:%S")
        line = f"\n## {ts} UTC\n\n{ctx.get('body', '')}\n"
        with path.open("a", encoding="utf-8") as f:
            f.write(line)
        logger.info("daily note appended: %s", path)

    def _do_append_dashboard_log(self, ctx: Dict[str, Any]) -> None:
        path = self.cfg.dashboard_path()
        if not path.exists():
            path.write_text(
                "---\ntype: dashboard\nproject: GOLD-Trading\ntags: [dashboard, live]\n---\n\n# GOLD Trading — Live Dashboard\n",
                encoding="utf-8",
            )
        ts = _now_utc().strftime("%Y-%m-%d %H:%M:%S")
        line = f"\n- `{ts} UTC` | **{_safe(ctx.get('event'))}** | {_safe(ctx.get('detail'))}\n"
        with path.open("a", encoding="utf-8") as f:
            f.write(line)
        logger.info("dashboard log: %s | %s", ctx.get("event"), ctx.get("detail"))

    def _do_dashboard_snapshot(self, ctx: Dict[str, Any]) -> None:
        path = self.cfg.dashboard_path()
        if not path.exists():
            path.write_text(
                "---\ntype: dashboard\nproject: GOLD-Trading\ntags: [dashboard, live]\n---\n\n# GOLD Trading — Live Dashboard\n",
                encoding="utf-8",
            )
        ts = _now_utc().strftime("%Y-%m-%d %H:%M:%S")
        snap = (
            f"\n## Snapshot @ {ts} UTC\n\n"
            f"| Field | Value |\n|---|---|\n"
            f"| Account | {_safe(ctx.get('account'))} |\n"
            f"| Balance | {_safe(ctx.get('balance'))} |\n"
            f"| Equity | {_safe(ctx.get('equity'))} |\n"
            f"| Open P&L | {_safe(ctx.get('open_pnl'))} |\n"
            f"| Win rate (rolling 20) | {_safe(ctx.get('win_rate'))} |\n"
            f"| Trades today | {_safe(ctx.get('trades_today'))} |\n\n"
        )
        with path.open("a", encoding="utf-8") as f:
            f.write(snap)
        logger.info("dashboard snapshot appended")

    def write_signal(self, ctx: Dict[str, Any]) -> None:
        self._enqueue("signal", ctx)

    def write_trade(self, ctx: Dict[str, Any]) -> None:
        self._enqueue("trade", ctx)

    def write_backtest(self, ctx: Dict[str, Any]) -> None:
        self._enqueue("backtest", ctx)

    def append_daily(self, ctx: Dict[str, Any]) -> None:
        self._enqueue("daily", ctx)

    def append_dashboard_log(self, event: str, detail: str = "") -> None:
        self._enqueue("dashboard_log", {"event": event, "detail": detail})

    def append_dashboard_snapshot(self, ctx: Dict[str, Any]) -> None:
        self._enqueue("dashboard_snapshot", ctx)

    def open_in_obsidian(self, subpath: str = "") -> str:
        """Return an obsidian:// URI for a vault-relative path."""
        v = self.cfg.vault_path.replace("\\", "/")
        name = Path(v).name
        rel = subpath.replace("\\", "/").lstrip("/")
        if rel:
            return f"obsidian://open?vault={name}&file={rel}"
        return f"obsidian://open?vault={name}"


_writer: Optional[ObsidianWriter] = None
_writer_lock = threading.Lock()


def get_writer(cfg: Optional[ObsidianConfig] = None) -> ObsidianWriter:
    global _writer
    with _writer_lock:
        if _writer is None:
            if cfg is None:
                from .config import ObsidianConfig as _OC
                cfg = _OC()
            _writer = ObsidianWriter(cfg)
        return _writer
