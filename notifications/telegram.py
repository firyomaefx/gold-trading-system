import urllib.request
import urllib.parse
import json
import os
import time
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class TelegramNotifier:
    _instance = None

    def __init__(self, token: str = None, chat_id: str = None):
        self.token = token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        self.base_url = f"https://api.telegram.org/bot{self.token}"
        self._last_hourly = 0

    @classmethod
    def get(cls) -> "TelegramNotifier":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _send(self, text: str, parse_mode: str = "HTML", silent: bool = False) -> bool:
        if not self.token or not self.chat_id:
            logger.warning("Telegram not configured (missing token or chat_id)")
            return False

        url = f"{self.base_url}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_notification": silent,
        }

        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                if not result.get("ok"):
                    logger.error(f"Telegram API error: {result}")
                    return False
                return True
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return False

    def _escape_html(self, text: str) -> str:
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def send_entry_alert(self, direction: str, price: float, zscore: float,
                         hurst: float, volume: float, sl: float, reason: str = "") -> bool:
        arrow = "\u2191" if direction.upper() == "BUY" else "\u2193"
        emoji = "\U0001f7e2" if direction.upper() == "BUY" else "\U0001f534"

        msg = (
            f"{emoji} <b>ENTRY ALERT</b> {emoji}\n"
            f"\n"
            f"<b>{arrow} {self._escape_html(direction.upper())}</b>\n"
            f"<b>Price:</b> {price:.2f}\n"
            f"<b>Volume:</b> {volume} lots\n"
            f"<b>SL:</b> {sl:.2f}\n"
            f"\n"
            f"<b>Z-Score:</b> {zscore:.2f}\u03c3\n"
            f"<b>Hurst:</b> {hurst:.3f}\n"
        )
        if reason:
            msg += f"<b>Reason:</b> {self._escape_html(reason)}\n"

        msg += f"\n<code>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</code>"

        return self._send(msg)

    def send_exit_alert(self, direction: str, entry_price: float, exit_price: float,
                        pnl: float, reason: str = "") -> bool:
        emoji = "\U0001f7e2" if pnl >= 0 else "\U0001f534"
        pnl_str = f"+{pnl:.2f}" if pnl >= 0 else f"{pnl:.2f}"

        msg = (
            f"{emoji} <b>EXIT ALERT</b> {emoji}\n"
            f"\n"
            f"<b>Close {self._escape_html(direction.upper())}</b>\n"
            f"<b>Entry:</b> {entry_price:.2f}\n"
            f"<b>Exit:</b> {exit_price:.2f}\n"
            f"<b>P&amp;L:</b> ${pnl_str}\n"
        )
        if reason:
            msg += f"<b>Reason:</b> {self._escape_html(reason)}\n"

        msg += f"\n<code>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</code>"

        return self._send(msg)

    def send_dashboard_snapshot(self, data: dict) -> bool:
        now = datetime.now().strftime("%H:%M:%S")

        signal = data.get("signal_text", "WAIT")
        zscore = data.get("zscore", 0)
        hurst = data.get("hurst", 0.5)
        hmm_prob = data.get("hmm_prob", 0.5)
        regime = data.get("hurst_regime", "N/A")
        bid = data.get("bid", 0)
        spread = data.get("spread", 0)
        equity = data.get("account_equity", 0)
        pnl = data.get("total_pnl", 0)
        has_pos = data.get("has_position", False)

        zscore_v = f"{zscore:.2f}" if not (isinstance(zscore, float) and zscore != zscore) else "N/A"
        hurst_v = f"{hurst:.3f}" if not (isinstance(hurst, float) and hurst != hurst) else "N/A"

        hmm_str = f"{hmm_prob*100:.0f}% ranging" if hmm_prob >= 0.5 else f"{(1-hmm_prob)*100:.0f}% trending"

        pnl_str = f"+{pnl:.2f}" if pnl >= 0 else f"{pnl:.2f}"
        pnl_emoji = "\U0001f7e2" if pnl >= 0 else "\U0001f534"

        msg = (
            f"\U0001f4ca <b>HOURLY SNAPSHOT</b>\n"
            f"\n"
            f"<b>Signal:</b> {self._escape_html(signal)}\n"
            f"<b>Z-Score:</b> {zscore_v}\n"
            f"<b>Hurst:</b> {hurst_v} ({self._escape_html(regime)})\n"
            f"<b>HMM:</b> {self._escape_html(hmm_str)}\n"
            f"\n"
            f"<b>Bid:</b> {bid:.2f} | <b>Spread:</b> {spread}pts\n"
            f"<b>Equity:</b> ${equity:.2f}\n"
            f"<b>P&amp;L:</b> {pnl_emoji} ${pnl_str}\n"
        )

        if has_pos:
            pos_type = data.get("position_type", "").upper()
            pos_vol = data.get("position_volume", 0)
            pos_pnl = data.get("position_pnl", 0)
            pos_pnl_s = f"+{pos_pnl:.2f}" if pos_pnl >= 0 else f"{pos_pnl:.2f}"
            msg += (
                f"\n<b>Position:</b> {self._escape_html(pos_type)} {pos_vol} lots\n"
                f"<b>Pos P&amp;L:</b> ${pos_pnl_s}\n"
            )

        # Daily / Weekly / Monthly performance
        daily = data.get("daily", "")
        weekly = data.get("weekly", "")
        monthly = data.get("monthly", "")
        if daily or weekly or monthly:
            msg += "\n<b>Performance:</b>\n"
            if daily:
                msg += f"  {self._escape_html(daily)}\n"
            if weekly:
                msg += f"  {self._escape_html(weekly)}\n"
            if monthly:
                msg += f"  {self._escape_html(monthly)}\n"

        msg += f"\n<code>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</code>"

        return self._send(msg, silent=True)

    def should_send_hourly(self) -> bool:
        now = time.time()
        # Use file-based persistence so restarts don't reset the timer
        stamp_file = os.path.join(os.path.dirname(__file__), ".last_hourly")
        last = 0
        try:
            if os.path.exists(stamp_file):
                with open(stamp_file, "r") as f:
                    last = float(f.read().strip())
        except Exception:
            pass

        if now - last >= 3600:
            try:
                with open(stamp_file, "w") as f:
                    f.write(str(now))
            except Exception:
                pass
            self._last_hourly = now
            return True
        self._last_hourly = last
        return False

    def send_error(self, error_msg: str) -> bool:
        msg = (
            f"\u26a0\ufe0f <b>ERROR</b>\n"
            f"\n"
            f"{self._escape_html(error_msg)}\n"
            f"\n<code>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</code>"
        )
        return self._send(msg)

    def send_image(self, chat_id: str, image_path: str, caption: str = "") -> bool:
        """Send an image to Telegram."""
        if not self.token:
            logger.warning("Telegram not configured (missing token)")
            return False

        url = f"{self.base_url}/sendPhoto"
        
        try:
            # Build multipart form data
            boundary = "----WebKitFormBoundary7MA4YWxk"
            body = []
            body.append(f"--{boundary}".encode())
            body.append(f'Content-Disposition: form-data; name="chat_id"'.encode())
            body.append(b"")
            body.append(chat_id.encode())
            
            body.append(f"--{boundary}".encode())
            body.append(f'Content-Disposition: form-data; name="caption"'.encode())
            body.append(b"")
            body.append(caption.encode("utf-8"))
            
            body.append(f"--{boundary}".encode())
            body.append(f'Content-Disposition: form-data; name="photo"; filename="{os.path.basename(image_path)}"'.encode())
            body.append(b"Content-Type: image/png")
            body.append(b"")
            
            with open(image_path, "rb") as f:
                body.append(f.read())
            
            body.append(f"--{boundary}--".encode())
            
            data = b"\r\n".join(body)
            
            req = urllib.request.Request(
                url,
                data=data,
                headers={
                    "Content-Type": f"multipart/form-data; boundary={boundary}",
                },
            )
            
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                if not result.get("ok"):
                    logger.error(f"Telegram API error: {result}")
                    return False
                return True
        except Exception as e:
            logger.error(f"Telegram send_image failed: {e}")
            return False

    def send_startup(self) -> bool:
        msg = (
            f"\U0001f680 <b>TRADING SYSTEM STARTED</b>\n"
            f"\n"
            f"GOLD-Pro V1 Live\n"
            f"HMM gate: DISABLED\n"
            f"Interval: 60s\n"
            f"\n<code>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</code>"
        )
        return self._send(msg)

    def send_shutdown(self, summary: str = "") -> bool:
        msg = (
            f"\U0001f6d1 <b>TRADING SYSTEM STOPPED</b>\n"
        )
        if summary:
            msg += f"\n<code>{self._escape_html(summary)}</code>\n"
        msg += f"\n<code>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</code>"
        return self._send(msg)