"""
V1-Silver Telegram Notifier
Same as base notifier but prefixes all messages with [V1-Silver].
"""
from notifications.telegram import TelegramNotifier


class SilverTelegramNotifier(TelegramNotifier):
    """Telegram notifier that tags all messages as [V1-Silver]."""

    def _send(self, text: str, parse_mode: str = "HTML", silent: bool = False) -> bool:
        # Prefix every message with the module tag
        if not text.startswith("[V1-Silver]"):
            text = f"[V1-Silver]\n{text}"
        return super()._send(text, parse_mode, silent)
