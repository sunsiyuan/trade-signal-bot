"""Notification helpers for Telegram, WeChat (Server酱) and generic webhooks.

Keys/tokens are intentionally left blank for the user to fill in.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import requests


@dataclass
class Notifier:
    telegram_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    ftqq_key: Optional[str] = None
    webhook_url: Optional[str] = None
    session: requests.Session = field(default_factory=requests.Session)

    def has_channels(self) -> bool:
        """Return True if at least one channel is configured."""

        return bool(
            (self.telegram_token and self.telegram_chat_id)
            or self.ftqq_key
            or self.webhook_url
        )

    def send(
        self,
        message: str,
        title: str = "Trade Signal",
        include_ftqq: bool = True,
    ) -> Dict[str, bool]:
        """Send notifications to all configured channels.

        ``include_ftqq`` allows callers to suppress Server酱 pushes for
        low-priority/status updates while still emitting Telegram/webhook
        messages.

        Returns a mapping of channel name to success flag. Channels without
        configuration are skipped.
        """

        results: Dict[str, bool] = {}

        if self.telegram_token and self.telegram_chat_id:
            results["telegram"] = self.send_telegram(message)

        if include_ftqq and self.ftqq_key:
            results["wechat_ftqq"] = self.send_wechat_ftqq(title=title, message=message)

        if self.webhook_url:
            results["webhook"] = self.send_webhook(message)

        return results

    def send_telegram(self, message: str) -> bool:
        """Send a message via Telegram bot.

        Configure ``telegram_token`` and ``telegram_chat_id`` before calling.
        """

        url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
        payload = {
            "chat_id": self.telegram_chat_id,
            "text": message,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        }

        try:
            response = self.session.post(url, json=payload, timeout=10)
        except requests.RequestException:
            return False

        return response.ok

    def send_wechat_ftqq(self, title: str, message: str) -> bool:
        """Send a notification through Server酱 (ftqq)."""

        url = f"https://sctapi.ftqq.com/{self.ftqq_key}.send"
        payload = {"title": title, "desp": message}

        try:
            response = self.session.post(url, data=payload, timeout=10)
        except requests.RequestException:
            return False

        return response.ok

    def send_webhook(self, message: str) -> bool:
        """Send a generic webhook notification with a simple JSON payload."""

        payload = {"text": message}

        try:
            response = self.session.post(self.webhook_url, json=payload, timeout=10)
        except requests.RequestException:
            return False

        return response.ok


__all__ = ["Notifier"]
