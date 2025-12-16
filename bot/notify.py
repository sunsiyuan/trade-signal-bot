"""
Notification helpers for Telegram, WeChat (Server酱) and generic webhooks.

Keys/tokens are intentionally left blank for the user to fill in.

本文件定位：
- 纯通知渠道封装（I/O 边界层），不做策略判断，不做去重
- 上游（main / dispatcher）负责决定 “发不发、发到哪”
- 这里负责 “怎么发、失败了怎么兜底（返回 False 而不是抛异常）”
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import requests


@dataclass
class Notifier:
    """
    Notifier = 多渠道通知器，支持：
    - Telegram Bot
    - Server酱（ftqq，微信推送）
    - Generic webhook（自定义 HTTP endpoint）

    设计特点：
    1) 所有凭证都是 Optional，没配就跳过，不会报错
    2) send() 会把“所有已配置渠道”都尝试发一遍，返回每个渠道是否成功
       这使得上游可以：
       - 记录日志
       - 做失败重试（如果你未来想加）
       - 或做降级（例如 webhook 失败就只发 TG）
    3) 复用 requests.Session：减少连接开销，在 Actions/长进程里更稳定
    """

    # Telegram bot token（形如 123456:ABC...）
    telegram_token: Optional[str] = None

    # Telegram chat id（可以是个人 chat_id 或群组 id）
    telegram_chat_id: Optional[str] = None

    # Server酱 key（ftqq 的 send key）
    ftqq_key: Optional[str] = None

    # 通用 webhook URL（你可以接 Slack/飞书/自建服务）
    webhook_url: Optional[str] = None

    # 复用 Session（连接池 + keep-alive）。default_factory 避免 dataclass 的共享可变默认值坑。
    session: requests.Session = field(default_factory=requests.Session)

    def has_channels(self) -> bool:
        """
        用于上游快速判断：是否配置了至少一个可发送渠道。
        常见用途：
        - 本地开发时未配置 token，避免误以为“发出去了”
        - main.py 可以在启动时提醒用户配置
        """
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
        """
        向所有“已配置渠道”发送消息。

        参数：
        - message：正文（对 Telegram/webhook 就是 text；对 ftqq 是 desp）
        - title：仅用于 Server酱（ftqq）的 title 字段
        - include_ftqq：上游控制是否发微信推送
          你之前的产品设计里，很可能是：
          - WATCH/summary：不打扰 -> include_ftqq=False
          - LIMIT_4H / EXECUTE_NOW：强提醒 -> include_ftqq=True

        返回：
        - {"telegram": True/False, "wechat_ftqq": True/False, "webhook": True/False}
        - 未配置的渠道不会出现在 results 里（等于“跳过”而不是 False）
        """
        results: Dict[str, bool] = {}

        # Telegram：必须同时有 token + chat_id 才发送
        if self.telegram_token and self.telegram_chat_id:
            results["telegram"] = self.send_telegram(message)

        # Server酱：需要 include_ftqq=True 且 ftqq_key 存在
        if include_ftqq and self.ftqq_key:
            results["wechat_ftqq"] = self.send_wechat_ftqq(title=title, message=message)

        # Webhook：只要 webhook_url 存在就发送
        if self.webhook_url:
            results["webhook"] = self.send_webhook(message)

        return results

    def send_telegram(
        self,
        message: str,
        *,
        token: Optional[str] = None,
        chat_id: Optional[str] = None,
    ) -> bool:
        """
        Telegram 发送实现（单渠道）。

        设计点：
        - 支持 per-call 覆盖 token/chat_id：
          这样你可以在同一个 Notifier 实例上，发到不同 bot / 不同 chat
          （比如 summary bot 和 action bot 用两个 token）
        - 网络异常捕获 requests.RequestException：
          → 返回 False，不中断主流程（非常重要：通知失败不能让交易逻辑崩）

        Telegram API 选择：
        - /sendMessage
        - disable_web_page_preview=True：避免消息里有链接时触发网页预览，影响可读性
        """
        final_token = token or self.telegram_token
        final_chat_id = chat_id or self.telegram_chat_id

        if not final_token or not final_chat_id:
            return False

        url = f"https://api.telegram.org/bot{final_token}/sendMessage"
        payload = {
            "chat_id": final_chat_id,
            "text": message,
            "disable_web_page_preview": True,
        }

        try:
            response = self.session.post(url, json=payload, timeout=10)
        except requests.RequestException:
            return False

        # response.ok = HTTP 2xx
        return response.ok

    def send_telegram_with(self, token: str, chat_id: str, message: str) -> bool:
        """
        帮助函数：明确指定 token/chat_id，但不修改 self.telegram_token/self.telegram_chat_id。

        用途：
        - 你想在 Notifier 内“临时用另一个 bot 发消息”，又不想污染默认配置
        - 比如：summary bot / action bot 分流（你之前提到会这么做）
        """
        return self.send_telegram(message, token=token, chat_id=chat_id)

    def send_wechat_ftqq(self, title: str, message: str) -> bool:
        """
        Server酱（ftqq）发送实现。

        这里用的是 sctapi.ftqq.com 的新接口：
        - url: https://sctapi.ftqq.com/{SENDKEY}.send
        - payload: title / desp

        同样：
        - 网络异常返回 False，不抛异常
        - response.ok 作为成功判定
        """
        if not self.ftqq_key:
            return False

        url = f"https://sctapi.ftqq.com/{self.ftqq_key}.send"
        payload = {"title": title, "desp": message}

        try:
            response = self.session.post(url, data=payload, timeout=10)
        except requests.RequestException:
            return False

        return response.ok

    def send_webhook(self, message: str) -> bool:
        """
        通用 webhook 发送实现。

        约定 payload 结构极简：
        - {"text": message}

        这样你可以在下游 webhook 服务里做转换：
        - 转 Slack/Discord/飞书/企业微信
        - 或者直接写入你自己的日志服务 / DB

        同样：网络异常 -> False，不中断主流程。
        """
        if not self.webhook_url:
            return False

        payload = {"text": message}

        try:
            response = self.session.post(self.webhook_url, json=payload, timeout=10)
        except requests.RequestException:
            return False

        return response.ok


# 对外导出（避免 from notify import * 时带出 requests 等杂物）
__all__ = ["Notifier"]
