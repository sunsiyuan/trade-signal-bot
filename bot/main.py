# bot/main.py
from datetime import timezone, timedelta

from .config import Settings
from .data_client import HyperliquidDataClient
from .notify import Notifier
from .signal_engine import SignalEngine


def print_signal(signal):
    snap = signal.snapshot

    print("====== TRADE SIGNAL ======")
    utc_ts = snap.ts.astimezone(timezone.utc)
    beijing_ts = utc_ts.astimezone(timezone(timedelta(hours=8)))
    print(f"Time:   {utc_ts.isoformat()} (UTC)")
    print(f"        {beijing_ts.isoformat()} (Beijing)")
    print(f"Symbol: {signal.symbol}")
    print()

    # 1）多周期指标
    def tf_row(tf):
        return (
            f"{tf.timeframe:>4} | "
            f"close={tf.close:.4f} | "
            f"MA7={tf.ma7:.2f}, MA25={tf.ma25:.2f}, MA99={tf.ma99:.2f} | "
            f"RSI6={tf.rsi6:.2f}, RSI12={tf.rsi12:.2f}, RSI24={tf.rsi24:.2f} | "
            f"MACD={tf.macd:.4f}, SIG={tf.macd_signal:.4f}, HIST={tf.macd_hist:.4f} | "
            f"ATR={tf.atr:.4f} | Trend={tf.trend_label}"
        )

    print("=== Timeframe Indicators ===")
    print(tf_row(snap.tf_4h))
    print(tf_row(snap.tf_1h))
    print(tf_row(snap.tf_15m))
    print()

    # 2）衍生品指标
    d = snap.deriv
    print("=== Derivative Indicators ===")
    oi_change_text = "N/A" if d.oi_change_24h is None else f"{d.oi_change_24h:.2f}"
    print(
        f"Funding={d.funding:.6f}, "
        f"OI={d.open_interest:.2f}, "
        f"OI_24h_change={oi_change_text}, "
        f"Liquidity={d.liquidity_comment}"
    )
    # 3）信号结果
    print("=== Trade Signal ===")
    print(f"Direction : {signal.direction}")
    print(f"Confidence: {signal.confidence:.2f}")
    print(f"Reason    : {signal.reason}")

    if signal.direction != "none":
        print(f"Entry range: {signal.entry_range}")
        print(f"TP1: {signal.tp1} | TP2: {signal.tp2} | SL: {signal.sl}")
        print(
            f"Position: core={signal.core_position_pct * 100:.0f}% "
            f"+ add={signal.add_position_pct * 100:.0f}%"
        )
    else:
        print("暂无交易信号，等待下一次机会。")
    print("===============================")


def format_notification(signal, threshold: float = 0.8):
    """Render a Markdown Telegram/webhook notification."""

    snap = signal.snapshot

    if not snap:
        return "Market snapshot unavailable; awaiting next refresh."

    def _fmt_price(value: float) -> str:
        if value is None:
            return "N/A"
        return f"{value:.4f}" if value and abs(value) < 100 else f"{value:.2f}"

    def _macd_state(tf) -> str:
        if tf.macd_hist > 0 and tf.macd > tf.macd_signal:
            return "Bullish"
        if tf.macd_hist < 0 and tf.macd < tf.macd_signal:
            return "Bearish"
        return "Neutral"

    def _trend_phrase() -> tuple[str, str]:
        if not snap:
            return "❔", "No snapshot"

        mapping = {"up": "Uptrend", "down": "Downtrend", "range": "Sideway"}
        emoji_map = {"up": "🚀", "down": "🔻", "range": "⚖️"}
        trend = snap.tf_4h.trend_label
        return emoji_map.get(trend, "❔"), mapping.get(trend, "Undefined")

    def _decision_text() -> tuple[str, str]:
        if signal.direction == "long":
            return "🟢", "Bias Long"
        if signal.direction == "short":
            return "🔴", "Bias Short"
        return "🧊", "Stay on sidelines"

    def _macd_summary(tf):
        return f"{_macd_state(tf)} ({'+' if tf.macd_hist >= 0 else ''}{tf.macd_hist:.4f})"

    def _atr_comment():
        if not snap:
            return "N/A"
        ratio = snap.tf_15m.atr / max(snap.tf_1h.atr, 1e-6)
        if ratio > 1.3:
            return "High intraday volatility"
        if ratio < 0.8:
            return "Calm tape"
        return "Normal swings"

    def _oi_state():
        change = snap.deriv.oi_change_24h
        if change is None:
            return "N/A"
        if change > 1.5:
            return "Rising OI"
        if change < -1.5:
            return "Dropping OI"
        return "Flat OI"

    def _liquidity_summary():
        if not snap:
            return "N/A", "N/A"
        asks = sum(order.get("size", 0) for order in snap.deriv.orderbook_asks)
        bids = sum(order.get("size", 0) for order in snap.deriv.orderbook_bids)
        if asks == bids == 0:
            skew = "Balanced"
        elif asks > bids * 1.1:
            skew = "Asks dominant"
        elif bids > asks * 1.1:
            skew = "Bids dominant"
        else:
            skew = "Balanced"
        summary = f"Asks {asks:.0f} vs Bids {bids:.0f}"
        return summary, skew

    def _confidence_label():
        pct = round(signal.confidence * 100)
        suffix = " (High Probability)" if pct > 80 else ""
        return f"{pct}%{suffix}"

    def _execution_type():
        if signal.entry_range and len(signal.entry_range) > 1:
            return "Ladder"
        if signal.entry_range:
            return "Limit"
        return "Market"

    def _format_levels():
        entries = []
        if signal.entry_range:
            entries = [_fmt_price(x) for x in signal.entry_range]
        elif signal.entry:
            entries = [_fmt_price(signal.entry)]

        tps = [tp for tp in [signal.tp1, signal.tp2, signal.tp3] if tp]
        tp_lines = " → ".join(_fmt_price(tp) for tp in tps)

        sl_line = _fmt_price(signal.sl) if signal.sl else None
        return entries, tp_lines, sl_line

    def _contextual_comment(exec_type: str) -> str:
        direction = signal.direction
        if direction == "long":
            if exec_type == "Market":
                return "Execute long now; align size with plan and protect with SL."
            if exec_type == "Ladder":
                return "Layer long orders; watch fills and adjust TP/SL accordingly."
            return "Place long limit; wait for trigger and keep alerts on."
        if direction == "short":
            if exec_type == "Market":
                return "Hit market short; confirm size and trailing plan."
            if exec_type == "Ladder":
                return "Stagger short limits; tighten risk once filled."
            return "Set short limit at trigger and monitor."
        return "No trade trigger yet — stay patient and monitor key levels."

    trend_emoji, trend_phrase = _trend_phrase()
    decision_emoji, decision_text = _decision_text()
    liquidity_summary, liquidity_note = _liquidity_summary()
    entries, tp_lines, sl_line = _format_levels()
    execution_type = _execution_type()
    conf_text = _confidence_label()
    status_mode = signal.direction == "none" or signal.confidence < threshold

    utc_ts = snap.ts.astimezone(timezone.utc)
    beijing_ts = utc_ts.astimezone(timezone(timedelta(hours=8)))
    local_time_string = beijing_ts.strftime("%Y-%m-%d %H:%M")

    header = [
        f"📌 {signal.symbol} — Trade Signal Update",
        f"⏱ {local_time_string} (UTC+8)",
        "",
        f"💰 Price: {_fmt_price(snap.tf_15m.close)}",
        f"📈 Trend: {trend_emoji} {trend_phrase}",
        f"🧩 Decision: {decision_emoji} {decision_text}",
        "",
        "-------------------",
        "",
    ]

    oi_change = snap.deriv.oi_change_24h
    oi_line = (
        f"• OI: {snap.deriv.open_interest:.2f}"
        if oi_change is None
        else f"• OI: {snap.deriv.open_interest:.2f} → {_oi_state()} ({oi_change:+.2f}%)"
    )

    core_metrics = [
        "📍 Core Metrics",
        f"• 4H RSI6: {snap.tf_4h.rsi6:.2f} | MACD: {_macd_summary(snap.tf_4h)}",
        f"• 1H RSI6: {snap.tf_1h.rsi6:.2f} | MACD: {_macd_summary(snap.tf_1h)}",
        f"• 15m RSI6: {snap.tf_15m.rsi6:.2f} | MACD: {_macd_summary(snap.tf_15m)}",
        "",
        "• ATR(15m / 1h): "
        f"{snap.tf_15m.atr:.4f} / {snap.tf_1h.atr:.4f} → {_atr_comment()}",
        f"• Funding: {snap.deriv.funding * 100:.4f}%",
        oi_line,
        "",
        "-------------------",
        "",
        "📍 Liquidity Check",
        f"Orderbook: {liquidity_summary} → {liquidity_note}",
    ]

    if status_mode:
        reminder = [
            "",
            "-------------------",
            "",
            "🔔 Reminder",
            signal.reason or _contextual_comment(execution_type),
        ]
        return "\n".join(header + core_metrics + reminder)

    action_lines = [
        "",
        "-------------------",
        "",
        "🎯 Action Summary",
        f"➡️ Direction: {signal.direction}",
        f"➡️ Execution: {execution_type}",
        "➡️ Levels:",
    ]

    if entries:
        action_lines.append(f"   - Entry: {' → '.join(entries)}")
    if tp_lines:
        action_lines.append(f"   - TP: {tp_lines}")
    if sl_line:
        action_lines.append(f"   - SL: {sl_line}")
    action_lines.append(f"➡️ Confidence: {conf_text}")

    reminder = [
        "",
        "-------------------",
        "",
        "🔔 Reminder",
        _contextual_comment(execution_type),
    ]

    return "\n".join(header + core_metrics + action_lines + reminder)


def main():
    settings = Settings()
    client = HyperliquidDataClient(settings)
    snapshot = client.build_market_snapshot()

    engine = SignalEngine(settings)
    signal = engine.generate_signal(snapshot)

    print_signal(signal)

    notifier = Notifier(
        telegram_token=settings.telegram_token,
        telegram_chat_id=settings.telegram_chat_id,
        ftqq_key=settings.ftqq_key,
        webhook_url=settings.webhook_url,
    )

    if notifier.has_channels():
        message = format_notification(signal, threshold=settings.signal_confidence_threshold)
        execution_mode = signal.direction != "none" and signal.confidence >= settings.signal_confidence_threshold
        results = notifier.send(
            message=message,
            title="Hyperliquid Trade Signal",
            include_ftqq=execution_mode,
        )
        print("Notification results:", results)
    else:
        print("No notification channels configured; skipping notify.")


if __name__ == "__main__":
    main()
