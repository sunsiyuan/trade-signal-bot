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
    print(
        f"Funding={d.funding:.6f}, "
        f"OI={d.open_interest:.2f}, "
        f"OI_24h_change={d.oi_change_24h:.2f}, "
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


def format_notification(signal):
    """Build a concise notification payload for chat/webhook channels."""

    snap = signal.snapshot
    lines = ["====== TRADE SIGNAL ======"]

    if snap:
        utc_ts = snap.ts.astimezone(timezone.utc)
        beijing_ts = utc_ts.astimezone(timezone(timedelta(hours=8)))
        lines.extend(
            [
                f"Time:   {utc_ts.isoformat()} (UTC)",
                f"        {beijing_ts.isoformat()} (Beijing)",
                f"Symbol: {signal.symbol}",
                "",
            ]
        )

        def tf_row(tf):
            return (
                f"{tf.timeframe:>4} | "
                f"close={tf.close:.4f} | "
                f"MA7={tf.ma7:.2f}, MA25={tf.ma25:.2f}, MA99={tf.ma99:.2f} | "
                f"RSI6={tf.rsi6:.2f}, RSI12={tf.rsi12:.2f}, RSI24={tf.rsi24:.2f} | "
                f"MACD={tf.macd:.4f}, SIG={tf.macd_signal:.4f}, HIST={tf.macd_hist:.4f} | "
                f"ATR={tf.atr:.4f} | Trend={tf.trend_label}"
            )

        lines.extend(
            [
                "=== Timeframe Indicators ===",
                tf_row(snap.tf_4h),
                tf_row(snap.tf_1h),
                tf_row(snap.tf_15m),
            ]
        )

    lines.extend(
        [
            "", "=== Derivative Indicators ===",
            f"Funding: {snap.deriv.funding * 100:.4f}%" if snap and snap.deriv else "Funding: N/A",
            f"Open Interest: {snap.deriv.open_interest:.2f}" if snap and snap.deriv else "Open Interest: N/A",
            f"OI 24h Change: {snap.deriv.oi_change_24h:.2f}" if snap and snap.deriv else "OI 24h Change: N/A",
            f"Liquidity: {snap.deriv.liquidity_comment}" if snap and snap.deriv else "Liquidity: N/A",
        ]
    )

    lines.extend(
        [
            "", "=== Trade Signal ===",
            f"Direction : {signal.direction}",
            f"Confidence: {signal.confidence:.2f}",
            f"Reason    : {signal.reason}",
        ]
    )

    if signal.direction != "none":
        lines.extend(
            [
                f"Entry range: {signal.entry_range or signal.entry}",
                f"TP1: {signal.tp1} | TP2: {signal.tp2} | SL: {signal.sl}",
                (
                    "Position: core="
                    f"{signal.core_position_pct * 100:.0f}%"
                    f" + add={signal.add_position_pct * 100:.0f}%"
                ),
            ]
        )
    else:
        lines.append("暂无交易信号，等待下一次机会。")

    lines.append("===============================")

    return "\n".join(lines)


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
        message = format_notification(signal)
        results = notifier.send(message=message, title="Hyperliquid Trade Signal")
        print("Notification results:", results)
    else:
        print("No notification channels configured; skipping notify.")


if __name__ == "__main__":
    main()
