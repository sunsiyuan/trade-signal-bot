# bot/main.py
import json
import subprocess
from dataclasses import replace
from datetime import timezone, timedelta, datetime

import ccxt

from .config import Settings
from .data_client import HyperliquidDataClient
from .notify import Notifier
from .signal_engine import SignalEngine


def _direction_text(direction: str) -> str:
    if direction == "long":
        return "偏多（找多头机会）"
    if direction == "short":
        return "偏空（找空头机会）"
    return "观望"


def _direction_sentence(direction: str) -> str:
    if direction == "long":
        return "偏多，允许择机做多"
    if direction == "short":
        return "偏空，允许择机做空"
    return "多空优势都不明显，观望优先"


def _fmt_macd(value: float) -> str:
    if value > 0:
        return f"📈 Bullish ({value:+.4f})"
    if value < 0:
        return f"📉 Bearish ({value:+.4f})"
    return f"⚖️ Flat ({value:+.4f})"


def _ob_bias(asks: float, bids: float) -> str:
    if asks > bids * 1.2:
        return "卖盘略强（Asks 主导）"
    if bids > asks * 1.2:
        return "买盘略强（Bids 主导）"
    return "多空挂单相对均衡"


def _trend_icon(trend: str) -> tuple[str, str]:
    mapping = {"up": ("🟢", "up"), "down": ("🔴", "down"), "range": ("⚖️", "sideway")}
    return mapping.get(trend, ("⚖️", "sideway"))


def _decision_icon(direction: str) -> str:
    return {"long": "📈", "short": "📉"}.get(direction, "🧊")


def format_signal_brief(signal):
    snap = signal.snapshot
    if not snap:
        return f"{signal.symbol:<11} | 数据缺失"

    trend_emoji, trend_text = _trend_icon(snap.tf_4h.trend_label)
    price = snap.tf_15m.close
    direction_text = _direction_text(signal.direction)
    trade_conf = signal.trade_confidence or signal.confidence
    confidence_pct = int(trade_conf * 100)
    edge_pct = int((signal.edge_confidence or 0.0) * 100)
    rsi_4h = snap.tf_4h.rsi6
    oi = snap.deriv.open_interest

    return (
        f"{signal.symbol:<11} | "
        f"💰 {price:>8.4f} | "
        f"{trend_emoji} {trend_text:7} | "
        f"{_decision_icon(signal.direction)} {direction_text:<10} | "
        f"TradeConf {confidence_pct:>2d}% | "
        f"EdgeConf {edge_pct:>2d}% | "
        f"4H RSI {rsi_4h:>5.1f} | "
        f"OI {oi:>10,.0f}"
    )


def format_signal_detail(signal):
    snap = signal.snapshot
    if not snap:
        return f"{signal.symbol}: snapshot unavailable"

    trade_conf = signal.trade_confidence or signal.confidence
    edge_conf = signal.edge_confidence if hasattr(signal, "edge_confidence") else 0.0
    trend_emoji, _ = _trend_icon(snap.tf_4h.trend_label)
    decision_map = {
        "long": ("✅ 准备做多", "多头 setup 出现，可以找入场点"),
        "short": ("✅ 准备做空", "空头 setup 出现，可以找入场点"),
        "none": ("🧊 继续观望", "多空优势不足，不交易更有优势"),
    }
    decision_title, decision_explain = decision_map.get(signal.direction or "none")

    raw_reason = signal.reason or ""
    if "|" in raw_reason:
        human_reason, technical_note = raw_reason.split("|", 1)
    else:
        human_reason, technical_note = raw_reason, ""
    human_reason = human_reason.strip() or "没有出现符合模板的入场信号。"
    technical_note = technical_note.strip()

    asks = sum(order.get("size", 0) for order in snap.deriv.orderbook_asks)
    bids = sum(order.get("size", 0) for order in snap.deriv.orderbook_bids)
    ob_bias = _ob_bias(asks, bids)

    def _atr_comment() -> str:
        ratio = snap.tf_15m.atr / max(snap.tf_1h.atr, 1e-6)
        if ratio > 1.3:
            return "High vol"
        if ratio < 0.8:
            return "Calm tape"
        return "Normal swings"

    utc_ts = snap.ts.astimezone(timezone.utc)
    beijing_ts = utc_ts.astimezone(timezone(timedelta(hours=8)))

    lines = [
        f"📌 {signal.symbol} — Trade Signal",
        f"⏱ {beijing_ts.strftime('%Y-%m-%d %H:%M')} (UTC+8)",
        f"💰 Price: {snap.tf_15m.close:.4f}",
        f"{trend_emoji} | {decision_title} | 信心 {int(trade_conf * 100)}%",
        "",
        "🏁 Summary",
        f"• 核心结论：{decision_title}（{_direction_sentence(signal.direction)})",
        f"• 执行建议：{decision_explain}",
        f"• 主要原因：{human_reason}",
        f"• Edge confidence: {int(edge_conf * 100)}%",
        "",
        "📍 Multi-TF Snapshot",
        f"• 4H → RSI6: {snap.tf_4h.rsi6:.2f} | MACD: {_fmt_macd(snap.tf_4h.macd_hist)}",
        f"• 1H → RSI6: {snap.tf_1h.rsi6:.2f} | MACD: {_fmt_macd(snap.tf_1h.macd_hist)}",
        f"• 15m → RSI6: {snap.tf_15m.rsi6:.2f} | MACD: {_fmt_macd(snap.tf_15m.macd_hist)}",
        "",
        "📍 Vol & Positioning",
        f"• ATR (15m / 1h): {snap.tf_15m.atr:.4f} / {snap.tf_1h.atr:.4f} → {_atr_comment()}",
        f"• Funding: {snap.deriv.funding:.4%}",
        f"• OI: {snap.deriv.open_interest:,.2f}",
        "",
        "📍 Liquidity",
        f"• Orderbook: Asks {asks:,.0f} vs Bids {bids:,.0f} → {ob_bias}",
    ]

    notes = technical_note or None
    if notes:
        lines.extend([
            "",
            "🔔 Note",
            f"• {notes}",
        ])

    return "\n".join(lines)


def format_notification(signal, threshold: float = 0.8):
    """Render a Markdown Telegram/webhook notification."""

    snap = signal.snapshot

    trade_conf = signal.trade_confidence or signal.confidence
    edge_conf = signal.edge_confidence if hasattr(signal, "edge_confidence") else 0.0

    if not snap:
        return "Market snapshot unavailable; awaiting next refresh."

    def _fmt_with_commas(value: float, decimals: int) -> str:
        if value is None:
            return "N/A"
        fmt = f"{value:,.{decimals}f}" if abs(value) >= 1000 else f"{value:.{decimals}f}"
        return fmt

    def _fmt_price(value: float) -> str:
        if value is None:
            return "N/A"
        decimals = 4 if value and abs(value) < 100 else 2
        return _fmt_with_commas(value, decimals)

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
        summary = f"Asks {_fmt_with_commas(asks, 0)} vs Bids {_fmt_with_commas(bids, 0)}"
        return summary, skew

    def _confidence_label():
        pct = round(trade_conf * 100)
        suffix = " (High Probability)" if pct > 80 else ""
        return f"{pct}%{suffix}"

    def _reason_text(execution_hint: str) -> str:
        if signal.reason:
            return signal.reason
        return _contextual_comment(execution_hint)

    def _execution_type():
        if signal.entry_range and len(signal.entry_range) > 1:
            return "Ladder"
        if signal.entry_range:
            return "Limit"
        if not signal.entry:
            return "N/A"
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
            if exec_type == "Limit":
                return "Place long limit; wait for trigger and keep alerts on."
            return "No trade trigger yet — stay patient and monitor key levels."
        if direction == "short":
            if exec_type == "Market":
                return "Hit market short; confirm size and trailing plan."
            if exec_type == "Ladder":
                return "Stagger short limits; tighten risk once filled."
            if exec_type == "Limit":
                return "Set short limit at trigger and monitor."
            return "No trade trigger yet — stay patient and monitor key levels."
        return "No trade trigger yet — stay patient and monitor key levels."

    trend_emoji, trend_phrase = _trend_phrase()
    decision_emoji, decision_text = _decision_text()
    liquidity_summary, liquidity_note = _liquidity_summary()
    entries, tp_lines, sl_line = _format_levels()
    execution_type = _execution_type()
    conf_text = _confidence_label()
    status_mode = signal.direction == "none" or trade_conf < threshold

    utc_ts = snap.ts.astimezone(timezone.utc)
    beijing_ts = utc_ts.astimezone(timezone(timedelta(hours=8)))
    local_time_string = beijing_ts.strftime("%Y-%m-%d %H:%M")

    header = [
        f"📌 {signal.symbol} — Trade Signal",
        f"⏱ {local_time_string} (UTC+8)",
        f"{trend_emoji} Trend: {trend_phrase} | {decision_emoji} {decision_text}",
        f"💰 Price: {_fmt_price(snap.tf_15m.close)}",
    ]

    summary_lines = [
        "",
        "🏁 Quick Take",
        f"• Bias: {decision_emoji} {decision_text} | Direction: {signal.direction}",
        f"• Confidence: {conf_text}",
        f"• Edge: {int(edge_conf * 100)}%",
        f"• Plan: {execution_type if not status_mode else 'Monitor only'}",
        f"• Why: {_reason_text(execution_type)}",
    ]

    if not status_mode:
        summary_lines.extend(
            [
                f"• Entry: {' → '.join(entries)}" if entries else "• Entry: -",
                f"• TP: {tp_lines}" if tp_lines else "• TP: -",
                f"• SL: {sl_line}" if sl_line else "• SL: -",
            ]
        )

    oi_change = snap.deriv.oi_change_24h
    formatted_oi = _fmt_with_commas(snap.deriv.open_interest, 2)
    oi_line = (
        f"• OI: {formatted_oi}"
        if oi_change is None
        else f"• OI: {formatted_oi} → {_oi_state()} ({oi_change:+.2f}%)"
    )

    core_metrics = [
        "",
        "-------------------",
        "",
        "📍 Core Metrics",
        f"• 4H RSI6: {snap.tf_4h.rsi6:.2f} | MACD: {_macd_summary(snap.tf_4h)}",
        f"• 1H RSI6: {snap.tf_1h.rsi6:.2f} | MACD: {_macd_summary(snap.tf_1h)}",
        f"• 15m RSI6: {snap.tf_15m.rsi6:.2f} | MACD: {_macd_summary(snap.tf_15m)}",
        "",
        "• ATR(15m / 1h): ",
        f"{snap.tf_15m.atr:.4f} / {snap.tf_1h.atr:.4f} → {_atr_comment()}",
        f"• Funding: {snap.deriv.funding * 100:.4f}%",
        oi_line,
        "",
        "📍 Liquidity",
        f"• Orderbook: {liquidity_summary} → {liquidity_note}",
    ]

    if status_mode:
        reminder = [
            "",
            "🔔 Reminder",
            signal.reason or _contextual_comment(execution_type),
        ]
        return "\n".join(header + summary_lines + core_metrics + reminder)

    action_lines = [
        "",
        "🎯 Action Plan",
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
        "🔔 Reminder",
        _contextual_comment(execution_type),
    ]

    return "\n".join(header + summary_lines + core_metrics + action_lines + reminder)


def print_signal_dashboard(signals):
    print(render_signal_dashboard(signals))


def render_signal_dashboard(signals) -> str:
    if not signals:
        return "暂无交易信号。"

    lines = ["====== 多币种概览 ======"]
    for sig in signals:
        lines.append(format_signal_brief(sig))

    lines.append("")
    lines.append("====== 详细解析 ======")
    for sig in signals:
        lines.append(format_signal_detail(sig))
        lines.append("-------------------")

    return "\n".join(lines)


def _tf_to_timedelta(tf: str) -> timedelta:
    tf = tf.lower()
    if tf.endswith("m"):
        return timedelta(minutes=int(tf[:-1]))
    if tf.endswith("h"):
        return timedelta(hours=int(tf[:-1]))
    raise ValueError(f"Unsupported timeframe: {tf}")


def emit_multi_tf_log(snapshot, signal, settings: Settings, exchange_id: str = "") -> None:
    run_ts = datetime.now(timezone.utc)
    tf_map = {
        snapshot.tf_15m.timeframe: snapshot.tf_15m,
        snapshot.tf_1h.timeframe: snapshot.tf_1h,
        snapshot.tf_4h.timeframe: snapshot.tf_4h,
    }
    tf_list = [settings.tf_15m, settings.tf_1h, settings.tf_4h]
    anchor_tf = settings.tf_1h
    anchor = tf_map.get(anchor_tf)
    anchor_close = anchor.last_candle_close_utc if anchor else None

    tf_close_delta_sec = {}
    alignment_ok = True
    alignment_reason = ""
    for tf_name, tf in tf_map.items():
        if anchor_close and tf.last_candle_close_utc:
            delta = (tf.last_candle_close_utc - anchor_close).total_seconds()
            tf_close_delta_sec[tf_name] = delta
            if abs(delta) > 1e-6:
                alignment_ok = False
                alignment_reason = f"{tf_name} close misaligned vs {anchor_tf}"
        else:
            tf_close_delta_sec[tf_name] = None
            alignment_ok = False
            alignment_reason = alignment_reason or "missing_close_time"

        if not tf.is_last_candle_closed:
            alignment_ok = False
            alignment_reason = alignment_reason or f"{tf_name} last candle open"

    latest_close = max(
        (tf.last_candle_close_utc for tf in tf_map.values() if tf.last_candle_close_utc),
        default=run_ts,
    )
    data_latency_sec = (run_ts - latest_close).total_seconds()

    def _tf_block(tf):
        return {
            "tf": tf.timeframe,
            "last_candle_open_utc": tf.last_candle_open_utc.isoformat() if tf.last_candle_open_utc else None,
            "last_candle_close_utc": tf.last_candle_close_utc.isoformat() if tf.last_candle_close_utc else None,
            "is_last_candle_closed": tf.is_last_candle_closed,
            "bars_used": tf.bars_used,
            "lookback_window": tf.lookback_window,
            "missing_bars_count": tf.missing_bars_count,
            "gap_list": tf.gap_list,
            "prices": {
                "price_last": tf.price_last,
                "price_mid": tf.price_mid,
                "typical_price": tf.typical_price,
                "mark": None,
                "index": None,
                "return_last": tf.return_last,
            },
            "volatility": {
                "atr": tf.atr,
                "atr_rel": tf.atr_rel,
                "tr_last": tf.tr_last,
            },
            "indicators": {
                "rsi_6": tf.rsi6,
                "rsi_12": tf.rsi12,
                "rsi_24": tf.rsi24,
                "macd_value": tf.macd,
                "macd_signal": tf.macd_signal,
                "macd_hist": tf.macd_hist,
                "ma7": tf.ma7,
                "ma25": tf.ma25,
                "ma99": tf.ma99,
                "ma_angle": getattr(tf, "ma_angle", None),
            },
        }

    alignment_block = {
        "anchor_tf": anchor_tf,
        "anchor_close_utc": anchor_close.isoformat() if anchor_close else None,
        "tf_close_delta_sec": tf_close_delta_sec,
        "alignment_ok": alignment_ok,
        "alignment_reason": alignment_reason or "ok",
    }

    def _action() -> str:
        if signal.direction == "long":
            return "LONG"
        if signal.direction == "short":
            return "SHORT"
        return "NO_TRADE"

    anchor_delta = _tf_to_timedelta(anchor_tf)
    valid_until_utc = (anchor_close + anchor_delta).isoformat() if anchor_close else None

    debug_scores = signal.debug_scores or {}
    long_score = debug_scores.get("long")
    short_score = debug_scores.get("short")
    best_score = None
    if long_score is not None and short_score is not None:
        best_score = max(long_score, short_score)

    try:
        source_commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True)
            .strip()
        )
    except Exception:
        source_commit = ""

    payload = {
        "meta": {
            "symbol": snapshot.symbol,
            "exchange": exchange_id,
            "run_ts_utc": run_ts.isoformat(),
            "source_commit": source_commit,
            "data_latency_sec": data_latency_sec,
            "tf_list": tf_list,
        },
        "timeframes": [_tf_block(tf_map[tf]) for tf in tf_list if tf in tf_map],
        "alignment": alignment_block,
        "gates": {
            "regime": snapshot.regime,
            "long_score": long_score,
            "short_score": short_score,
            "best_score": best_score,
            "edge_score": signal.edge_confidence,
            "edge_conf": signal.edge_confidence,
            "trade_conf": signal.trade_confidence or signal.confidence,
            "rejected_reasons": signal.rejected_reasons or [],
            "thresholds_snapshot": signal.thresholds_snapshot or {},
        },
        "decision": {
            "action": _action(),
            "valid_until_utc": valid_until_utc,
        },
    }

    print(json.dumps(payload, ensure_ascii=False))


def main():
    base_settings = Settings()
    tracked = base_settings.tracked_symbols or [base_settings.symbol]

    exchange = ccxt.hyperliquid({"enableRateLimit": True})
    exchange.load_markets()

    funding_rates = None
    try:
        funding_rates = exchange.fetch_funding_rates()
    except Exception:
        funding_rates = None

    engine = SignalEngine(base_settings)
    notifier = Notifier(
        telegram_token=base_settings.telegram_token,
        telegram_chat_id=base_settings.telegram_chat_id,
        ftqq_key=base_settings.ftqq_key,
        webhook_url=base_settings.webhook_url,
    )

    signals = []
    for symbol in tracked:
        symbol_settings = replace(base_settings, symbol=symbol)
        client = HyperliquidDataClient(
            symbol_settings, exchange=exchange, funding_rates=funding_rates
        )
        snapshot = client.build_market_snapshot()
        signal = engine.generate_signal(snapshot)
        emit_multi_tf_log(snapshot, signal, symbol_settings, exchange_id=exchange.id)
        signals.append(signal)

    print_signal_dashboard(signals)

    if notifier.has_channels() and signals:
        dashboard_text = render_signal_dashboard(signals)
        execution_mode = any(
            sig.direction != "none"
            and (sig.trade_confidence or sig.confidence)
            >= base_settings.signal_confidence_threshold
            for sig in signals
        )
        results = notifier.send(
            message=dashboard_text,
            title="Hyperliquid Trade Signal",
            include_ftqq=execution_mode,
        )
        print("Notification results:", results)
    else:
        print("No notification channels configured; skipping notify.")


if __name__ == "__main__":
    main()
