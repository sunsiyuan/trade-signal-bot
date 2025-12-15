# bot/main.py
import json
import os
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

import ccxt

from .config import Settings
from .data_client import HyperliquidDataClient
from .logging_schema import build_signal_event, write_jsonl_event
from .notify import Notifier
from .signal_engine import SignalEngine


def _beijing_now() -> datetime:
    return datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=8)))


def _decision_icon(direction: str) -> str:
    return {"long": "📈", "short": "📉"}.get(direction, "🧊")


def _decision_cn(direction: str) -> str:
    return {"long": "多", "short": "空"}.get(direction, "观望")


def _regime_display(regime: str, trend_label: str) -> Tuple[str, str]:
    regime_display = regime or "unknown"
    if regime_display == "trending":
        if trend_label == "up":
            return "🟢", "上涨趋势"
        if trend_label == "down":
            return "🔻", "下跌趋势"
        return "🟣", "趋势态势（方向未定）"
    if regime_display == "high_vol_ranging":
        return "🌪️", "高波动震荡"
    if regime_display == "low_vol_ranging":
        return "🌤️", "低波动震荡"
    return "❔", "未知态势"


def _setup_code(setup_type: str) -> str:
    mapping = {
        "trend_long": "趋势跟随(TF)",
        "trend_short": "趋势跟随(TF)",
        "mean_reversion": "均值回归(MR)",
        "liquidity_hunt": "流动性狩猎(LH)",
        "none": "无",
    }
    return mapping.get(setup_type, setup_type or "none")


def _format_pct(value: float) -> str:
    if value is None:
        return "0%"
    return f"{round(value * 100):d}%"


def _format_price(value: float) -> str:
    if value is None:
        return "NA"
    return f"{value:.4f}"


def _format_oi(value: float) -> str:
    if value is None:
        return "NA"
    return f"{value:,.0f}"


def _format_macd_hist(value: float) -> str:
    if value is None:
        return "NA"
    return f"{value:.4f}"


def _format_levels(signal) -> str:
    if signal.direction == "none":
        return "-"

    parts: List[str] = []
    if signal.entry_range and len(signal.entry_range) > 1:
        parts.append(f"E {signal.entry_range[0]:.4f}-{signal.entry_range[-1]:.4f}")
    elif signal.entry_range:
        parts.append(f"E {signal.entry_range[0]:.4f}")
    elif signal.entry:
        parts.append(f"E {signal.entry:.4f}")

    tps = [tp for tp in [signal.tp1, signal.tp2, signal.tp3] if tp is not None]
    if tps:
        tp_text = "/".join(f"{tp:.0f}" if abs(tp) >= 100 else f"{tp:.4f}" for tp in tps)
        parts.append(f"TP {tp_text}")

    if signal.sl is not None:
        sl_text = f"{signal.sl:.0f}" if abs(signal.sl) >= 100 else f"{signal.sl:.4f}"
        parts.append(f"SL {sl_text}")

    return " | ".join(parts) if parts else "-"


def _extract_rsi_15m(snapshot) -> str:
    try:
        return f"{snapshot.tf_15m.rsi6:.1f}"
    except Exception:
        return "NA"


def _extract_mark_price(snapshot) -> Optional[float]:
    if snapshot is None:
        return None
    try:
        return getattr(snapshot.deriv, "mark_price", None)
    except Exception:
        return None


def _extract_gate(signal) -> str:
    thresholds = signal.thresholds_snapshot or {}
    debug_scores = signal.debug_scores or {}
    return thresholds.get("gate") or debug_scores.get("gate_tag") or "NA"


def _top_reasons(signal, action_level: str) -> str:
    reasons = signal.rejected_reasons or []
    thresholds = signal.thresholds_snapshot or {}
    if action_level == "EXECUTE" and signal.direction != "none":
        if reasons:
            return reasons[0]
        return "ok"

    if reasons:
        return ",".join(reasons[:2])

    if thresholds:
        for key, value in thresholds.items():
            if isinstance(value, (int, float)):
                return f"{key}={value}"
            if value:
                return f"{key}"

    return "insufficient"


def _bias_from_scores(signal) -> str:
    if getattr(signal, "conditional_plan", None):
        plans = signal.conditional_plan.get("plans") or []
        if plans:
            return plans[0].get("direction", "NONE").upper()
    if signal.direction and signal.direction != "none":
        return signal.direction.upper()
    scores = signal.debug_scores or {}
    long_score = scores.get("long")
    short_score = scores.get("short")
    if long_score is None and short_score is None:
        return "NONE"
    if short_score is None or (long_score is not None and long_score >= short_score):
        return "LONG"
    return "SHORT"


def is_actionable(signal, snapshot, settings: Settings):
    cfg = getattr(settings, "notification", {}) or {}
    execute_trade_conf = cfg.get("execute_trade_conf", 0.75)
    watch_trade_conf = cfg.get("watch_trade_conf", 0.55)
    watch_edge_conf = cfg.get("watch_edge_conf", 0.80)
    near_miss_delta = cfg.get("near_miss_delta", 0.05)

    trade_conf = signal.trade_confidence or 0.0
    edge_conf = signal.edge_confidence or 0.0
    has_conditional_plan = bool(getattr(signal, "conditional_plan", None))

    execute = signal.direction != "none" and trade_conf >= execute_trade_conf
    watch = (
        (edge_conf >= watch_edge_conf and trade_conf >= watch_trade_conf)
        or (signal.setup_type != "none" and edge_conf >= watch_edge_conf)
        or (trade_conf >= execute_trade_conf - near_miss_delta
            and edge_conf >= watch_edge_conf)
        or has_conditional_plan
    )

    if execute:
        return True, "EXECUTE", _bias_from_scores(signal)
    if watch:
        return True, "WATCH", _bias_from_scores(signal)
    return False, "NONE", "NONE"


def format_action_line(symbol, snapshot, signal, action_level: str, bias: str) -> str:
    mark_price = _extract_mark_price(snapshot)
    fallback_price = getattr(snapshot.tf_15m, "close", None) if snapshot else None
    price = _format_price(mark_price if mark_price is not None else fallback_price)
    regime_icon, regime_cn = _regime_display(
        getattr(snapshot, "regime", ""),
        getattr(snapshot.tf_4h, "trend_label", "") if snapshot else "",
    )
    action_icon = "✅" if action_level == "EXECUTE" else "👀"
    action_cn = "可执行（考虑下单）" if action_level == "EXECUTE" else "关注（准备介入）"

    bias_icon = ""
    bias_cn = ""
    if bias == "LONG":
        bias_icon = "📈"
        bias_cn = " 偏多"
    elif bias == "SHORT":
        bias_icon = "📉"
        bias_cn = " 偏空"

    strat = _setup_code(getattr(signal, "setup_type", "none"))
    trade_conf = _format_pct(signal.trade_confidence or 0.0)
    edge_conf = _format_pct(signal.edge_confidence or 0.0)
    rsi_15m = _extract_rsi_15m(snapshot) if snapshot else "NA"
    macd_hist_4h = _format_macd_hist(
        getattr(snapshot.tf_4h, "macd_hist", None) if snapshot else None
    )
    levels = _format_levels(signal)
    risk = (
        f"Core {signal.core_position_pct * 100:.1f}% + "
        f"Add {signal.add_position_pct * 100:.1f}%"
    )

    gate_tag = _extract_gate(signal)
    top_reason = _top_reasons(signal, action_level)
    why = f"Setup={signal.setup_type or 'none'}; Gate={gate_tag}; Top={top_reason}"

    bias_block = f" {bias_icon}{bias_cn}" if bias_icon else ""
    return (
        f"{symbol} | 💰 {price} | {regime_icon}{regime_cn} | "
        f"{action_icon} {action_cn}{bias_block} | "
        f"Strat {strat} | Trade {trade_conf} / Edge {edge_conf} | "
        f"15m RSI6 {rsi_15m} | 4H MACD hist {macd_hist_4h} | Levels {levels} | "
        f"Risk {risk} | Why {why}"
    )


def format_summary_line(symbol, snapshot, signal) -> str:
    mark_price = _extract_mark_price(snapshot)
    fallback_price = getattr(snapshot.tf_15m, "close", None) if snapshot else None
    price = _format_price(mark_price if mark_price is not None else fallback_price)
    regime_icon, regime_cn = _regime_display(
        getattr(snapshot, "regime", ""),
        getattr(snapshot.tf_4h, "trend_label", "") if snapshot else "",
    )
    decision_icon = _decision_icon(signal.direction)
    decision_cn = _decision_cn(signal.direction)
    trade_conf = _format_pct(signal.trade_confidence or 0.0)
    edge_conf = _format_pct(signal.edge_confidence or 0.0)
    rsi_15m = _extract_rsi_15m(snapshot) if snapshot else "NA"
    macd_hist_4h = _format_macd_hist(
        getattr(snapshot.tf_4h, "macd_hist", None) if snapshot else None
    )
    setup = _setup_code(getattr(signal, "setup_type", "none"))

    return (
        f"{symbol} | 💰 {price} | {regime_icon}{regime_cn} | "
        f"{decision_icon} {decision_cn} | Trade {trade_conf} / Edge {edge_conf} | "
        f"15m RSI6 {rsi_15m} | 4H MACD hist {macd_hist_4h} | Setup {setup}"
    )


def format_signal_detail(signal):
    snap = signal.snapshot
    if not snap:
        return f"{signal.symbol}: snapshot unavailable"

    trade_conf = signal.trade_confidence or 0.0
    edge_conf = signal.edge_confidence if hasattr(signal, "edge_confidence") else 0.0
    regime_icon, regime_cn = _regime_display(
        snap.regime,
        snap.tf_4h.trend_label,
    )
    beijing_ts = snap.ts.astimezone(timezone(timedelta(hours=8)))

    debug_scores = signal.debug_scores or {}
    long_score = debug_scores.get("long", "NA")
    short_score = debug_scores.get("short", "NA")
    gate_tag = _extract_gate(signal)
    rejection_text = ", ".join(signal.rejected_reasons or []) or "-"

    lines = [
        f"📌 {signal.symbol} — Trade Signal",
        f"⏱ {beijing_ts.strftime('%Y-%m-%d %H:%M')} (UTC+8)",
        f"{regime_icon}{regime_cn} | Regime reason: {snap.regime_reason or 'N/A'}",
        f"Direction: {signal.direction} | Setup: {signal.setup_type}",
        f"Confidence: trade {int(trade_conf * 100)}% / edge {int(edge_conf * 100)}%",
        f"Scores → long {long_score} / short {short_score}",
        f"Gate: {gate_tag}",
        f"Thresholds: {json.dumps(signal.thresholds_snapshot or {}, ensure_ascii=False)}",
        f"Rejections: {rejection_text}",
        f"Position sizing: core {signal.core_position_pct * 100:.1f}% + add {signal.add_position_pct * 100:.1f}%",
        f"Levels: {_format_levels(signal)}",
        "",
        "Snapshot highlights:",
        f"• Price: {_format_price(snap.tf_15m.close)}",
        f"• 4H RSI6: {snap.tf_4h.rsi6:.2f} | 1H RSI6: {snap.tf_1h.rsi6:.2f} | 15m RSI6: {snap.tf_15m.rsi6:.2f}",
        f"• OI: {snap.deriv.open_interest:,.2f} | Funding: {snap.deriv.funding * 100:.4f}%",
    ]

    conditional = getattr(signal, "conditional_plan", None) or {}
    plans = conditional.get("plans") or []
    if conditional:
        lines.append("")
        validity = conditional.get("validity", {})
        lines.append("⏳ 4H Conditional Plan")
        lines.append(f"Valid until: {validity.get('valid_until_utc', 'N/A')}")
        for idx, plan in enumerate(plans, start=1):
            entry_zone = plan.get("entry_zone", [])
            lines.append(
                f"Plan {idx}: {plan.get('plan_type', '')} {plan.get('direction', '')} | "
                f"Zone {entry_zone} | Logic: {plan.get('entry_logic', '')} | "
                f"SL {plan.get('risk', {}).get('sl', 'N/A')} | TP {plan.get('risk', {}).get('tp', [])} | "
                f"Conf if triggered {plan.get('confidence_if_triggered', 0)}"
            )

    return "\n".join(lines)


def format_conditional_plan_line(signal) -> str:
    plan = getattr(signal, "conditional_plan", None)
    if not plan:
        return ""

    plans = plan.get("plans") or []
    if not plans:
        return ""

    item = plans[0]
    entry_zone = item.get("entry_zone") or []
    if len(entry_zone) >= 2:
        entry_text = f"{entry_zone[0]:.4f}-{entry_zone[1]:.4f}"
    elif entry_zone:
        entry_text = f"{entry_zone[0]:.4f}"
    else:
        entry_text = "N/A"

    risk = item.get("risk", {}) or {}
    sl = risk.get("sl")
    sl_text = f"{sl:.2f}" if isinstance(sl, (int, float)) else (sl or "N/A")
    tps = risk.get("tp") or []
    if tps:
        tp_text = "/".join(
            f"{tp:.0f}" if abs(tp) >= 100 else f"{tp:.2f}" for tp in tps
        )
    else:
        tp_text = "N/A"

    confidence = item.get("confidence_if_triggered", 0.0)
    plan_type = item.get("plan_type", "")
    plan_label = "条件单"
    if "BOUNCE" in plan_type.upper():
        plan_label = "超跌反弹多"
    elif "REJECT" in plan_type.upper():
        plan_label = "冲高回落空"
    elif "PULLBACK" in plan_type.upper():
        plan_label = "趋势回调入场"

    valid_until = plan.get("validity", {}).get("valid_until_utc", "N/A")

    return (
        f"{signal.symbol} | ⏳4H 条件单（{plan_label} | {plan_type}） 区间 {entry_text} | "
        f"触发: {item.get('entry_logic', '')} | SL {sl_text} | TP {tp_text} | "
        f"触发信心 {_format_pct(confidence)} | 有效期 {valid_until}"
    )


def render_signal_dashboard(signals) -> str:
    if not signals:
        return "暂无交易信号。"

    lines = ["====== 多币种概览 ======"]
    for sig in signals:
        lines.append(
            format_summary_line(sig.symbol, sig.snapshot, sig)
            if sig.snapshot
            else f"{sig.symbol:<11} | 数据缺失"
        )

    lines.append("")
    lines.append("====== 详细解析 ======")
    for sig in signals:
        lines.append(format_signal_detail(sig))
        lines.append("-------------------")

    return "\n".join(lines)


def emit_multi_tf_log(snapshot, signal, settings: Settings, exchange_id: str = "") -> None:
    event = build_signal_event(snapshot, signal, settings, exchange_id=exchange_id)
    print(json.dumps(event, ensure_ascii=False))
    log_path = os.getenv("LOG_JSONL_PATH", "data/logs/signals.jsonl")
    write_jsonl_event(event, log_path)


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

    beijing_line = _beijing_now().strftime("%Y-%m-%d %H:%M") + " (UTC+8)"
    summary_lines = []
    action_lines = []
    execute_lines = []
    conditional_lines = []

    for sig in signals:
        summary_lines.append(format_summary_line(sig.symbol, sig.snapshot, sig))
        actionable, action_level, bias = is_actionable(sig, sig.snapshot, base_settings)
        if actionable:
            action_lines.append(
                format_action_line(sig.symbol, sig.snapshot, sig, action_level, bias)
            )
        if action_level == 'EXECUTE':
            execute_lines.append(
                format_action_line(sig.symbol, sig.snapshot, sig, action_level, bias)
            )

        conditional_line = format_conditional_plan_line(sig)
        if conditional_line:
            conditional_lines.append(conditional_line)

    summary_message = "\n".join([beijing_line] + summary_lines)

    print(summary_message)

    action_message = (
        "\n".join([beijing_line] + action_lines) if action_lines else None
    )
    execute_message = (
        "\n".join([beijing_line] + execute_lines) if execute_lines else None
    )
    conditional_message = (
        "\n\n".join([beijing_line] + conditional_lines) if conditional_lines else None
    )

    action_token = base_settings.telegram_action_token
    action_chat = base_settings.telegram_action_chat_id

    summary_token = base_settings.telegram_summary_token
    summary_chat = base_settings.telegram_summary_chat_id

    results = {}
    if action_message and action_token and action_chat:
        results["telegram_action"] = notifier.send_telegram(
            action_message, token=action_token, chat_id=action_chat
        )

    if summary_token and summary_chat:
        results["telegram_summary"] = notifier.send_telegram(
            summary_message, token=summary_token, chat_id=summary_chat
        )

    if conditional_message and summary_token and summary_chat:
        results["telegram_conditional"] = notifier.send_telegram(
            conditional_message, token=summary_token, chat_id=summary_chat
        )

    if execute_message and (notifier.ftqq_key or notifier.webhook_url):
        results.update(
            notifier.send(
                message=execute_message,
                title="交易执行信号",
                include_ftqq=bool(notifier.ftqq_key),
            )
        )

    if results:
        print("Notification results:", results)
    else:
        print("No notification channels configured; skipping notify.")


if __name__ == "__main__":
    main()
