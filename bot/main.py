# bot/main.py
import json
import os
from dataclasses import asdict, is_dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import ccxt

from .config import Settings
from .data_client import HyperliquidDataClient
from .logging_schema import build_signal_event, write_jsonl_event
from .notify import Notifier
from .signal_engine import SignalEngine
from .state_store import load_state, save_state


def _beijing_now() -> datetime:
    return datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=8)))


def _plan_dict(plan):
    if plan is None:
        return None
    if is_dataclass(plan):
        return asdict(plan)
    if isinstance(plan, dict):
        return plan
    return None


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
    if signal.entry:
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
    plan = _plan_dict(getattr(signal, "conditional_plan", None))
    if plan:
        direction = plan.get("direction")
        if direction:
            return direction.upper()
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


def _normalize_execution_mode(plan: Optional[Dict]) -> str:
    mode = (plan or {}).get("execution_mode") or "WATCH_ONLY"
    if mode == "WATCH_ONLY":
        return "WATCH"
    if mode == "PLACE_LIMIT_4H":
        return "PLACE_LIMIT_4H"
    if mode == "EXECUTE_NOW":
        return "EXECUTE_NOW"
    return "WATCH"


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None


def _event_sent(sent_events: Dict[str, List[str]], signal_id: Optional[str], event: str) -> bool:
    if not signal_id:
        return False
    return event in sent_events.get(signal_id, [])


def _mark_event_sent(sent_events: Dict[str, List[str]], signal_id: Optional[str], event: str) -> None:
    if not signal_id:
        return
    sent_events.setdefault(signal_id, [])
    if event not in sent_events[signal_id]:
        sent_events[signal_id].append(event)


def format_summary_compact(symbol, snapshot, action: str) -> str:
    mark_price = _extract_mark_price(snapshot)
    fallback_price = getattr(snapshot.tf_15m, "close", None) if snapshot else None
    price = _format_price(mark_price if mark_price is not None else fallback_price)
    regime_icon, regime_cn = _regime_display(
        getattr(snapshot, "regime", ""),
        getattr(snapshot.tf_4h, "trend_label", "") if snapshot else "",
    )
    return f"{symbol} | 💰 {price} | {regime_icon}{regime_cn} | {_action_label(action)}"


def _format_action_event_message(event: str, plan: Dict, reason: str, signal_id: str) -> str:
    entry = _format_price(plan.get("entry_price"))
    invalidation = _format_price(plan.get("invalidation_price"))
    return "\n".join(
        [
            f"【{event}】交易动作更新",
            f"ID: {signal_id}",
            f"标的: {plan.get('symbol')} | 方向: {plan.get('direction', '').upper()} | 模式: {plan.get('execution_mode')}",
            f"入场: {entry} | 失效: {invalidation}",
            f"市场态势: {plan.get('regime', 'unknown')}",
            f"原因: {reason}",
        ]
    )


def _action_label(action: str) -> str:
    mapping = {
        "WATCH": "🧊 观望",
        "LIMIT_4H": "⏳ 限价4H",
        "EXECUTE_NOW": "⚡️ 立即执行",
        "NONE": "⏸️ 暂无动作",
    }
    return mapping.get(action, "⏸️ 暂无动作")


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
    edge_type = getattr(signal, "edge_type", None)
    edge_conf = _format_pct(signal.edge_confidence or 0.0)
    edge_conf_display = edge_conf + (f"（{edge_type}）" if edge_type else "")
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
        f"Strat {strat} | Trade {trade_conf} / Edge {edge_conf_display} | "
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
    edge_type = getattr(signal, "edge_type", None)
    edge_conf = _format_pct(signal.edge_confidence or 0.0)
    edge_conf_display = edge_conf + (f"（{edge_type}）" if edge_type else "")
    rsi_15m = _extract_rsi_15m(snapshot) if snapshot else "NA"
    macd_hist_4h = _format_macd_hist(
        getattr(snapshot.tf_4h, "macd_hist", None) if snapshot else None
    )
    setup = _setup_code(getattr(signal, "setup_type", "none"))

    return (
        f"{symbol} | 💰 {price} | {regime_icon}{regime_cn} | "
        f"{decision_icon} {decision_cn} | Trade {trade_conf} / Edge {edge_conf_display} | "
        f"15m RSI6 {rsi_15m} | 4H MACD hist {macd_hist_4h} | Setup {setup}"
    )


def format_signal_detail(signal):
    snap = signal.snapshot
    if not snap:
        return f"{signal.symbol}: snapshot unavailable"

    trade_conf = signal.trade_confidence or 0.0
    edge_conf = signal.edge_confidence if hasattr(signal, "edge_confidence") else 0.0
    edge_type = getattr(signal, "edge_type", None)
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
        f"Confidence: trade {int(trade_conf * 100)}% / edge {int(edge_conf * 100)}%"
        f"{f'（{edge_type}）' if edge_type else ''}",
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

    conditional = _plan_dict(getattr(signal, "conditional_plan", None))
    if conditional:
        lines.append("")
        lines.append("⏳ 4H Conditional Plan")
        lines.append(
            f"Mode: {conditional.get('execution_mode', '')} | Direction: {conditional.get('direction', '')}"
        )
        lines.append(
            f"Entry: {_format_price(conditional.get('entry_price'))} | Valid until: {conditional.get('valid_until_utc') or 'N/A'}"
        )
        if conditional.get("explain"):
            lines.append(f"Explain: {conditional.get('explain')}")

    return "\n".join(lines)


def format_conditional_plan_line(signal) -> str:
    plan = _plan_dict(getattr(signal, "conditional_plan", None))
    if not plan:
        return ""

    entry_price = plan.get("entry_price")
    entry_text = _format_price(entry_price) if entry_price is not None else "N/A"
    valid_until = plan.get("valid_until_utc") or "N/A"

    return (
        f"{signal.symbol} | ⏳4H 执行 {plan.get('execution_mode', '')} {plan.get('direction', '').upper()} "
        f"@ {entry_text} | 有效期 {valid_until} | {plan.get('explain', '')}"
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

    state_path = os.path.join(".state", "state.json")
    state = load_state(state_path)

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
    snapshots = {}
    for symbol in tracked:
        symbol_settings = replace(base_settings, symbol=symbol)
        client = HyperliquidDataClient(
            symbol_settings, exchange=exchange, funding_rates=funding_rates
        )
        snapshot = client.build_market_snapshot()
        signal = engine.generate_signal(snapshot)
        emit_multi_tf_log(snapshot, signal, symbol_settings, exchange_id=exchange.id)
        signals.append(signal)
        snapshots[symbol] = snapshot

    summary_lines = []
    action_messages = []
    execute_now_messages = []
    now = datetime.now(timezone.utc)

    # Step 1: reconcile existing active plans
    for symbol, plan in list(state.get("active_plans", {}).items()):
        signal_id = plan.get("signal_id")
        snap = snapshots.get(symbol)
        mark_price = _extract_mark_price(snap)
        regime = getattr(snap, "regime", None) if snap else None

        event = None
        reason = ""
        valid_until = _parse_dt(plan.get("valid_until_utc"))
        if valid_until and now > valid_until:
            event = "EXPIRED"
            reason = "超过有效期，撤销计划单"
        elif mark_price is not None and plan.get("invalidation_price") is not None:
            if plan.get("direction") == "long" and mark_price <= plan.get("invalidation_price"):
                event = "INVALIDATED"
                reason = "价格跌破失效位"
            elif plan.get("direction") == "short" and mark_price >= plan.get("invalidation_price"):
                event = "INVALIDATED"
                reason = "价格突破失效位"

        if event is None and regime and plan.get("regime") and plan.get("regime") != regime:
            event = "REGIME_CHANGED"
            reason = f"Regime {plan.get('regime')} → {regime}"

        if event:
            if not _event_sent(state.get("sent_events", {}), signal_id, event):
                plan_for_msg = {**plan, "symbol": symbol}
                action_messages.append(
                    _format_action_event_message(event, plan_for_msg, reason, signal_id or "")
                )
                _mark_event_sent(state.setdefault("sent_events", {}), signal_id, event)
            state.get("active_plans", {}).pop(symbol, None)

    # Step 2: handle new signals
    for sig in signals:
        snap = sig.snapshot
        plan = _plan_dict(getattr(sig, "conditional_plan", None)) or {}
        mode = _normalize_execution_mode(plan)
        current_action = "NONE"

        entry_price = plan.get("entry_price")
        invalidation_price = None
        if getattr(sig, "execution_intent", None):
            invalidation_price = sig.execution_intent.invalidation_price
        elif hasattr(sig, "sl"):
            invalidation_price = getattr(sig, "sl", None)

        base_plan = {
            "signal_id": getattr(sig, "signal_id", None),
            "symbol": sig.symbol,
            "execution_mode": mode,
            "direction": plan.get("direction") or sig.direction,
            "entry_price": entry_price,
            "invalidation_price": invalidation_price,
            "regime": getattr(snap, "regime", None) if snap else None,
            "valid_until_utc": plan.get("valid_until_utc"),
            "created_utc": now.isoformat(),
            "status": "ACTIVE",
        }

        if mode == "WATCH":
            current_action = "WATCH"
        elif mode == "PLACE_LIMIT_4H":
            current_action = "LIMIT_4H"
            existing = state.get("active_plans", {}).get(sig.symbol)
            if not existing or existing.get("signal_id") != base_plan.get("signal_id"):
                if not _event_sent(state.get("sent_events", {}), base_plan.get("signal_id"), "CREATED"):
                    reason = plan.get("explain") or sig.reason or "创建4H限价计划"
                    action_messages.append(
                        _format_action_event_message(
                            "CREATED",
                            base_plan,
                            reason,
                            base_plan.get("signal_id") or "",
                        )
                    )
                    _mark_event_sent(state.setdefault("sent_events", {}), base_plan.get("signal_id"), "CREATED")
                state.setdefault("active_plans", {})[sig.symbol] = base_plan
        elif mode == "EXECUTE_NOW":
            current_action = "EXECUTE_NOW"
            reason = plan.get("explain") or sig.reason or "立即执行"
            if not _event_sent(state.get("sent_events", {}), base_plan.get("signal_id"), "EXECUTE_NOW"):
                action_messages.append(
                    _format_action_event_message(
                        "EXECUTE_NOW",
                        base_plan,
                        reason,
                        base_plan.get("signal_id") or "",
                    )
                )
                execute_now_messages.append(
                    _format_action_event_message(
                        "EXECUTE_NOW",
                        base_plan,
                        reason,
                        base_plan.get("signal_id") or "",
                    )
                )
                _mark_event_sent(state.setdefault("sent_events", {}), base_plan.get("signal_id"), "EXECUTE_NOW")

        summary_lines.append(format_summary_compact(sig.symbol, snap, current_action))

    summary_message = "\n".join(summary_lines)
    print(summary_message)

    action_token = base_settings.telegram_action_token
    action_chat = base_settings.telegram_action_chat_id
    summary_token = base_settings.telegram_summary_token
    summary_chat = base_settings.telegram_summary_chat_id

    results = {}
    if summary_token and summary_chat and summary_message:
        results["telegram_summary"] = notifier.send_telegram(
            summary_message, token=summary_token, chat_id=summary_chat
        )

    if action_messages and action_token and action_chat:
        results["telegram_action"] = notifier.send_telegram(
            "\n\n".join(action_messages), token=action_token, chat_id=action_chat
        )

    if execute_now_messages and (notifier.ftqq_key or notifier.webhook_url):
        combined = "\n\n".join(execute_now_messages)
        results.update(
            notifier.send(
                message=combined,
                title="交易执行信号",
                include_ftqq=bool(notifier.ftqq_key),
            )
        )

    save_state(state_path, state)

    if results:
        print("Notification results:", results)
    else:
        print("No notification channels configured; skipping notify.")


if __name__ == "__main__":
    main()
