# bot/main.py
import json
import os
from dataclasses import asdict, is_dataclass, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import ccxt

from .config import Settings


_DEFAULT_PRICE_QUANTIZATION = Settings().price_quantization
from .data_client import HyperliquidDataClient
from .logging_schema import build_signal_event, write_jsonl_event
from .notify import Notifier
from .signal_engine import SignalEngine
from .state_store import (
    ACTION_TTLS,
    compute_action_hash,
    compute_signal_id,
    load_global_state,
    load_state,
    mark_sent,
    save_global_state,
    save_state,
    should_send,
)


def _beijing_now() -> datetime:
    return datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=8)))


def _beijing_time_header() -> str:
    return f"⏱ 北京时间: {_beijing_now().strftime('%Y-%m-%d %H:%M')}"


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
        return "↔️", "趋势分歧"
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


def _setup_cn(setup_type: Optional[str]) -> str:
    mapping = {
        "trend_long": "趋势跟随做多",
        "trend_short": "趋势跟随做空",
        "mr_long": "均值回归做多",
        "mr_short": "均值回归做空",
        "lh_long": "流动性猎杀做多",
        "lh_short": "流动性猎杀做空",
    }
    if not setup_type:
        return "-"
    return mapping.get(setup_type, setup_type)


def _format_pct(value: float) -> str:
    if value is None:
        return "0%"
    return f"{round(value * 100):d}%"


def _get_price_decimals(symbol: Optional[str], settings: Optional[Settings]) -> int:
    mapping = getattr(settings, "price_quantization", None) or _DEFAULT_PRICE_QUANTIZATION
    base = symbol.split("/")[0] if symbol else None
    step = mapping.get(base) if mapping else None

    if step is None:
        return 4

    try:
        step_decimal = Decimal(str(step)).normalize()
        exponent = step_decimal.as_tuple().exponent
        decimals = -exponent if exponent < 0 else 0
        return decimals + 1
    except Exception:
        return 4


def _format_price(value: float, symbol: Optional[str] = None, settings: Optional[Settings] = None) -> str:
    if value is None:
        return "NA"
    decimals = _get_price_decimals(symbol, settings)
    return f"{value:.{decimals}f}"


def _display_symbol(symbol: Optional[str]) -> str:
    if not symbol:
        return ""
    return symbol.split(":")[0]


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
        deriv = getattr(snapshot, "deriv", None)
        if deriv and getattr(deriv, "mark_price", None) is not None:
            return deriv.mark_price

        tf_15m = getattr(snapshot, "tf_15m", None)
        if tf_15m is not None:
            prices = getattr(tf_15m, "prices", {}) or {}
            if isinstance(prices, dict) and prices.get("mark") is not None:
                return prices.get("mark")
            if getattr(tf_15m, "price_last", None) is not None:
                return getattr(tf_15m, "price_last")

        return getattr(snapshot, "price", None)
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


def _log_dedupe(info: Dict[str, Any]) -> None:
    enriched = {**info, "deduped": info.get("result") == "DEDUPED"}
    print(json.dumps(enriched, ensure_ascii=False))


def format_summary_compact(symbol, snapshot, action: str) -> str:
    display_symbol = _display_symbol(symbol)
    mark_price = _extract_mark_price(snapshot)
    fallback_price = getattr(snapshot.tf_15m, "close", None) if snapshot else None
    price = _format_price(
        mark_price if mark_price is not None else fallback_price, symbol=symbol
    )
    regime_icon, regime_cn = _regime_display(
        getattr(snapshot, "regime", ""),
        getattr(snapshot.tf_4h, "trend_label", "") if snapshot else "",
    )
    return f"{display_symbol} | 💰 {price} | {regime_icon}{regime_cn} | {_action_label(action)}"


def _extract_rsi6_value(snapshot) -> Optional[float]:
    try:
        tf_15m = getattr(snapshot, "tf_15m", None)
        if tf_15m is None:
            return None

        indicators = getattr(tf_15m, "indicators", None)
        if isinstance(indicators, dict):
            value = indicators.get("rsi_6")
            if value is not None:
                return value

        return getattr(tf_15m, "rsi6", None)
    except Exception:
        return None


def _format_valid_until(plan: Dict) -> str:
    valid_until = plan.get("valid_until_utc")
    dt = _parse_dt(valid_until)
    if dt:
        return dt.astimezone(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M")
    return valid_until or "N/A"


def _format_tp_values(signal, plan: Dict) -> str:
    tps = []
    tp_container = getattr(signal, "tp", None) if signal else None
    symbol = plan.get("symbol") or getattr(signal, "symbol", None)
    for key in ("tp1", "tp2", "tp3"):
        value = plan.get(key)
        if value is None and signal:
            if tp_container is not None:
                value = getattr(tp_container, key, None) if not isinstance(tp_container, dict) else tp_container.get(key)
            if value is None:
                value = getattr(signal, key, None)
        if value is not None:
            tps.append(_format_price(value, symbol=symbol))

    return "/".join(tps) if tps else "-"


def _format_sl_value(signal, plan: Dict) -> str:
    sl_candidates = [
        plan.get("sl"),
        getattr(signal, "sl", None) if signal else None,
        getattr(getattr(signal, "execution_intent", None), "invalidation_price", None)
        if signal
        else None,
        plan.get("invalidation_price"),
    ]

    symbol = plan.get("symbol") or getattr(signal, "symbol", None)

    for value in sl_candidates:
        if value is not None:
            return _format_price(value, symbol=symbol)
    return "-"


def format_action_plan_message(
    signal,
    snap,
    plan: Dict,
    signal_id: str,
    event: str = "CREATED",
    reason: str = "",
) -> str:
    plan = _plan_dict(plan) or {}
    symbol = plan.get("symbol") or getattr(signal, "symbol", "")
    display_symbol = _display_symbol(symbol)
    price = _format_price(_extract_mark_price(snap), symbol=symbol)
    rsi6 = _extract_rsi6_value(snap)
    rsi_text = f"{rsi6:.1f}" if rsi6 is not None else "NA"

    direction = (plan.get("direction") or getattr(signal, "direction", "")) or ""
    setup_type = getattr(signal, "setup_type", None) or plan.get("setup_type")
    execution_mode = _setup_cn(setup_type)
    entry_price = plan.get("entry_price")
    entry_text = (
        _format_price(entry_price, symbol=symbol) if entry_price is not None else "-"
    )
    sl_text = _format_sl_value(signal, plan)
    tp_text = _format_tp_values(signal, plan)
    valid_until = _format_valid_until(plan)
    reason_text = reason or plan.get("explain") or getattr(signal, "reason", "") or "-"

    event_display = {
        "CREATED": "设置限价单",
        "TRADE_NOW": "立刻交易",
        "TRADENOW": "立刻交易",
        "EXECUTE_NOW": "立刻交易",
    }.get(event, "设置限价单" if event.startswith("CREATED") else event)

    return "\n".join(
        [
            _beijing_time_header(),
            f"【{event_display}】",
            f"ID: {signal_id}",
            f"标的: {display_symbol} | 方向: {direction.upper()} | 模式: {execution_mode}",
            f"现价: {price} | 15m RSI6: {rsi_text}",
            f"入场: {entry_text} | SL: {sl_text} | TP: {tp_text}",
            f"有效期: {valid_until}",
            f"原因: {reason_text}",
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
    display_symbol = _display_symbol(symbol)
    mark_price = _extract_mark_price(snapshot)
    fallback_price = getattr(snapshot.tf_15m, "close", None) if snapshot else None
    price = _format_price(
        mark_price if mark_price is not None else fallback_price, symbol=symbol
    )
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
        f"{display_symbol} | 💰 {price} | {regime_icon}{regime_cn} | "
        f"{action_icon} {action_cn}{bias_block} | "
        f"Strat {strat} | Trade {trade_conf} / Edge {edge_conf_display} | "
        f"15m RSI6 {rsi_15m} | 4H MACD hist {macd_hist_4h} | Levels {levels} | "
        f"Risk {risk} | Why {why}"
    )


def format_summary_line(symbol, snapshot, signal) -> str:
    display_symbol = _display_symbol(symbol)
    mark_price = _extract_mark_price(snapshot)
    fallback_price = getattr(snapshot.tf_15m, "close", None) if snapshot else None
    price = _format_price(
        mark_price if mark_price is not None else fallback_price, symbol=symbol
    )
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
        f"{display_symbol} | 💰 {price} | {regime_icon}{regime_cn} | "
        f"{decision_icon} {decision_cn} | Trade {trade_conf} / Edge {edge_conf_display} | "
        f"15m RSI6 {rsi_15m} | 4H MACD hist {macd_hist_4h} | Setup {setup}"
    )


def format_signal_detail(signal):
    snap = signal.snapshot
    display_symbol = _display_symbol(signal.symbol)
    if not snap:
        return f"{display_symbol}: snapshot unavailable"

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
        f"📌 {display_symbol} — Trade Signal",
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
        f"• Price: {_format_price(snap.tf_15m.close, symbol=signal.symbol)}",
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
            f"Entry: {_format_price(conditional.get('entry_price'), symbol=signal.symbol)} | Valid until: {conditional.get('valid_until_utc') or 'N/A'}"
        )
        if conditional.get("explain"):
            lines.append(f"Explain: {conditional.get('explain')}")

    return "\n".join(lines)


def format_conditional_plan_line(signal) -> str:
    plan = _plan_dict(getattr(signal, "conditional_plan", None))
    if not plan:
        return ""

    display_symbol = _display_symbol(signal.symbol)
    entry_price = plan.get("entry_price")
    entry_text = (
        _format_price(entry_price, symbol=signal.symbol)
        if entry_price is not None
        else "N/A"
    )
    valid_until = plan.get("valid_until_utc") or "N/A"

    return (
        f"{display_symbol} | ⏳4H 执行 {plan.get('execution_mode', '')} {plan.get('direction', '').upper()} "
        f"@ {entry_text} | 有效期 {valid_until} | {plan.get('explain', '')}"
    )


def render_signal_dashboard(signals) -> str:
    if not signals:
        return "暂无交易信号。"

    lines = [_beijing_time_header(), "====== 多币种概览 ======"]
    for sig in signals:
        lines.append(
            format_summary_line(sig.symbol, sig.snapshot, sig)
            if sig.snapshot
            else f"{_display_symbol(sig.symbol):<11} | 数据缺失"
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
    state = load_global_state(state_path)
    symbol_states: Dict[str, Dict[str, Any]] = {}
    dirty_symbols: set[str] = set()

    def _get_symbol_state(sym: str) -> Dict[str, Any]:
        if sym not in symbol_states:
            symbol_states[sym] = load_state(sym)
        return symbol_states[sym]

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
        signal.signal_id = compute_signal_id(signal)
        emit_multi_tf_log(snapshot, signal, symbol_settings, exchange_id=exchange.id)
        signals.append(signal)
        snapshots[symbol] = snapshot

    summary_lines = []
    action_messages = []
    execute_now_messages = []
    header = _beijing_time_header()
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
                    format_action_plan_message(
                        None, snap, plan_for_msg, signal_id or "", event=event, reason=reason
                    )
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

        signal_id = getattr(sig, "signal_id", None) or compute_signal_id(sig)

        base_plan = {
            "signal_id": signal_id,
            "symbol": sig.symbol,
            "execution_mode": mode,
            "direction": plan.get("direction") or sig.direction,
            "setup_type": getattr(sig, "setup_type", None),
            "entry_price": entry_price,
            "invalidation_price": invalidation_price,
            "tp1": getattr(sig, "tp1", None),
            "tp2": getattr(sig, "tp2", None),
            "tp3": getattr(sig, "tp3", None),
            "sl": getattr(sig, "sl", None),
            "regime": getattr(snap, "regime", None) if snap else None,
            "valid_until_utc": plan.get("valid_until_utc"),
            "created_utc": now.isoformat(),
            "status": "ACTIVE",
        }

        valid_until_dt = _parse_dt(base_plan.get("valid_until_utc"))

        if mode == "WATCH":
            current_action = "WATCH"
            allowed, info = should_send(
                {},
                signal_id,
                sig.symbol,
                current_action,
                now,
                action_hash=None,
            )
            _log_dedupe(info)
        elif mode == "PLACE_LIMIT_4H":
            current_action = "LIMIT_4H"
            symbol_state = _get_symbol_state(sig.symbol)
            reason = plan.get("explain") or sig.reason or "创建4H限价计划"
            action_payload = {**base_plan, "reason": reason}
            action_hash = compute_action_hash(current_action, action_payload)
            allowed, info = should_send(
                symbol_state,
                signal_id,
                sig.symbol,
                current_action,
                now,
                action_hash=action_hash,
            )
            _log_dedupe(info)
            if allowed:
                action_messages.append(
                    format_action_plan_message(
                        sig,
                        snap,
                        base_plan,
                        signal_id or "",
                        event="CREATED",
                        reason=reason,
                    )
                )
                _mark_event_sent(state.setdefault("sent_events", {}), signal_id, "CREATED")
                mark_sent(
                    symbol_state,
                    signal_id,
                    sig.symbol,
                    current_action,
                    now,
                    valid_until=valid_until_dt,
                    action_hash=action_hash,
                )
                dirty_symbols.add(sig.symbol)
            state.setdefault("active_plans", {})[sig.symbol] = base_plan
        elif mode == "EXECUTE_NOW":
            current_action = "EXECUTE_NOW"
            reason = plan.get("explain") or sig.reason or "立即执行"
            symbol_state = _get_symbol_state(sig.symbol)
            action_payload = {**base_plan, "reason": reason}
            action_hash = compute_action_hash(current_action, action_payload)
            allowed, info = should_send(
                symbol_state,
                signal_id,
                sig.symbol,
                current_action,
                now,
                action_hash=action_hash,
            )
            _log_dedupe(info)
            if allowed:
                action_messages.append(
                    format_action_plan_message(
                        sig,
                        snap,
                        base_plan,
                        signal_id or "",
                        event="EXECUTE_NOW",
                        reason=reason,
                    )
                )
                execute_now_messages.append(
                    format_action_plan_message(
                        sig,
                        snap,
                        base_plan,
                        signal_id or "",
                        event="EXECUTE_NOW",
                        reason=reason,
                    )
                )
                _mark_event_sent(state.setdefault("sent_events", {}), signal_id, "EXECUTE_NOW")
                mark_sent(
                    symbol_state,
                    signal_id,
                    sig.symbol,
                    current_action,
                    now,
                    valid_until=None,
                    action_hash=action_hash,
                )
                dirty_symbols.add(sig.symbol)

        summary_lines.append(format_summary_compact(sig.symbol, snap, current_action))

    summary_message = "\n".join([header] + summary_lines)
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

    if execute_now_messages and notifier.ftqq_key:
        combined = "\n\n".join(execute_now_messages)
        results.update(
            notifier.send(
                message=combined,
                title="交易执行信号",
                include_ftqq=bool(notifier.ftqq_key),
            )
        )

    for symbol in dirty_symbols:
        save_state(symbol, symbol_states[symbol])

    save_global_state(state_path, state)

    if results:
        print("Notification results:", results)
    else:
        print("No notification channels configured; skipping notify.")


if __name__ == "__main__":
    main()
