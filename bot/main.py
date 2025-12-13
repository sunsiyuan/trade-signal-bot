# bot/main.py
import json
import os
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import List, Tuple

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


def _regime_display(regime: str, market_mode: str, trend_label: str) -> Tuple[str, str]:
    mapping = {
        "trending": "趋势",
        "high_vol_ranging": "高波动震荡",
        "low_vol_ranging": "低波动震荡",
    }
    regime_cn = mapping.get(regime, regime or "unknown")
    if regime_cn == "趋势":
        if trend_label == "up":
            return "🟢", "上涨趋势"
        if trend_label == "down":
            return "🔻", "下跌趋势"
        return "🟣", "趋势态势（方向未定）"
    return "⚖️", regime_cn


def _setup_code(setup_type: str) -> str:
    mapping = {
        "trend_long": "趋势跟随TF",
        "trend_short": "趋势跟随TF",
        "mean_reversion": "均值回归MR",
        "liquidity_hunt": "流动性狩猎LH",
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


def _extract_rsi_4h(snapshot) -> str:
    try:
        return f"{snapshot.tf_4h.rsi6:.1f}"
    except Exception:
        return "NA"


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

    execute = signal.direction != "none" and trade_conf >= execute_trade_conf
    watch = (
        (edge_conf >= watch_edge_conf and trade_conf >= watch_trade_conf)
        or (signal.setup_type != "none" and edge_conf >= watch_edge_conf)
        or (trade_conf >= execute_trade_conf - near_miss_delta
            and edge_conf >= watch_edge_conf)
    )

    if execute:
        return True, "EXECUTE", _bias_from_scores(signal)
    if watch:
        return True, "WATCH", _bias_from_scores(signal)
    return False, "NONE", "NONE"


def format_action_line(symbol, snapshot, signal, action_level: str, bias: str) -> str:
    price = _format_price(getattr(snapshot.tf_15m, "close", None) if snapshot else None)
    regime_icon, regime_cn = _regime_display(
        getattr(snapshot, "regime", ""),
        getattr(snapshot, "market_mode", ""),
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
    rsi4h = _extract_rsi_4h(snapshot) if snapshot else "NA"
    oi = _format_oi(getattr(snapshot.deriv, "open_interest", None) if snapshot else None)
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
        f"4H RSI {rsi4h} | OI {oi} | Levels {levels} | "
        f"Risk {risk} | Why {why}"
    )


def format_summary_line(symbol, snapshot, signal) -> str:
    price = _format_price(getattr(snapshot.tf_15m, "close", None) if snapshot else None)
    regime_icon, regime_cn = _regime_display(
        getattr(snapshot, "regime", ""),
        getattr(snapshot, "market_mode", ""),
        getattr(snapshot.tf_4h, "trend_label", "") if snapshot else "",
    )
    decision_icon = _decision_icon(signal.direction)
    decision_cn = _decision_cn(signal.direction)
    trade_conf = _format_pct(signal.trade_confidence or 0.0)
    edge_conf = _format_pct(signal.edge_confidence or 0.0)
    rsi4h = _extract_rsi_4h(snapshot) if snapshot else "NA"
    oi = _format_oi(getattr(snapshot.deriv, "open_interest", None) if snapshot else None)
    setup = _setup_code(getattr(signal, "setup_type", "none"))

    return (
        f"{symbol} | 💰 {price} | {regime_icon}{regime_cn} | "
        f"{decision_icon} {decision_cn} | Trade {trade_conf} / Edge {edge_conf} | "
        f"4H RSI {rsi4h} | OI {oi} | Setup {setup}"
    )


def format_signal_detail(signal):
    snap = signal.snapshot
    if not snap:
        return f"{signal.symbol}: snapshot unavailable"

    trade_conf = signal.trade_confidence or signal.confidence
    edge_conf = signal.edge_confidence if hasattr(signal, "edge_confidence") else 0.0
    regime_icon, regime_cn = _regime_display(
        snap.regime,
        snap.market_mode,
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

    return "\n".join(lines)


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
        telegram_token=None,
        telegram_chat_id=None,
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

    summary_message = "\n".join([beijing_line] + summary_lines)

    print(summary_message)

    action_message = (
        "\n".join([beijing_line] + action_lines) if action_lines else None
    )
    execute_message = (
        "\n".join([beijing_line] + execute_lines) if execute_lines else None
    )

    action_token = base_settings.telegram_action_token
    action_chat = base_settings.telegram_action_chat_id
    
    summary_token = base_settings.telegram_summary_token
    summary_chat = base_settings.telegram_summary_chat_id

    results = {}
    if action_message and action_token and action_chat:
        results["telegram_action"] = notifier.send_telegram_with(
            action_token, action_chat, action_message
        )

    if summary_token and summary_chat:
        results["telegram_summary"] = notifier.send_telegram_with(
            summary_token, summary_chat, summary_message
        )

    if execute_message and notifier.has_channels() and (notifier.ftqq_key or notifier.webhook_url):
        results.update(
            notifier.send(
                message=action_message,
                title="交易执行信号",
                include_ftqq=True,
            )
        )

    if results:
        print("Notification results:", results)
    else:
        print("No notification channels configured; skipping notify.")


if __name__ == "__main__":
    main()
