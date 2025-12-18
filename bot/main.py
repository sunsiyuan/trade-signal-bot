# bot/main.py
# 这个文件基本承担了“程序入口 + 多币种循环 + 生成信号 + 去重/状态机 + 发通知 + 落日志”的总编排职责。
# 你可以把它看成 orchestrator：把各模块（数据、信号、状态、通知、日志）串起来。

import json
import os
from dataclasses import asdict, is_dataclass, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import ccxt

from .config import Settings

# 默认价格步进（quantization）从 Settings 读取。
# 注意这里在 import 阶段就实例化 Settings() 了：意味着 Settings 的默认读取（env/文件）会在 import 时发生一次。
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


# =========================
# 时间相关（北京时间展示）
# =========================

def _beijing_now() -> datetime:
    """返回当前北京时间（UTC+8）的 datetime。"""
    return datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=8)))


def _beijing_time_header() -> str:
    """用于通知消息顶部的时间行。"""
    return f"⏱ 北京时间: {_beijing_now().strftime('%Y-%m-%d %H:%M')}"


# =========================
# 工具函数：把 plan 标准化成 dict
# =========================

def _plan_dict(plan):
    """
    把各种可能形态的 plan 转成 dict：
    - None -> None
    - dataclass -> asdict
    - dict -> 原样返回
    - 其他 -> None

    这个函数的核心作用：让后续代码都用 dict 访问 plan 字段，避免到处做 isinstance 判断。
    """
    if plan is None:
        return None
    if is_dataclass(plan):
        return asdict(plan)
    if isinstance(plan, dict):
        return plan
    return None


# =========================
# UI：方向 icon / 中文
# =========================

def _decision_icon(direction: str) -> str:
    """把 long/short/none 映射到图标。"""
    return {"long": "📈", "short": "📉"}.get(direction, "🧊")


def _decision_cn(direction: str) -> str:
    """把 long/short/none 映射到中文动作（这里只是方向，不是 action）。"""
    return {"long": "多", "short": "空"}.get(direction, "观望")


# =========================
# UI：Regime 展示（icon + 中文）
# =========================

def _regime_display(regime: str, trend_label: str) -> Tuple[str, str]:
    """
    将 snapshot.regime + trend_label 映射为更友好的展示文本。
    - trending: 还会细分 up/down/unknown（趋势分歧）
    - high_vol_ranging / low_vol_ranging: 直出
    - 其他: 未知
    """
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


# =========================
# UI：策略/Setup 名称映射
# =========================

def _setup_code(setup_type: str) -> str:
    """
    用于摘要行里展示 setup（更像“策略类别”）。
    注意：这里 mapping 里出现了 mean_reversion / liquidity_hunt / trend_long/short / none
    但你另一处 _setup_cn 用的是 mr_long/lh_short 等更细的 key。
    """
    mapping = {
        "trend_long": "趋势跟随(TF)",
        "trend_short": "趋势跟随(TF)",
        "mean_reversion": "均值回归(MR)",
        "liquidity_hunt": "流动性狩猎(LH)",
        "none": "无",
    }
    return mapping.get(setup_type, setup_type or "none")


def _setup_cn(setup_type: Optional[str]) -> str:
    """
    用于 action plan 消息里展示“具体策略方向”（做多/做空）。
    这里的 key 更细：mr_long / mr_short / lh_long / lh_short / trend_long/short。
    """
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


# =========================
# 格式化：百分比、价格、小数位
# =========================

def _format_pct(value: float) -> str:
    """
    这里的 value 预期是 0~1 的比例（例如 trade_confidence=0.75）。
    输出会乘以 100 并四舍五入成整数百分比字符串。
    """
    if value is None:
        return "0%"
    return f"{round(value * 100):d}%"


def _get_price_decimals(symbol: Optional[str], settings: Optional[Settings]) -> int:
    """
    根据 Settings.price_quantization 给不同 base 币种决定展示小数位。
    - mapping: { "BTC": 0.1, "ETH": 0.01, ... } 之类（取决于你的 Settings）
    - 逻辑：通过 step 的 Decimal exponent 推导小数位，再 +1（这里明显是“偏保守多显示一位”）
    - step 缺失就用 4 位
    """
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
    """统一价格展示：按币种推导小数位；None -> NA。"""
    if value is None:
        return "NA"
    decimals = _get_price_decimals(symbol, settings)
    return f"{value:.{decimals}f}"


def _display_symbol(symbol: Optional[str]) -> str:
    """
    你的 symbol 可能长这样： "ETH/USDC:USDC"
    这里做截断，取冒号前面： "ETH/USDC"
    """
    if not symbol:
        return ""
    return symbol.split(":")[0]


def _format_macd_hist(value: float) -> str:
    """MACD hist 固定 4 位小数。"""
    if value is None:
        return "NA"
    return f"{value:.4f}"


# =========================
# 展示 Levels：Entry/TP/SL
# =========================

def _format_levels(signal) -> str:
    """
    摘要/详情用的价位串（E / TP / SL）。
    - 若方向 none -> "-"
    - entry 用 4 位小数
    - tp/sl 对 >=100 的数用整数显示（适配像 ZEC 这种高价）
    """
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


# =========================
# 从 snapshot 提取核心指标（RSI/价格）
# =========================

def _extract_rsi_15m(snapshot) -> str:
    """尝试读 snapshot.tf_15m.rsi6；失败返回 NA。"""
    try:
        return f"{snapshot.tf_15m.rsi6:.1f}"
    except Exception:
        return "NA"


def _extract_mark_price(snapshot) -> Optional[float]:
    """
    尝试用“最可靠”的 mark 价：
    1) snapshot.deriv.mark_price（如果 deriv 存在）
    2) snapshot.tf_15m.prices['mark']（如果 tf_15m 的 prices dict 里有 mark）
    3) snapshot.tf_15m.price_last（如果存在）
    4) snapshot.price（兜底）
    任一步失败都吞掉异常返回 None。
    """
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


# =========================
# Gate & Reason（解释层辅助）
# =========================

def _extract_gate(signal) -> str:
    """
    从 thresholds_snapshot 或 debug_scores 里提取 gate 信息，供解释/调试展示。
    """
    thresholds = signal.thresholds_snapshot or {}
    debug_scores = signal.debug_scores or {}
    return thresholds.get("gate") or debug_scores.get("gate_tag") or "NA"


def _top_reasons(signal, action_level: str) -> str:
    """
    返回“最主要原因”（或拒绝原因）用于解释文本：
    - EXECUTE 且 direction!=none：优先返回 rejected_reasons[0]（如果存在）否则 ok
    - 非 EXECUTE：返回前 2 个 rejected_reasons（逗号拼接）
    - 否则尝试从 thresholds_snapshot 里找一个可展示的 key/value
    - 再否则返回 insufficient
    """
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
    """
    推断“偏向”（LONG/SHORT/NONE），用于 WATCH / EXECUTE 级别提示。
    优先级：
    1) conditional_plan.direction（如果有）
    2) signal.direction（如果不是 none）
    3) debug_scores long vs short（谁大选谁）
    """
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


# =========================
# 执行模式标准化（plan.execution_mode -> 内部枚举）
# =========================

def _normalize_execution_mode(plan: Optional[Dict]) -> str:
    """
    把 conditional_plan.execution_mode 统一成：
    - WATCH
    - PLACE_LIMIT_4H
    - EXECUTE_NOW
    其余都回退 WATCH
    """
    mode = (plan or {}).get("execution_mode") or "WATCH_ONLY"
    if mode == "WATCH_ONLY":
        return "WATCH"
    if mode == "PLACE_LIMIT_4H":
        return "PLACE_LIMIT_4H"
    if mode == "EXECUTE_NOW":
        return "EXECUTE_NOW"
    return "WATCH"


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    """把 isoformat 字符串解析成 datetime；失败返回 None。"""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None


# =========================
# sent_events：同 signal_id 的事件去重（跨多次运行）
# =========================

def _event_sent(sent_events: Dict[str, List[str]], signal_id: Optional[str], event: str) -> bool:
    """查询某个 signal_id 的某个 event 是否已经发过。"""
    if not signal_id:
        return False
    return event in sent_events.get(signal_id, [])


def _mark_event_sent(sent_events: Dict[str, List[str]], signal_id: Optional[str], event: str) -> None:
    """把 signal_id 的 event 标记为已发送。"""
    if not signal_id:
        return
    sent_events.setdefault(signal_id, [])
    if event not in sent_events[signal_id]:
        sent_events[signal_id].append(event)


# =========================
# 打印去重信息（should_send 的调试输出）
# =========================

def _log_dedupe(info: Dict[str, Any]) -> None:
    """
    should_send 返回 info（包含 result、原因等）。
    这里补一个 deduped 字段方便 grep：
    - deduped=True 代表 result == "DEDUPED"
    """
    enriched = {**info, "deduped": info.get("result") == "DEDUPED"}
    print(json.dumps(enriched, ensure_ascii=False))


def _update_rolling_state(
    symbol_state: Dict[str, Any],
    candidate: Optional[str],
    direction: Optional[str],
    ts: Optional[datetime],
) -> tuple[Dict[str, Any], bool]:
    """Update rolling_state in symbol_state with streak tracking."""

    rolling = symbol_state.setdefault(
        "rolling_state",
        {"candidate": None, "dir": None, "streak": 0, "last_ts": None},
    )

    prev_candidate = rolling.get("candidate")
    prev_dir = rolling.get("dir")
    prev_streak = int(rolling.get("streak", 0) or 0)
    prev_ts = rolling.get("last_ts")

    normalized_candidate = candidate
    normalized_dir = direction if candidate == "trending" else None

    if normalized_candidate == prev_candidate and normalized_dir == prev_dir:
        streak = prev_streak + 1
    else:
        streak = 1

    rolling["candidate"] = normalized_candidate
    rolling["dir"] = normalized_dir
    rolling["streak"] = streak
    rolling["last_ts"] = ts.isoformat() if ts else None

    changed = (
        rolling["candidate"] != prev_candidate
        or rolling["dir"] != prev_dir
        or rolling["streak"] != prev_streak
        or rolling["last_ts"] != prev_ts
    )

    return rolling, changed


def _log_rolling_state(symbol: str, snapshot, rolling_prepared: bool) -> None:
    candidate = getattr(snapshot, "rolling_candidate", None)
    direction = getattr(snapshot, "rolling_candidate_dir", None)
    streak = getattr(snapshot, "rolling_candidate_streak", None)
    print(
        json.dumps(
            {
                "type": "rolling_state",
                "symbol": symbol,
                "candidate": candidate,
                "dir": direction,
                "streak": streak,
                "rolling_prepared": rolling_prepared,
            },
            ensure_ascii=False,
        )
    )


# =========================
# Summary：更紧凑的一行（用于 summary bot）
# =========================

def format_summary_compact(symbol, snapshot, action: str) -> str:
    """
    单行概览：symbol | price | regime | action_label
    - price 优先 mark_price；兜底 tf_15m.close
    """
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
    """
    提取 15m RSI6 的数值（float）：
    - 优先 snapshot.tf_15m.indicators['rsi_6']（如果 indicators 是 dict）
    - 否则 snapshot.tf_15m.rsi6
    """
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
    """
    把 valid_until_utc（字符串）转成北京时间显示。
    若解析失败，直接返回原值或 N/A。
    """
    valid_until = plan.get("valid_until_utc")
    dt = _parse_dt(valid_until)
    if dt:
        return dt.astimezone(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M")
    return valid_until or "N/A"


def _format_tp_values(signal, plan: Dict) -> str:
    """
    汇总 TP1/2/3：
    - 优先 plan 里的 tp1/tp2/tp3
    - 其次 signal.tp 容器（可能是 dataclass 或 dict）
    - 再其次 signal.tp1/tp2/tp3
    统一用 _format_price（会按币种决定小数位）
    """
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
    """
    SL 显示的候选顺序（从更显式到更兜底）：
    1) plan['sl']
    2) signal.sl
    3) signal.execution_intent.invalidation_price
    4) plan['invalidation_price']
    找到第一个非 None 的就格式化输出。
    """
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


# =========================
# Action plan：限价单 / 立即执行 消息模板
# =========================

def format_action_plan_message(
    signal,
    snap,
    plan: Dict,
    signal_id: str,
    event: str = "CREATED",
    reason: str = "",
) -> str:
    """
    输出“可执行动作”的完整文本（面向 action bot / 方糖）：
    - 包含：时间、事件类型、signal_id、标的/方向/模式、现价与RSI、入场/SL/TP、有效期、原因

    参数说明：
    - signal: TradeSignal（可能为 None，比如 reconcile 旧计划时只用 plan）
    - snap: MarketSnapshot（用于取现价/RSI）
    - plan: dict（必须尽量完整）
    - event: CREATED/EXECUTE_NOW/EXPIRED/INVALIDATED/REGIME_CHANGED...
    """
    plan = _plan_dict(plan) or {}
    symbol = plan.get("symbol") or getattr(signal, "symbol", "")
    display_symbol = _display_symbol(symbol)

    # 现价优先 mark
    price = _format_price(_extract_mark_price(snap), symbol=symbol)

    # 15m RSI6（用于快速判断热度）
    rsi6 = _extract_rsi6_value(snap)
    rsi_text = f"{rsi6:.1f}" if rsi6 is not None else "NA"

    direction = (plan.get("direction") or getattr(signal, "direction", "")) or ""
    setup_type = getattr(signal, "setup_type", None) or plan.get("setup_type")

    # 这里变量名叫 execution_mode，但实际放的是“策略中文”（_setup_cn 输出），可能命名上会让人误会。
    execution_mode = _setup_cn(setup_type)

    entry_price = plan.get("entry_price")
    entry_text = (
        _format_price(entry_price, symbol=symbol) if entry_price is not None else "-"
    )
    sl_text = _format_sl_value(signal, plan)
    tp_text = _format_tp_values(signal, plan)
    valid_until = _format_valid_until(plan)
    reason_text = reason or plan.get("explain") or getattr(signal, "reason", "") or "-"

    # event -> 中文显示（用于消息标题）
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


# =========================
# Action label：给 summary line 用（WATCH/LIMIT/EXECUTE）
# =========================

def _action_label(action: str) -> str:
    """
    当前动作（不是方向）：
    - WATCH：值得关注（但不一定下单）
    - LIMIT_4H：创建一个 4H 有效的限价计划
    - EXECUTE_NOW：立刻执行
    - NONE：不输出动作
    """
    mapping = {
        "WATCH": "🧊 观望",
        "LIMIT_4H": "⏳ 限价4H",
        "EXECUTE_NOW": "⚡️ 立即执行",
        "NONE": "⏸️ 暂无动作",
    }
    return mapping.get(action, "⏸️ 暂无动作")


# =========================
# 可行动判断（目前 main 里没直接用它驱动下单，只是工具函数）
# =========================

def is_actionable(signal, snapshot, settings: Settings):
    """
    用 trade_confidence / edge_confidence + 一些阈值配置，判断是否进入 WATCH 或 EXECUTE。

    这里的设计意图大概是：
    - EXECUTE：方向明确且 trade_conf 达到 execute 门槛
    - WATCH：满足“机会出现（edge 高）+ 信心尚可（trade 不太低）”或“接近 execute 的 near-miss”等
    - 有 conditional_plan 也算 watch（因为已经有明确执行意图）

    返回：(bool是否值得发, action_level字符串, bias LONG/SHORT/NONE)
    """
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


# =========================
# 另一种 summary（更长），目前 main 没用到
# =========================

def format_summary_line(symbol, snapshot, signal) -> str:
    """
    更长的一行（包含 Trade/Edge、RSI、MACD、Setup 等）。
    目前 main 用的是 format_summary_compact。
    """
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


def format_conditional_plan_line(signal) -> str:
    """
    将 conditional_plan 简化成一行文字（用于展示当前计划）。
    目前 main 也没用到（可能是之前版本留存）。
    """
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


# =========================
# 结构化日志：打印 + 写 jsonl
# =========================

def emit_multi_tf_log(snapshot, signal, settings: Settings, exchange_id: str = "") -> None:
    """
    生成结构化事件：
    - build_signal_event(...)：把 snapshot/signal/settings 等打包成统一 schema（便于回放/分析）
    - stdout 打印一份（GitHub Actions log 可见）
    - 写入 jsonl 文件（默认 data/logs/signals.jsonl，可用 env LOG_JSONL_PATH 覆盖）
    """
    event = build_signal_event(snapshot, signal, settings, exchange_id=exchange_id)
    print(json.dumps(event, ensure_ascii=False))
    log_path = os.getenv("LOG_JSONL_PATH", "data/logs/signals.jsonl")
    write_jsonl_event(event, log_path)


# =========================
# 主入口：多币种跑一轮 -> reconcile 旧计划 -> 处理新信号 -> 通知 -> 持久化状态
# =========================

def main():
    # 读取基础配置（env / config），并决定监控币种列表
    base_settings = Settings()
    tracked = base_settings.tracked_symbols or [base_settings.symbol]

    # 全局状态文件：目前用于 active_plans / sent_events（跨币种共享）
    state_path = os.path.join(".state", "state.json")
    state = load_global_state(state_path)

    # 每个 symbol 独立的状态（用于 should_send / mark_sent 的去重记忆）
    symbol_states: Dict[str, Dict[str, Any]] = {}
    dirty_symbols: set[str] = set()  # 哪些 symbol 的 state 需要写回磁盘

    def _get_symbol_state(sym: str) -> Dict[str, Any]:
        """lazy-load 某个 symbol 的 state（避免每次都读磁盘）。"""
        if sym not in symbol_states:
            symbol_states[sym] = load_state(sym)
        return symbol_states[sym]

    # 初始化交易所（Hyperliquid）对象：enableRateLimit=True 让 ccxt 自己做节流
    exchange = ccxt.hyperliquid({"enableRateLimit": True})
    exchange.load_markets()

    # funding_rates 可选：取不到也不阻塞主流程
    funding_rates = None
    try:
        funding_rates = exchange.fetch_funding_rates()
    except Exception:
        funding_rates = None

    # 信号引擎 + 通知器
    engine = SignalEngine(base_settings)
    notifier = Notifier(
        ftqq_key=base_settings.ftqq_key,          # 方糖 key（用于更强提醒）
        webhook_url=base_settings.webhook_url,    # 其他 webhook（如果有）
    )

    # 先把所有 tracked symbol 的 snapshot + signal 跑出来（一次性）
    signals = []
    snapshots = {}
    for symbol in tracked:
        # 为每个 symbol 派生一份 settings（只替换 symbol 字段）
        symbol_settings = replace(base_settings, symbol=symbol)

        # 数据 client：负责拉K线/指标/衍生数据并构造 MarketSnapshot
        client = HyperliquidDataClient(
            symbol_settings, exchange=exchange, funding_rates=funding_rates
        )
        snapshot = client.build_market_snapshot()

        # rolling regime candidate：使用 forming bar 先行确认，提前准备 PLACE_LIMIT_4H
        symbol_state = _get_symbol_state(symbol)
        rolling_state, rolling_dirty = _update_rolling_state(
            symbol_state,
            getattr(snapshot, "rolling_candidate", None),
            getattr(snapshot, "rolling_candidate_dir", None),
            getattr(snapshot, "ts", None),
        )
        snapshot.rolling_candidate = rolling_state.get("candidate")
        snapshot.rolling_candidate_dir = rolling_state.get("dir")
        snapshot.rolling_candidate_streak = int(rolling_state.get("streak", 0) or 0)
        rolling_prepared = (
            snapshot.rolling_candidate == "trending"
            and snapshot.rolling_candidate_dir in {"up", "down"}
            and snapshot.rolling_candidate_streak >= 2
        )
        setattr(snapshot, "rolling_prepared", rolling_prepared)
        _log_rolling_state(symbol, snapshot, rolling_prepared)
        if rolling_dirty:
            dirty_symbols.add(symbol)

        # 从 snapshot 生成 signal
        signal = engine.generate_signal(snapshot)

        # 给 signal 注入 signal_id（用于 dedupe、状态机、通知关联）
        signal.signal_id = compute_signal_id(
            signal, price_quantization=base_settings.price_quantization
        )

        # 结构化日志（stdout + jsonl）
        emit_multi_tf_log(snapshot, signal, symbol_settings, exchange_id=exchange.id)

        signals.append(signal)
        snapshots[symbol] = snapshot

    summary_lines = []           # summary bot 要发的一行行
    action_messages = []         # action bot 要发的多段（可执行动作）
    execute_now_messages = []    # 方糖等强提醒要发的（通常只有 EXECUTE_NOW）
    header = _beijing_time_header()
    now = datetime.now(timezone.utc)

    # ==========================================================
    # Step 1: reconcile 既有 active_plans（旧计划的过期/失效/regime变化）
    # ==========================================================
    for symbol, plan in list(state.get("active_plans", {}).items()):
        # 注意：state['active_plans'] 存的是 plan dict（之前运行时写进去的）
        signal_id = plan.get("signal_id")
        snap = snapshots.get(symbol)
        mark_price = _extract_mark_price(snap)
        regime = getattr(snap, "regime", None) if snap else None

        event = None
        reason = ""

        # 1) 过期检查
        valid_until = _parse_dt(plan.get("valid_until_utc"))
        if valid_until and now > valid_until:
            event = "EXPIRED"
            reason = "超过有效期，撤销计划单"

        # 2) 失效位检查（invalidation_price）
        elif mark_price is not None and plan.get("invalidation_price") is not None:
            if plan.get("direction") == "long" and mark_price <= plan.get("invalidation_price"):
                event = "INVALIDATED"
                reason = "价格跌破失效位"
            elif plan.get("direction") == "short" and mark_price >= plan.get("invalidation_price"):
                event = "INVALIDATED"
                reason = "价格突破失效位"

        # 3) Regime 变化检查：如果计划创建时 regime != 当前 regime，则认为计划不再适用
        if event is None and regime and plan.get("regime") and plan.get("regime") != regime:
            event = "REGIME_CHANGED"
            reason = f"Regime {plan.get('regime')} → {regime}"

        # 如果触发了某种“计划结束事件”，则（可选）发一条通知，并把计划从 active_plans 中移除
        if event:
            # sent_events 基于 signal_id+event 做强去重：同一个信号不重复发 EXPIRED/INVALIDATED 等
            if not _event_sent(state.get("sent_events", {}), signal_id, event):
                plan_for_msg = {**plan, "symbol": symbol}
                action_messages.append(
                    format_action_plan_message(
                        None, snap, plan_for_msg, signal_id or "", event=event, reason=reason
                    )
                )
                _mark_event_sent(state.setdefault("sent_events", {}), signal_id, event)

            # 无论是否发消息，计划都从 active_plans 移除
            state.get("active_plans", {}).pop(symbol, None)

    # ==========================================================
    # Step 2: 处理本轮新 signals（WATCH / PLACE_LIMIT_4H / EXECUTE_NOW）
    # ==========================================================
    for sig in signals:
        snap = sig.snapshot

        # conditional_plan 里包含“执行模式/入场/有效期/解释”等信息
        plan = _plan_dict(getattr(sig, "conditional_plan", None)) or {}

        # 把 execution_mode 标准化成 WATCH / PLACE_LIMIT_4H / EXECUTE_NOW
        mode = _normalize_execution_mode(plan)

        current_action = "NONE"

        # entry_price 来自 plan
        entry_price = plan.get("entry_price")

        # invalidation_price（失效位）优先从 execution_intent 里拿，其次用 signal.sl
        invalidation_price = None
        if getattr(sig, "execution_intent", None):
            invalidation_price = sig.execution_intent.invalidation_price
        elif hasattr(sig, "sl"):
            invalidation_price = getattr(sig, "sl", None)

        # signal_id：若 signal 里已有就用，否则现算一个
        signal_id = getattr(sig, "signal_id", None) or compute_signal_id(
            sig, price_quantization=base_settings.price_quantization
        )

        # base_plan：写入全局 active_plans 时用，也用于通知消息展示
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

        # ----------------------------
        # 分支 A：WATCH（只做关注，不创建计划）
        # ----------------------------
        if mode == "WATCH":
            current_action = "WATCH"
            # 注意：这里 should_send 的 symbol_state 传的是 {}（空 dict），
            # 意味着 WATCH 的去重可能不落到 per-symbol state（取决于 should_send 实现）。
            allowed, info = should_send(
                {},
                signal_id,
                sig.symbol,
                current_action,
                now,
                action_hash=None,
            )
            _log_dedupe(info)

        # ----------------------------
        # 分支 B：PLACE_LIMIT_4H（创建 4H 限价计划并写入 active_plans）
        # ----------------------------
        elif mode == "PLACE_LIMIT_4H":
            current_action = "LIMIT_4H"
            symbol_state = _get_symbol_state(sig.symbol)

            # reason 用 plan.explain 优先，否则用 sig.reason，再兜底固定文案
            reason = plan.get("explain") or sig.reason or "创建4H限价计划"

            # action_hash：把 action + payload 哈希，确保“相同计划”不会重复发消息
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
                # 发 action bot 的消息（CREATED）
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
                # 同时记录 sent_events，避免同 signal_id 重复发 CREATED
                _mark_event_sent(state.setdefault("sent_events", {}), signal_id, "CREATED")

                # mark_sent：把这次发送写到 per-symbol state（用于跨运行去重/TTL）
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

            # 无论 allowed 与否，都把 base_plan 写入 active_plans
            # 这意味着：即使 dedupe 拦住不发消息，plan 依然会被维护，
            # 之后 Step 1 会对其做 EXPIRED/INVALIDATED/REGIME_CHANGED 的管理。
            state.setdefault("active_plans", {})[sig.symbol] = base_plan

        # ----------------------------
        # 分支 C：EXECUTE_NOW（立即执行：action bot + 方糖强提醒）
        # ----------------------------
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
                # action bot（Telegram action channel）
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

                # execute_now_messages（用于方糖等更强提醒渠道）
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

        # 每个币都输出一行 compact summary（包含 current_action）
        summary_lines.append(format_summary_compact(sig.symbol, snap, current_action))

    # 汇总消息：顶部时间 + 多行币种概览
    summary_message = "\n".join([header] + summary_lines)
    print(summary_message)

    # 配置里拆了两个 Telegram bot（summary vs action）
    action_token = base_settings.telegram_action_token
    action_chat = base_settings.telegram_action_chat_id
    summary_token = base_settings.telegram_summary_token
    summary_chat = base_settings.telegram_summary_chat_id

    results = {}

    # 1) 发 summary bot（轻量消息）
    if summary_token and summary_chat and summary_message:
        results["telegram_summary"] = notifier.send_telegram(
            summary_message, token=summary_token, chat_id=summary_chat
        )

    # 2) 发 action bot（可执行动作，可能多段）
    if action_messages and action_token and action_chat:
        results["telegram_action"] = notifier.send_telegram(
            "\n\n".join(action_messages), token=action_token, chat_id=action_chat
        )

    # 3) 方糖：只在 EXECUTE_NOW 的时候发（更强提醒）
    if execute_now_messages and notifier.ftqq_key:
        combined = "\n\n".join(execute_now_messages)
        results.update(
            notifier.send(
                message=combined,
                title="交易执行信号",
                include_ftqq=bool(notifier.ftqq_key),
            )
        )

    # 写回 per-symbol state（只写 dirty 的，减少 IO）
    for symbol in dirty_symbols:
        save_state(symbol, symbol_states[symbol])

    # 写回 global state（active_plans + sent_events 等）
    save_global_state(state_path, state)

    # 打印通知结果，方便在 Actions log 里确认是否发送成功
    if results:
        print("Notification results:", results)
    else:
        print("No notification channels configured; skipping notify.")


# 支持 python -m bot.main 或直接 python bot/main.py 运行
if __name__ == "__main__":
    main()
