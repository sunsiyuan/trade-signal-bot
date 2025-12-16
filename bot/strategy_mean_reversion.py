# =========================
# strategy_mean_reversion.py
# =========================
# 目标：在震荡（ranging）行情里做“均值回归”（回归 MA25）
# 核心触发：价格偏离 MA25 达到一定 ATR 倍数 + RSI 极端 + OI 变化（或允许 OI 缺失 fallback）
# 输出：TradeSignal(long/short) 或 None（不触发）
# =========================

from typing import Optional, TYPE_CHECKING

from .conditional_plan import resolve_atr_4h
from .models import ExecutionIntent, MarketSnapshot, Direction
from .regime_detector import Regime

# 为了避免运行时循环 import：只有类型检查时才导入 TradeSignal
if TYPE_CHECKING:  # pragma: no cover
    from .signal_engine import TradeSignal


# -------------------------
# 帮手：取某个 timeframe 对象
# -------------------------
def _get_tf(snap: MarketSnapshot, tf: str):
    """
    从 MarketSnapshot 里取出对应 tf 的指标对象
    例如 tf="1h" 返回 snap.get_timeframe("1h")
    """
    return snap.get_timeframe(tf)


# -------------------------
# 帮手：读取 settings.mean_reversion.xxx
# 兼容 Settings 对象和 dict 两种结构
# -------------------------
def _get_nested(settings, group: str, key: str, default):
    """
    读取 settings[group][key] 的通用函数，兼容：
    1) settings 是 Settings 对象：
       - hasattr(settings, group) -> container = settings.group
       - container 可能是 dict 或对象
    2) settings 是 dict：
       - settings[group] 是一个 dict
    读取失败则返回 default
    """
    if hasattr(settings, group):
        container = getattr(settings, group)
        if isinstance(container, dict):
            return container.get(key, default)
        if hasattr(container, key):
            return getattr(container, key)
    if isinstance(settings, dict) and group in settings:
        return settings[group].get(key, default)
    return default


def _clamp(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    return max(min_value, min(max_value, value))


# =========================
# 主函数：构建均值回归信号
# =========================
def build_mean_reversion_signal(
    snap: MarketSnapshot, regime: Regime, settings
) -> Optional["TradeSignal"]:
    """Generate a mean-reversion signal when the market is ranging."""

    def _attach_intent(ts: "TradeSignal") -> "TradeSignal":
        ts.execution_intent = build_execution_intent_mr(snap, ts)
        return ts

    # ---- 0) 仅在 ranging regime 才运行 ----
    # 注意：如果 regime 不是 high/low vol ranging，这里直接返回 None
    # 上层（signal_engine._decide_range）收到 None 就认为“这个策略不触发”
    if regime not in {"high_vol_ranging", "low_vol_ranging"}:
        return None

    # ---- 1) 读入 mean_reversion 策略参数（全部可调） ----
    # tf：使用哪个周期做 MR（默认 1h）
    tf_name = _get_nested(settings, "mean_reversion", "tf", "1h")

    # RSI 触发阈值：极端超卖/超买（默认非常极端：12/88）
    rsi_oversold = _get_nested(settings, "mean_reversion", "rsi_oversold", 12)
    rsi_overbought = _get_nested(settings, "mean_reversion", "rsi_overbought", 88)

    # 价格偏离 MA25 的 ATR 倍数阈值：越大越“等极端再出手”（默认 1.2 ATR）
    atr_dev_mult = _get_nested(settings, "mean_reversion", "atr_dev_mult", 1.6)

    # OI 变化阈值：认为“flush / squeeze” 的最小变化百分比（默认 3%）
    min_oi_change_pct = _get_nested(settings, "mean_reversion", "min_oi_change_pct", 3.0)

    # 这里虽然读了 tp_to_sl_ratio，但当前实现并未用到（保留未来扩展）
    tp_to_sl_ratio = _get_nested(settings, "mean_reversion", "tp_to_sl_ratio", 1.5)

    # 仓位建议：核心仓位 + 加仓仓位（默认 0.5 / 0.25）
    core_pct = _get_nested(settings, "mean_reversion", "core_position_pct", 0.5)
    add_pct = _get_nested(settings, "mean_reversion", "add_position_pct", 0.25)

    # SL buffer：ATR 的倍数（默认 0.8 ATR）
    sl_buffer_mult = _get_nested(settings, "mean_reversion", "sl_buffer_mult", 0.4)

    # 是否强制要求 OI（默认 True）
    require_oi = _get_nested(settings, "mean_reversion", "require_oi", True)

    # 如果 OI 缺失，是否允许 fallback 仍然出信号（默认 True）
    allow_oi_missing_fallback = _get_nested(
        settings, "mean_reversion", "allow_oi_missing_fallback", True
    )

    # fallback 模式的“降权系数”：降低 confidence / 缩小仓位（默认 0.75）
    fallback_confidence_mult = _get_nested(
        settings, "mean_reversion", "fallback_confidence_mult", 0.75
    )
    fallback_core_position_mult = _get_nested(
        settings, "mean_reversion", "fallback_core_position_mult", 0.5
    )
    fallback_add_position_mult = _get_nested(
        settings, "mean_reversion", "fallback_add_position_mult", 0.0
    )

    # ---- 2) 取指标：价格、MA25、ATR、RSI6、OI_change ----
    tf = _get_tf(snap, tf_name)

    price = tf.close
    ma25 = tf.ma25
    atr = max(tf.atr, 1e-6)     # 防止 atr=0 导致除零
    rsi6 = tf.rsi6

    # OI 变化（百分比），来自衍生品信息 deriv
    oi_change_24h = snap.deriv.oi_change_24h
    oi_missing = oi_change_24h is None

    # ---- 3) 处理 OI 缺失策略 ----
    # 如果缺 OI 且 require_oi=True 且 allow_fallback=False，则直接不触发（返回 None）
    if oi_missing and require_oi and not allow_oi_missing_fallback:
        return None

    # =========================
    # A) 做多（long）均值回归触发条件
    # =========================

    # 条件 1：价格显著低于 MA25（低于 MA25 - atr_dev_mult*ATR）
    cond_far_below_ma = price <= ma25 - atr_dev_mult * atr

    # 条件 2：RSI6 极端超卖
    cond_oversold = rsi6 <= rsi_oversold

    # 条件 3：OI 正在“出清”（OI_change_24h <= -min_oi_change_pct）
    # 解释：OI 显著下降可能代表多空平仓出清 → 震荡里更容易反弹回均线
    cond_oi_flushing_out = (
        oi_change_24h is not None and oi_change_24h <= -min_oi_change_pct
    )

    # fallback_mode：OI 缺失但允许 fallback
    fallback_mode = oi_missing and allow_oi_missing_fallback

    # 方向 bias & edge confidence 由策略内部计算
    rsi_span = max(rsi_overbought - rsi_oversold, 1e-6)
    margin = 5

    long_rsi_score = _clamp((rsi_oversold + margin - rsi6) / rsi_span, 0.0, 1.0)
    short_rsi_score = _clamp((rsi6 - (rsi_overbought - margin)) / rsi_span, 0.0, 1.0)

    long_dev_score = _clamp((ma25 - price) / (atr_dev_mult * atr + 1e-9), 0.0, 1.0)
    short_dev_score = _clamp((price - ma25) / (atr_dev_mult * atr + 1e-9), 0.0, 1.0)

    long_oi_score = _clamp(-(oi_change_24h or 0.0) / max(min_oi_change_pct, 1e-6), 0.0, 1.0)
    short_oi_score = _clamp((oi_change_24h or 0.0) / max(min_oi_change_pct, 1e-6), 0.0, 1.0)

    long_score = _clamp(0.5 * long_dev_score + 0.35 * long_rsi_score + 0.15 * long_oi_score)
    short_score = _clamp(0.5 * short_dev_score + 0.35 * short_rsi_score + 0.15 * short_oi_score)

    edge_confidence = _clamp(
        max(long_score, short_score)
        * (fallback_confidence_mult if fallback_mode else 1.0)
    )

    debug_scores = {
        "long": round(long_score, 4),
        "short": round(short_score, 4),
        "long_rsi": round(long_rsi_score, 4),
        "short_rsi": round(short_rsi_score, 4),
        "long_dev": round(long_dev_score, 4),
        "short_dev": round(short_dev_score, 4),
        "long_oi": round(long_oi_score, 4),
        "short_oi": round(short_oi_score, 4),
        "fallback_mode": 1.0 if fallback_mode else 0.0,
    }

    rejected_reasons = ["oi_missing_fallback"] if fallback_mode else []

    # 触发逻辑：必须同时满足
    # 1) 远离均线（下方） + 2) 超卖 + 3) OI 出清 或者 OI 缺失 fallback
    if cond_far_below_ma and cond_oversold and (cond_oi_flushing_out or fallback_mode):
        from .signal_engine import TradeSignal

        # ---- 构建下单计划（做多） ----
        # 结构单设计
        # 止损：1 atr是结构破坏，sl_buffer_mult是噪音缓冲
        entry = ma25 - atr_dev_mult * atr
        sl = entry - 1 * atr - sl_buffer_mult * atr

        # 止盈：先看回归 MA25，再看略高于 MA25 半个 ATR
        # 这是典型“回归均线”的 TP 设计
        tp1 = ma25
        tp2 = ma25 + 0.5 * atr

        # rr（风险回报比）估算：到 tp1 的收益 / 到 sl 的风险
        risk = abs(entry - sl)
        reward = abs(tp1 - entry)
        rr = reward / max(risk, 1e-9)

        # trade_confidence：基础 0.6，然后按 rr 增加一些，封顶 0.9
        # 注意：rr 越大（离均线越远 or sl 越近）→ confidence 越高
        trade_confidence = min(0.9, 0.6 + 0.1 * rr)

        # 如果是 fallback（OI 缺失），则降低 confidence 并缩仓位
        if fallback_mode:
            confidence *= fallback_confidence_mult
            core_pct *= fallback_core_position_mult
            add_pct *= fallback_add_position_mult

        # reason：给输出解释用
        reason = (
            f"Mean reversion long: price {price:.4f} < MA25 {ma25:.4f} - {atr_dev_mult} ATR, "
            f"RSI6={rsi6:.1f} oversold, OI_change={oi_change_24h if oi_change_24h is not None else float('nan'):.1f}% "
            f"{'flushing out' if cond_oi_flushing_out else 'missing'}"
        )
        if fallback_mode:
            reason += " | OI missing → fallback mode"

        # 输出做多信号
        signal = TradeSignal(
            symbol=snap.symbol,
            direction="long",
            trade_confidence=trade_confidence,
            edge_confidence=edge_confidence,
            entry=entry,
            tp1=tp1,
            tp2=tp2,
            sl=sl,
            core_position_pct=core_pct,
            add_position_pct=add_pct,
            reason=reason,
            setup_type="mr_long",
            snapshot=snap,
            debug_scores=debug_scores,
            rejected_reasons=rejected_reasons or None,
        )
        signal.edge_type = "位置优势"
        return _attach_intent(signal)

    # =========================
    # B) 做空（short）均值回归触发条件
    # =========================

    # 条件 1：价格显著高于 MA25（高于 MA25 + atr_dev_mult*ATR）
    cond_far_above_ma = price >= ma25 + atr_dev_mult * atr

    # 条件 2：RSI6 极端超买
    cond_overbought = rsi6 >= rsi_overbought

    # 条件 3：OI 正在“挤压”（OI_change_24h >= min_oi_change_pct）
    # 解释：OI 显著上升可能代表加杠杆追涨/追跌 → 震荡里更容易被反向收割回均线
    cond_oi_squeezing = (
        oi_change_24h is not None and oi_change_24h >= min_oi_change_pct
    )

    # 触发逻辑：同 long 对称
    if cond_far_above_ma and cond_overbought and (cond_oi_squeezing or fallback_mode):
        from .signal_engine import TradeSignal

        # ---- 构建下单计划（做空） ----
        # 结构单设计
        # 止损：1 atr是结构破坏，sl_buffer_mult是噪音缓冲
        entry = ma25 + atr_dev_mult * atr
        sl = entry + 1 * atr + sl_buffer_mult * atr

        # 止盈：先回归 MA25，再略低于 MA25 半个 ATR
        tp1 = ma25
        tp2 = ma25 - 0.5 * atr

        # rr 估算（到 tp1 的收益 / 到 sl 的风险）
        risk = abs(entry - sl)
        reward = abs(tp1 - entry)
        rr = reward / max(risk, 1e-9)

        # confidence 同 long：基础 0.6 + 0.1*rr，封顶 0.9
        trade_confidence = min(0.9, 0.6 + 0.1 * rr)

        # OI 缺失 fallback 降权与缩仓
        if fallback_mode:
            confidence *= fallback_confidence_mult
            core_pct *= fallback_core_position_mult
            add_pct *= fallback_add_position_mult

        reason = (
            f"Mean reversion short: price {price:.4f} > MA25 {ma25:.4f} + {atr_dev_mult} ATR, "
            f"RSI6={rsi6:.1f} overbought, OI_change={oi_change_24h if oi_change_24h is not None else float('nan'):.1f}% "
            f"{'squeezing' if cond_oi_squeezing else 'missing'}"
        )
        if fallback_mode:
            reason += " | OI missing → fallback mode"

        signal = TradeSignal(
            symbol=snap.symbol,
            direction="short",
            trade_confidence=trade_confidence,
            edge_confidence=edge_confidence,
            entry=entry,
            tp1=tp1,
            tp2=tp2,
            sl=sl,
            core_position_pct=core_pct,
            add_position_pct=add_pct,
            reason=reason,
            setup_type="mr_short",
            snapshot=snap,
            debug_scores=debug_scores,
            rejected_reasons=rejected_reasons or None,
        )
        signal.edge_type = "位置优势"
        return _attach_intent(signal)

    # ---- 没有触发任何一边 → 返回 None ----
    # 上层会把它视为“MR 没出手”，然后可能尝试 LH 或返回 none
    return None


def build_execution_intent_mr(
    snap: MarketSnapshot, signal: "TradeSignal"
) -> ExecutionIntent:
    ma25 = snap.tf_1h.ma25

    return ExecutionIntent(
        symbol=snap.symbol,
        direction=signal.direction,
        entry_price=signal.entry,
        entry_reason="MR_ENTRY",
        invalidation_price=signal.sl,
        atr_4h=resolve_atr_4h(snap),
        ttl_hours=2,
        reason=signal.reason,
        debug=signal.debug_scores,
    )
