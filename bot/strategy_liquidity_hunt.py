# =========================
# strategy_liquidity_hunt.py
# =========================
# 目标：在 high_vol_ranging（高波动震荡）里捕捉“流动性猎杀”（Liquidity Hunt）
#
# 核心想法：
# - 价格靠近区间 swing high / swing low（容易发生假突破）
# - 盘口出现明显墙（ask wall / bid wall），说明有“可见流动性”
# - OI 出现 spike/flush（代表追突破/被迫平仓）
# - 假突破之后出现连续“小实体蜡烛”（post-spike small candles），代表“冲高/杀低失败 → 回归”
#
# 输出：
# - 触发 short：靠近 swing_high + ask wall + OI spike + 事后小蜡烛 → 做空回归
# - 触发 long ：靠近 swing_low  + bid wall + OI flush + 事后小蜡烛 → 做多回归
#
# 注意：这个策略“只在 high_vol_ranging 下启用”，low_vol_ranging 不做 LH。
# =========================

from typing import Optional, TYPE_CHECKING

from .conditional_plan import resolve_atr_4h
from .models import ExecutionIntent, MarketSnapshot
from .regime_detector import Regime

# 避免运行时循环 import：只有类型检查时才导入 TradeSignal
if TYPE_CHECKING:  # pragma: no cover
    from .signal_engine import TradeSignal


# -------------------------
# 帮手：从 snapshot 中取指定 tf 的 Timeframe 对象
# -------------------------
def _get_tf(snap: MarketSnapshot, tf: str):
    return snap.get_timeframe(tf)


# -------------------------
# 帮手：读取 settings.liquidity_hunt.xxx
# 兼容 Settings 对象与 dict 两种结构
# -------------------------
def _get_nested(settings, group: str, key: str, default):
    """
    读取 settings[group][key]，兼容：
    - settings 是 Settings 对象：settings.liquidity_hunt.xxx
    - settings 是 dict：settings["liquidity_hunt"]["xxx"]
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


# -------------------------
# LH 所依赖的关键字段集合（概念层面）
# 注：这里并没有直接用这个集合做校验，而是作为“文档/约束提示”
# -------------------------
LH_REQUIRED = {
    "tf.recent_high",
    "tf.recent_low",
    "tf.post_spike_small_body_count",
    "deriv.ask_wall_size",
    "deriv.bid_wall_size",
    "deriv.has_large_ask_wall",
    "deriv.has_large_bid_wall",
}


# -------------------------
# 检查 LH 所需字段是否缺失（用于 missing fallback）
# -------------------------
def _lh_missing_fields(snap: MarketSnapshot, tf) -> list:
    """
    返回缺失字段列表，用于：
    - allow_fallback_when_missing=True：缺失时仍可“降级触发”
    - allow_fallback_when_missing=False：缺失时直接返回 None
    """
    missing = []

    # swing high/low：用于判断是否“靠近区间边缘”
    if tf.recent_high is None or tf.recent_low is None:
        missing.append("tf.recent_high/recent_low")

    # post-spike small candles：用于确认“假突破失败”
    if tf.post_spike_small_body_count is None:
        missing.append("tf.post_spike_small_body_count")

    # deriv（衍生品 / 订单簿墙）是 LH 的核心数据源
    deriv = getattr(snap, "deriv", None)
    if deriv is None:
        missing.append("snap.deriv")
        return missing

    # 墙的 size
    if not hasattr(deriv, "ask_wall_size") or not hasattr(deriv, "bid_wall_size"):
        missing.append("deriv.ask_wall_size/bid_wall_size")

    # 墙的 boolean 判断（是否大墙）
    if not hasattr(deriv, "has_large_ask_wall") or not hasattr(
        deriv, "has_large_bid_wall"
    ):
        missing.append("deriv.has_large_*_wall")

    return missing


# =========================
# 主函数：生成 LH 信号
# =========================
def build_liquidity_hunt_signal(
    snap: MarketSnapshot, regime: Regime, settings
) -> Optional["TradeSignal"]:
    """Look for liquidity hunt setups around swing highs/lows in high vol ranges."""

    def _attach_intent(ts: "TradeSignal") -> "TradeSignal":
        ts.execution_intent = build_execution_intent_lh(snap, ts)
        return ts

    # ---- 0) 只在 high_vol_ranging 才启用 LH ----
    # 低波动震荡不做 LH（因为“猎杀”通常发生在波动更大的区间）
    if regime != "high_vol_ranging":
        return None

    # ---- 1) 读取策略参数（全部可调） ----
    # tf：用哪个周期的 swing high/low（默认 1h）
    tf_name = _get_nested(settings, "liquidity_hunt", "tf", "1h")

    # 价格接近 swing 的判定阈值（默认 0.004=0.4%）
    # 注意：后面用的是 *100 转成百分比对比，因此这里本质是“相对价格比例”
    price_proximity_pct = _get_nested(settings, "liquidity_hunt", "price_proximity_pct", 0.004)

    # 墙大小倍数：ask_wall >= min_wall_mult * bid_wall 视为大 ask wall（默认 3x）
    min_wall_mult = _get_nested(settings, "liquidity_hunt", "min_wall_mult", 3.0)

    # OI spike/flush 的最小幅度（默认 5%）
    min_oi_spike_pct = _get_nested(settings, "liquidity_hunt", "min_oi_spike_pct", 5.0)

    # post-spike 小实体 candle 连续数量要求（默认 >=3 根）
    post_spike_candle_count = _get_nested(settings, "liquidity_hunt", "post_spike_candle_count", 3)

    # SL buffer：止损比 fake breakout 的 extreme 再多给一点（默认 0.15%）
    sl_buffer_pct = _get_nested(settings, "liquidity_hunt", "sl_buffer_pct", 0.0015)

    # 仓位建议
    core_pct = _get_nested(settings, "liquidity_hunt", "core_position_pct", 0.5)
    add_pct = _get_nested(settings, "liquidity_hunt", "add_position_pct", 0.25)

    # OI 是否必需（默认 True）
    require_oi = _get_nested(settings, "liquidity_hunt", "require_oi", True)

    # OI 缺失是否允许 fallback（默认 True）
    allow_oi_missing_fallback = _get_nested(
        settings, "liquidity_hunt", "allow_oi_missing_fallback", True
    )

    # OI 缺失 fallback 时的置信度与仓位降级
    fallback_confidence = _get_nested(settings, "liquidity_hunt", "fallback_confidence", 0.65)
    fallback_core_position_mult = _get_nested(
        settings, "liquidity_hunt", "fallback_core_position_mult", 0.5
    )
    fallback_add_position_mult = _get_nested(
        settings, "liquidity_hunt", "fallback_add_position_mult", 0.5
    )

    # 如果“关键字段缺失”（recent_high/post_spike/...）是否允许策略降级触发（默认 True）
    allow_missing_fallback = _get_nested(
        settings, "liquidity_hunt", "allow_fallback_when_missing", True
    )

    # 缺字段的 fallback：比 OI 缺失更弱（默认 confidence=0.35，并且几乎不加仓）
    missing_fallback_confidence = _get_nested(
        settings, "liquidity_hunt", "missing_fallback_confidence", 0.35
    )
    missing_fallback_core_mult = _get_nested(
        settings, "liquidity_hunt", "missing_fallback_core_mult", 0.5
    )
    missing_fallback_add_mult = _get_nested(
        settings, "liquidity_hunt", "missing_fallback_add_mult", 0.0
    )

    # ---- 2) 取 timeframe 并检查字段缺失 ----
    tf = _get_tf(snap, tf_name)

    missing = _lh_missing_fields(snap, tf)
    missing_reason = None
    missing_fallback_mode = False

    # 如果缺字段：
    # - allow_missing_fallback=True：继续跑，但后续触发会更宽松，confidence/仓位更低
    # - allow_missing_fallback=False：直接 return None
    if missing:
        missing_reason = f"lh_missing={','.join(missing)}"
        if allow_missing_fallback:
            missing_fallback_mode = True
        else:
            # 这里把原因挂到 snapshot 上，方便外部 debug
            snap.lh_missing_reason = missing_reason  # type: ignore[attr-defined]
            return None

    # ---- 3) 取关键行情数据 ----
    price = tf.close

    # swing high/low：优先 recent_high/recent_low，缺失时用 high_last_n/low_last_n，再不行就用 price 兜底
    swing_high = tf.recent_high or tf.high_last_n or price
    swing_low = tf.recent_low or tf.low_last_n or price

    # ATR：用于 TP 的“回归距离”尺度
    atr = max(tf.atr, 1e-6)

    # ---- 4) 计算距离 swing 的“百分比距离” ----
    # distance_high_pct：price 相对 swing_high 的距离（百分比）
    distance_high_pct = (price - swing_high) / max(swing_high, 1e-6) * 100
    distance_low_pct = (price - swing_low) / max(swing_low, 1e-6) * 100

    # near_swing：是否在 proximity 范围内（注意：proximity_pct*100 与 distance_*_pct 同为百分比单位）
    near_swing_high = abs(distance_high_pct) <= price_proximity_pct * 100
    near_swing_low = abs(distance_low_pct) <= price_proximity_pct * 100

    # score 化的接近程度（0~1），便于方向 bias
    near_high_score = _clamp(1 - abs(distance_high_pct) / (price_proximity_pct * 100 + 1e-9)) if near_swing_high else 0.0
    near_low_score = _clamp(1 - abs(distance_low_pct) / (price_proximity_pct * 100 + 1e-9)) if near_swing_low else 0.0

    # ---- 5) 判断是否存在“大墙” ----
    # has_large_ask_wall / has_large_bid_wall：
    # - 优先使用 deriv.has_large_*_wall（如果上游已经算好了）
    # - 否则就用 wall_size 的相对倍数来推断（>=3倍）
    has_large_ask_wall = snap.deriv.has_large_ask_wall or (
        (snap.deriv.ask_wall_size or 0) >= min_wall_mult * (snap.deriv.bid_wall_size or 1e-6)
    )
    has_large_bid_wall = snap.deriv.has_large_bid_wall or (
        (snap.deriv.bid_wall_size or 0) >= min_wall_mult * (snap.deriv.ask_wall_size or 1e-6)
    )

    # ---- 6) OI 变化（可能缺失） ----
    oi_change_pct = snap.deriv.oi_change_pct
    oi_missing = oi_change_pct is None

    # 如果 OI 缺失且 require_oi=True 且不允许 fallback，则整个策略直接不触发
    if oi_missing and require_oi and not allow_oi_missing_fallback:
        return None

    # spike/flush：
    # - spike：OI 大幅上升（追突破加杠杆）
    # - flush：OI 大幅下降（平仓/爆仓出清）
    oi_spike = oi_change_pct is not None and oi_change_pct >= min_oi_spike_pct
    oi_flush = oi_change_pct is not None and oi_change_pct <= -min_oi_spike_pct

    # post-spike 小实体蜡烛：用于确认“假突破后的无力延续”
    small_candles_after_spike = (
        tf.post_spike_small_body_count is not None
        and tf.post_spike_small_body_count >= post_spike_candle_count
    )

    # fallback_mode：两类 fallback 任意成立：
    # 1) OI 缺失但允许 fallback
    # 2) 关键字段缺失但允许 missing fallback
    fallback_mode = (oi_missing and allow_oi_missing_fallback) or missing_fallback_mode

    # 方向 bias / edge_confidence：由策略自身计算
    wall_ask_score = 1.0 if has_large_ask_wall else 0.0
    wall_bid_score = 1.0 if has_large_bid_wall else 0.0
    oi_spike_score = _clamp((oi_change_pct or 0.0) / max(min_oi_spike_pct, 1e-6), 0.0, 1.0)
    oi_flush_score = _clamp(-(oi_change_pct or 0.0) / max(min_oi_spike_pct, 1e-6), 0.0, 1.0)

    short_score = _clamp(near_high_score * (0.6 * wall_ask_score + 0.4 * oi_spike_score))
    long_score = _clamp(near_low_score * (0.6 * wall_bid_score + 0.4 * oi_flush_score))
    edge_confidence = _clamp(max(long_score, short_score) * (0.8 if fallback_mode else 1.0))

    debug_scores = {
        "long": round(long_score, 4),
        "short": round(short_score, 4),
        "near_high": round(near_high_score, 4),
        "near_low": round(near_low_score, 4),
        "wall_ask": wall_ask_score,
        "wall_bid": wall_bid_score,
        "oi_spike": round(oi_spike_score, 4),
        "oi_flush": round(oi_flush_score, 4),
        "fallback_mode": 1.0 if fallback_mode else 0.0,
    }

    rejected_reasons = []
    if missing_reason:
        rejected_reasons.append(missing_reason)
    if fallback_mode and not missing_reason:
        rejected_reasons.append("lh_fallback_mode")

    # =========================
    # A) LH Short：靠近 swing high 的假突破 → 做空回归
    # =========================
    if near_swing_high and has_large_ask_wall and (oi_spike or fallback_mode) and (
        small_candles_after_spike or missing_fallback_mode
    ):
        from .signal_engine import TradeSignal

        # fake_breakout_high：用于止损参考
        # 取 tf.high_last_n / swing_high / price 的最大值，认为是“假突破的最高点”
        fake_breakout_high = max(
            v for v in [tf.high_last_n, swing_high, price] if v is not None
        )

        # SL：放在 fake_breakout_high 再上方 sl_buffer_pct（例如 +0.15%）
        sl = fake_breakout_high * (1 + sl_buffer_pct)

        # TP：以 swing_high 为中心向下回归（用 ATR 做尺度）
        tp1 = swing_high - 0.5 * atr
        tp2 = swing_high - 1.0 * atr

        # 默认置信度/仓位
        confidence = 0.75
        core_position_pct = core_pct
        add_position_pct = add_pct

        # 缺字段 fallback（更弱）：直接覆盖为更低 confidence，并显著缩仓
        if missing_fallback_mode:
            confidence = missing_fallback_confidence
            core_position_pct *= missing_fallback_core_mult
            add_position_pct *= missing_fallback_add_mult

        # OI 缺失 fallback（比缺字段弱稍强）：用 fallback_confidence 并缩仓
        elif oi_missing and allow_oi_missing_fallback:
            confidence = fallback_confidence
            core_position_pct *= fallback_core_position_mult
            add_position_pct *= fallback_add_position_mult

        # reason：解释字符串（用于 Telegram / debug）
        reason = (
            f"Liquidity hunt short near swing high {swing_high:.4f}: price={price:.4f}, "
            f"large ask wall, OI {oi_change_pct if oi_change_pct is not None else float('nan'):.1f}% "
            f"{'spike' if oi_spike else 'missing'}"
        )
        reason += f", post-spike small candles={tf.post_spike_small_body_count}"
        if fallback_mode:
            reason += " | fallback_mode=1"
        if missing_reason:
            reason += f" | {missing_reason}"

        return _attach_intent(
            TradeSignal(
                symbol=snap.symbol,
                direction="short",
                trade_confidence=confidence,
                edge_confidence=edge_confidence,
                entry=price,
                tp1=tp1,
                tp2=tp2,
                sl=sl,
                core_position_pct=core_position_pct,
                add_position_pct=add_position_pct,
                reason=reason,
                setup_type="lh_short",
                snapshot=snap,
                debug_scores=debug_scores,
                rejected_reasons=rejected_reasons or None,
            )
        )

    # =========================
    # B) LH Long：靠近 swing low 的假跌破 → 做多回归
    # =========================
    if near_swing_low and has_large_bid_wall and (oi_flush or fallback_mode) and (
        small_candles_after_spike or missing_fallback_mode
    ):
        from .signal_engine import TradeSignal

        # fake_breakout_low：用于止损参考
        fake_breakout_low = min(
            v for v in [tf.low_last_n, swing_low, price] if v is not None
        )

        # SL：放在 fake_breakout_low 再下方 sl_buffer_pct（例如 -0.15%）
        sl = fake_breakout_low * (1 - sl_buffer_pct)

        # TP：以 swing_low 为中心向上回归
        tp1 = swing_low + 0.5 * atr
        tp2 = swing_low + 1.0 * atr

        # 默认置信度/仓位
        confidence = 0.75
        core_position_pct = core_pct
        add_position_pct = add_pct

        # 缺字段 fallback（最弱）
        if missing_fallback_mode:
            confidence = missing_fallback_confidence
            core_position_pct *= missing_fallback_core_mult
            add_position_pct *= missing_fallback_add_mult

        # OI 缺失 fallback（稍强）
        elif oi_missing and allow_oi_missing_fallback:
            confidence = fallback_confidence
            core_position_pct *= fallback_core_position_mult
            add_position_pct *= fallback_add_position_mult

        reason = (
            f"Liquidity hunt long near swing low {swing_low:.4f}: price={price:.4f}, "
            f"large bid wall, OI {oi_change_pct if oi_change_pct is not None else float('nan'):.1f}% "
            f"{'flush' if oi_flush else 'missing'}"
        )
        reason += f", post-spike small candles={tf.post_spike_small_body_count}"
        if fallback_mode:
            reason += " | fallback_mode=1"
        if missing_reason:
            reason += f" | {missing_reason}"

        return _attach_intent(
            TradeSignal(
                symbol=snap.symbol,
                direction="long",
                trade_confidence=confidence,
                edge_confidence=edge_confidence,
                entry=price,
                tp1=tp1,
                tp2=tp2,
                sl=sl,
                core_position_pct=core_position_pct,
                add_position_pct=add_position_pct,
                reason=reason,
                setup_type="lh_long",
                snapshot=snap,
                debug_scores=debug_scores,
                rejected_reasons=rejected_reasons or None,
            )
        )

    # ---- 任何一边不触发 → None ----
    return None


def build_execution_intent_lh(
    snap: MarketSnapshot, signal: "TradeSignal"
) -> ExecutionIntent:
    sweep_level = signal.entry

    return ExecutionIntent(
        symbol=snap.symbol,
        direction=signal.direction,
        entry_price=sweep_level,
        entry_reason="LH_sweep_or_wall",
        invalidation_price=signal.sl,
        atr_4h=resolve_atr_4h(snap),
        reason=signal.reason,
        debug=signal.debug_scores,
    )
