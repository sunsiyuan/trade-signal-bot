"""
Trend-following strategy extracted from the signal engine.

说明：
- 这个文件实现“趋势跟随策略”的完整决策链：
  1) _score_direction：用多周期 + 流动性 + funding 等给 long/short 打分
  2) high_conf gate：在极端条件下做软加权（更敢出手 / 更保守）
  3) 触发价 trigger：不是立刻下单，而是等价格穿过一个 ATR 偏移触发价
  4) 计算 SL/TP：用 swing 均线结构 + ATR buffer 做动态止损止盈
  5) 输出 TradeSignal：要么 none（观望/等待确认），要么 long/short 的执行计划
"""
from typing import Dict, TYPE_CHECKING, Any

from .conditional_plan import resolve_atr_4h
from .models import ExecutionIntent, MarketSnapshot, Direction      # 市场快照、方向（long/short/none）
from .regime_detector import RegimeSignal          # regime 检测输出（trending / ranging 等）

# 仅用于类型检查：避免运行时循环 import
if TYPE_CHECKING:  # pragma: no cover
    from .signal_engine import TradeSignal


# -------------------------
# clamp 工具：把数值限制在一个区间
# -------------------------
def clamp(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    return max(min_value, min(max_value, value))


def _clamp01(value: float) -> float:
    """固定裁剪到 [0,1]，是归一化后的常用范围。"""
    return clamp(value, 0.0, 1.0)


def _is_price_near_ma(close: float, ma_value: float, tolerance: float) -> bool:
    """判断价格是否在给定均线的相对距离内。"""
    if ma_value <= 0:
        return False
    return abs(close - ma_value) / ma_value <= tolerance


def _near_trend_pullback(tf15, settings: Any) -> bool:
    """价格是否回踩到关键均线附近。"""
    tolerance = _get_tf_setting(settings, "high_conf_ma_proximity_pct", 0.005)
    return _is_price_near_ma(tf15.close, tf15.ma7, tolerance) or _is_price_near_ma(
        tf15.close, tf15.ma25, tolerance
    )


def _macd_hist_turning(tf15, direction: Direction, settings: Any) -> bool:
    """利用 MACD 柱子方向作为“转向”近似判定。"""
    threshold = _get_tf_setting(settings, "macd_turn_threshold", 0.0)
    if direction == "short":
        return tf15.macd_hist <= threshold or tf15.macd <= tf15.macd_signal
    return tf15.macd_hist >= -threshold or tf15.macd >= tf15.macd_signal


# -------------------------
# 读取 trend_following 策略配置（Settings 或 dict）
# -------------------------
def _get_tf_setting(settings: Any, name: str, default):
    """
    用于兼容不同的 settings 形态：
    - settings 是 dict：优先读 settings[name]，否则读 settings["trend_following"][name]
    - settings 是 Settings：优先读 settings.trend_following.name，或者 settings.name
    """
    if settings is None:
        return default

    # 1) dict 情况
    if isinstance(settings, dict):
        # 优先支持直接放在根级（方便测试/临时 override）
        if name in settings:
            return settings.get(name, default)
        # 次级：trend_following 子 dict
        tf_settings = settings.get("trend_following")
        if isinstance(tf_settings, dict):
            return tf_settings.get(name, default)

    # 2) Settings 对象情况
    if hasattr(settings, "trend_following"):
        container = getattr(settings, "trend_following")
        if isinstance(container, dict):
            return container.get(name, default)
        if hasattr(container, name):
            return getattr(container, name)

    # 3) 最后兜底：settings.name
    if hasattr(settings, name):
        return getattr(settings, name)

    return default


# -------------------------
# “软门控”：
# high_conf=True 时奖励（提高 confidence、放大仓位）
# high_conf=False 时惩罚（降低 confidence、缩小仓位）
# -------------------------
def _apply_soft_gate(
    confidence: float,
    core_pct: float,
    add_pct: float,
    high_conf: bool,
    settings: Any,
):
    """
    soft gate 的核心思想：
    - 不是硬阈值一刀切，而是对 confidence/position 做“软加权”
    - 这样可以让策略更平滑，不会因一个条件差一点就完全不动作
    """
    if high_conf:
        # 高置信度：加分、放大仓位
        confidence = _clamp01(confidence + _get_tf_setting(settings, "high_conf_bonus", 0.0))
        core_pct *= _get_tf_setting(settings, "high_conf_core_mult", 1.0)
        add_pct *= _get_tf_setting(settings, "high_conf_add_mult", 1.0)
        gate_tag = "high_conf"
    else:
        # 非高置信：扣分、缩仓位
        confidence = _clamp01(confidence - _get_tf_setting(settings, "low_conf_penalty", 0.0))
        core_pct *= _get_tf_setting(settings, "low_conf_core_mult", 1.0)
        add_pct *= _get_tf_setting(settings, "low_conf_add_mult", 1.0)
        gate_tag = "low_conf"
    return confidence, core_pct, add_pct, gate_tag


# =========================
# 方向打分器：_score_direction
# =========================
def _score_direction(snap: MarketSnapshot) -> Dict[str, float]:
    """Score long/short bias based on multi-timeframe context."""

    # 这两个分数是核心输出：越大越偏向该方向
    short_score = 0.0
    long_score = 0.0

    # debug 列表记录每条规则加了多少分，便于解释“为什么”
    debug = []

    # 取出不同周期/衍生品字段
    tf4h = snap.tf_4h
    tf1h = snap.tf_1h
    tf15 = snap.tf_15m
    deriv = snap.deriv

    # ---- 1. 大周期趋势（4h 权重大） ---- #
    # 4h 趋势决定主方向：up/down 各加 0.35
    if tf4h.trend_label == "down":
        short_score += 0.35
        debug.append(("4h_trend_down", +0.35))
    elif tf4h.trend_label == "up":
        long_score += 0.35
        debug.append(("4h_trend_up", +0.35))

    # 1h 趋势作为次级确认：up/down 各加 0.15
    if tf1h.trend_label == "down":
        short_score += 0.15
        debug.append(("1h_trend_down", +0.15))
    elif tf1h.trend_label == "up":
        long_score += 0.15
        debug.append(("1h_trend_up", +0.15))

    # ---- 2. 15m RSI 极端区域（逆向因子） ---- #
    # 15m RSI 极端高 → 更像局部过热，偏向做空（+0.25）
    # 15m RSI 极端低 → 更像局部过冷，偏向做多（+0.25）
    if tf15.rsi6 >= 80:
        short_score += 0.25
        debug.append(("15m_rsi_overbought", +0.25))
    elif tf15.rsi6 <= 20:
        long_score += 0.25
        debug.append(("15m_rsi_oversold", +0.25))

    # 温和超买/超卖：给小分（+0.1）
    if 70 <= tf15.rsi6 < 80:
        short_score += 0.1
        debug.append(("15m_rsi_high", +0.1))
    if 20 < tf15.rsi6 <= 30:
        long_score += 0.1
        debug.append(("15m_rsi_low", +0.1))

    # ---- 3. 价格相对 MA99（把 MA99 当支撑/阻力） ---- #
    # dist_to_ma99 = (close - ma99)/ma99
    # dist>0 且不远：靠近“上方阻力” → 对空更友好
    # dist<0 且不远：靠近“下方支撑” → 对多更友好
    if tf15.ma99 > 0:
        dist_to_ma99 = (tf15.close - tf15.ma99) / tf15.ma99
    else:
        dist_to_ma99 = 0.0

    if dist_to_ma99 > 0 and abs(dist_to_ma99) <= 0.02:
        short_score += 0.15
        debug.append(("price_near_resistance_ma99", +0.15))
    if dist_to_ma99 < 0 and abs(dist_to_ma99) <= 0.02:
        long_score += 0.15
        debug.append(("price_near_support_ma99", +0.15))

    # ---- 4. 流动性倾斜（订单簿） ---- #
    # deriv.liquidity_comment 可能是 "asks ..." 或 "bids ..."
    liquidity = getattr(deriv, "liquidity_comment", "") or "balanced"
    if liquidity.startswith("asks"):
        liquidity = "asks>bids"
    elif liquidity.startswith("bids"):
        liquidity = "bids>asks"

    # asks 更重：卖压更大 → 偏空 +0.1
    # bids 更重：买压更大 → 偏多 +0.1
    if liquidity == "asks>bids":
        short_score += 0.1
        debug.append(("orderbook_asks_heavier", +0.1))
    elif liquidity == "bids>asks":
        long_score += 0.1
        debug.append(("orderbook_bids_heavier", +0.1))

    # ---- 5. Funding（用百分比） ---- #
    # funding 正：多头拥挤（多头付费）→ 偏空 +0.1
    # funding 负：空头拥挤（空头付费）→ 偏多 +0.1
    funding_pct = (deriv.funding or 0.0) * 100.0
    if funding_pct > 0.02:
        short_score += 0.1
        debug.append(("funding_positive_crowded_longs", +0.1))
    elif funding_pct < -0.02:
        long_score += 0.1
        debug.append(("funding_negative_crowded_shorts", +0.1))

    # ---- 6. 波动 & 流动性异常（短周期放量波动 + 盘口倾斜） ---- #
    # atr_ratio = 15m ATR / 1h ATR
    # price_to_ma25 = 价格离 MA25 的相对偏离
    atr_ratio = tf15.atr / max(tf1h.atr, 1e-6)
    price_to_ma25 = abs(tf15.close - tf15.ma25) / max(tf15.ma25, 1e-6)

    # 认为是“短周期波动显著放大”的判定
    is_vol_spike = atr_ratio > 1.3 and price_to_ma25 > 0.007

    if is_vol_spike:
        # 放量波动 + 盘口卖压重 + 价格在均线之上 → 偏空
        if liquidity == "asks>bids" and tf15.close >= tf15.ma25:
            short_score += 0.1
            debug.append(("vol_spike_with_heavy_asks", +0.1))

        # 放量波动 + 盘口买压重 + 价格在均线之下 → 偏多
        if liquidity == "bids>asks" and tf15.close <= tf15.ma25:
            long_score += 0.1
            debug.append(("vol_spike_with_heavy_bids", +0.1))

    # 返回方向分数 + debug 列表
    return {
        "short": short_score,
        "long": long_score,
        "debug": debug,
    }


# =========================
# 高置信度判定（更苛刻）
# =========================
def _high_conf_short(snap: MarketSnapshot, settings: Any) -> bool:
    """
    high_conf_short：用于触发 soft gate 的“高置信条件”
    默认规则：
    - 4h 明确下跌，且 1h 不允许是 up（避免逆大趋势）
    - 15m 回踩到 MA25/MA7 附近 + MACD 柱子转向向下
    - RSI 过滤：使用更平滑的 RSI12 且阈值放宽到 70+
    - 可选：要求 liquidity_comment 以 "asks" 开头（卖压确实更重）
    """
    tf15 = snap.tf_15m
    rsi_ok = tf15.rsi12 >= _get_tf_setting(settings, "rsi_extreme_short", 70)
    trend_ok = snap.tf_4h.trend_label == "down" and snap.tf_1h.trend_label != "up"
    pullback_ok = _near_trend_pullback(tf15, settings)
    macd_turn_ok = _macd_hist_turning(tf15, "short", settings)

    if _get_tf_setting(settings, "require_liquidity_prefix_for_high_conf", True):
        liquidity_ok = (getattr(snap.deriv, "liquidity_comment", "") or "").lower().startswith(
            "asks"
        )
    else:
        liquidity_ok = True

    return trend_ok and pullback_ok and macd_turn_ok and rsi_ok and liquidity_ok


def _high_conf_long(snap: MarketSnapshot, settings: Any) -> bool:
    """
    high_conf_long：与 short 对称
    - 4h 上涨，且 1h 不允许是 down（避免逆大趋势）
    - 15m 回踩到 MA25/MA7 附近 + MACD 柱子转向向上
    - RSI 过滤：使用更平滑的 RSI12 且阈值放宽到 30-
    - 可选：要求 liquidity_comment 以 "bids" 开头（买压确实更重）
    """
    tf15 = snap.tf_15m
    rsi_ok = tf15.rsi12 <= _get_tf_setting(settings, "rsi_extreme_long", 30)
    trend_ok = snap.tf_4h.trend_label == "up" and snap.tf_1h.trend_label != "down"
    pullback_ok = _near_trend_pullback(tf15, settings)
    macd_turn_ok = _macd_hist_turning(tf15, "long", settings)

    if _get_tf_setting(settings, "require_liquidity_prefix_for_high_conf", True):
        liquidity_ok = (getattr(snap.deriv, "liquidity_comment", "") or "").lower().startswith(
            "bids"
        )
    else:
        liquidity_ok = True

    return trend_ok and pullback_ok and macd_turn_ok and rsi_ok and liquidity_ok


# =========================
# 触发价：用 ATR 偏移，等“确认”
# =========================
def _compute_trigger(entry: float, atr: float, direction: Direction) -> float:
    """
    trigger 逻辑：
    - short：要求价格从当前 close 往下走至少 0.3*ATR 再触发（避免提前进）
    - long：要求价格从当前 close 往上走至少 0.3*ATR 再触发
    """
    if direction == "short":
        return entry - 0.3 * atr
    return entry + 0.3 * atr


# =========================
# Swing SL：用均线结构 + ATR buffer 做动态止损
# =========================
def _compute_swing_sl(snap: MarketSnapshot, atr: float, direction: Direction) -> float:
    """
    止损的核心想法：
    - short：止损放在“最近的 swing high”（用均线代理）之上 + buffer
    - long ：止损放在“最近的 swing low”（用均线代理）之下 - buffer
    """
    tf15 = snap.tf_15m
    tf1h = snap.tf_1h

    buffer = 0.2 * atr
    if direction == "short":
        # swing_high 用多个均线的 max 作为“上方结构位”
        swing_high = max(tf15.ma7, tf15.ma25, tf1h.ma25, tf1h.ma7)
        return swing_high + buffer

    # swing_low 用多个均线的 min 作为“下方结构位”
    swing_low = min(tf15.ma7, tf15.ma25, tf1h.ma25, tf1h.ma7)
    return swing_low - buffer


# =========================
# 仓位函数：不同 regime 下对 position 更保守/更激进
# =========================
def _compute_position(confidence: float, regime: str) -> Dict[str, float]:
    """
    position mapping：
    - 在 ranging regime 下：除非 confidence >= 0.8，否则不交易（非常保守）
    - 在 trending regime 下：分段仓位（0.7/0.3, 0.4/0.2 等）
    """
    if regime in ("high_vol_ranging", "low_vol_ranging"):
        if confidence >= 0.8:
            return {"core": 0.2, "add": 0.1}
        return {"core": 0.0, "add": 0.0}

    if confidence >= 0.85:
        return {"core": 0.7, "add": 0.3}
    if 0.55 <= confidence < 0.85:
        return {"core": 0.4, "add": 0.2}
    return {"core": 0.0, "add": 0.0}


# =========================
# 主函数：输出趋势跟随信号
# =========================
def build_trend_following_signal(
    snap: MarketSnapshot,
    regime_signal: RegimeSignal,
    min_confidence: float,
    settings: Any = None,
) -> "TradeSignal":
    """
    输入：
    - snap：含多周期指标、衍生品信息、订单簿等
    - regime_signal：trending/ranging 判定
    - min_confidence：主引擎给的最小出手阈值
    - settings：用于调参、soft gate、high_conf 条件等

    输出：
    - TradeSignal(direction=none)：观望/等待触发
    - TradeSignal(direction=long/short)：给 entry/tp/sl/position 的可执行计划
    """

    def _attach_intent(ts: "TradeSignal") -> "TradeSignal":
        ts.execution_intent = build_execution_intent_tf(snap, regime_signal, ts)
        return ts

    # 1) 方向打分
    scores = _score_direction(snap)
    short_score = scores["short"]
    long_score = scores["long"]

    # 2) 选择 bias（偏向哪个方向）
    bias: Direction = "short" if short_score > long_score else "long"

    # base_confidence：简单取 max(long, short)
    base_confidence = max(short_score, long_score)

    # trend_bias：long/short 差距，越大说明方向越“清晰”
    trend_bias = abs(long_score - short_score)
    trend_bias_conf = clamp(trend_bias / 0.3, 0.0, 1.0)

    # thresholds_snapshot：用于 debug 输出
    thresholds = {
        "min_confidence": min_confidence,
        "regime_trade_min": 0.8,   # 非 trending 时的出手门槛（见后面逻辑）
        "trigger_atr_mult": 0.3,
    }

    regime = regime_signal.regime

    # 3) 判断是否 high_conf（用于 soft gate）
    high_conf = (
        _high_conf_short(snap, settings)
        if bias == "short"
        else _high_conf_long(snap, settings)
    )

    # 4) 根据 base_confidence + regime 决定仓位
    position = _compute_position(base_confidence, regime)

    # 5) 应用 soft gate（会改变 confidence 和仓位）
    confidence, core_pct, add_pct, gate_tag = _apply_soft_gate(
        base_confidence, position["core"], position["add"], high_conf, settings
    )
    thresholds.update({"gate": gate_tag, "high_conf": high_conf})

    # -------------------------
    # A) 硬门槛 1：低于 min_confidence 直接不出手
    # -------------------------
    if confidence < min_confidence:
        from .signal_engine import TradeSignal

        trade_conf = round(confidence, 2)
        signal = TradeSignal(
            symbol=snap.symbol,
            direction="none",
            trade_confidence=trade_conf,
            edge_confidence=round(trend_bias_conf, 2),
            setup_type="none",
            reason=(
                f"信心 {confidence:.2f} 低于最小阈值，gate={gate_tag}"
                f"（long={long_score:.2f}, short={short_score:.2f}）"
            ),
            snapshot=snap,
            debug_scores={
                **scores,
                "trend_bias_conf": round(trend_bias_conf, 4),
                "high_conf": high_conf,
                "gate_tag": gate_tag,
                "base_confidence": round(base_confidence, 4),
            },
            rejected_reasons=["confidence_below_min"],
            thresholds_snapshot=thresholds,
        )
        signal.edge_type = "趋势清晰"
        return _attach_intent(signal)

    # -------------------------
    # B) 硬门槛 2：如果 regime 不是 trending，则要求 confidence >= 0.8 才允许交易
    # （等价于：在 ranging 下趋势策略极度保守）
    # -------------------------
    if regime != "trending" and confidence < 0.8:
        from .signal_engine import TradeSignal

        trade_conf = round(confidence, 2)
        signal = TradeSignal(
            symbol=snap.symbol,
            direction="none",
            trade_confidence=trade_conf,
            edge_confidence=round(trend_bias_conf, 2),
            setup_type="none",
            reason=(
                f"行情模式为 {regime}，经软加权后信号 {confidence:.2f} 不足以出手"
            ),
            snapshot=snap,
            debug_scores={
                **scores,
                "trend_bias_conf": round(trend_bias_conf, 4),
                "high_conf": high_conf,
                "gate_tag": gate_tag,
                "base_confidence": round(base_confidence, 4),
            },
            rejected_reasons=["regime_not_trending", "confidence_below_regime_threshold"],
            thresholds_snapshot=thresholds,
        )
        signal.edge_type = "趋势清晰"
        return _attach_intent(signal)

    # 6) 计算触发价 trigger（用 15m ATR）
    tf15 = snap.tf_15m
    atr = max(tf15.atr, 1e-6)
    spot_price = tf15.close
    trigger = _compute_trigger(spot_price, atr, bias)

    # 7) 如果价格还没触发，则输出 “等待确认价” 的 none 信号（很关键）
    price_crossed = (spot_price <= trigger) if bias == "short" else (spot_price >= trigger)
    if not price_crossed:
        from .signal_engine import TradeSignal

        trade_conf = round(confidence, 2)
        signal = TradeSignal(
            symbol=snap.symbol,
            direction="none",
            trade_confidence=trade_conf,
            edge_confidence=round(trend_bias_conf, 2),
            setup_type="none",
            reason=(f"信号待确认：等待价格触发 {trigger:.2f} 以执行 {bias} 入场"),
            entry=trigger,  # 注意：这里的 entry 实际是 trigger（等待价）
            snapshot=snap,
            debug_scores={**scores, "trend_bias_conf": round(trend_bias_conf, 4)},
            rejected_reasons=["price_not_triggered"],
            thresholds_snapshot=thresholds,
        )
        signal.edge_type = "趋势清晰"
        return _attach_intent(signal)

    # 8) 已触发：进入“下单计划”构建
    entry = trigger
    sl = _compute_swing_sl(snap, atr, bias)

    # R = 风险距离（entry 到 SL 的距离）
    R = abs(sl - entry)

    # 如果 R 太小（止损太近），至少给 0.25 ATR（避免被噪声扫掉）
    if R < atr * 0.25:
        R = atr * 0.25

    # 9) TP：按 0.5R / 1R / 2R 三段止盈
    if bias == "short":
        tp1 = entry - 0.5 * R
        tp2 = entry - 1.0 * R
        tp3 = entry - 2.0 * R
    else:
        tp1 = entry + 0.5 * R
        tp2 = entry + 1.0 * R
        tp3 = entry + 2.0 * R

    # 10) 输出最终 TradeSignal（可执行）
    from .signal_engine import TradeSignal

    trade_conf = round(confidence, 2)

    signal = TradeSignal(
        symbol=snap.symbol,
        direction=bias,
        trade_confidence=trade_conf,
        setup_type="trend_short" if bias == "short" else "trend_long",
        reason=(
            f"{regime} 模式下触发确认价，按 R 结构下单，TP/SL 动态；"
            f"score long={long_score:.2f} / short={short_score:.2f}"
        ),
        entry=entry,
        tp1=tp1,
        tp2=tp2,
        tp3=tp3,
        sl=sl,
        core_position_pct=core_pct,
        add_position_pct=add_pct,
        snapshot=snap,
        debug_scores={
            **scores,
            "regime": regime,
            "trigger": trigger,
            "R": R,
            "high_conf": high_conf,
            "gate_tag": gate_tag,
            "base_confidence": round(base_confidence, 4),
        },
        rejected_reasons=[],
        thresholds_snapshot=thresholds,
    )
    signal.edge_type = "趋势清晰"
    return _attach_intent(signal)


def build_execution_intent_tf(
    snap: MarketSnapshot,
    regime_signal: RegimeSignal,
    signal: "TradeSignal",
) -> ExecutionIntent:
    trigger = signal.entry

    return ExecutionIntent(
        symbol=snap.symbol,
        direction=signal.direction,
        entry_price=trigger,
        entry_reason="TF_trigger",
        invalidation_price=signal.sl,
        atr_4h=resolve_atr_4h(snap),
        reason=signal.reason,
        debug=signal.debug_scores,
    )
