"""Trend-following strategy extracted from the signal engine."""
from typing import Dict, TYPE_CHECKING, Any

from .models import MarketSnapshot, Direction
from .regime_detector import RegimeSignal

if TYPE_CHECKING:  # pragma: no cover
    from .signal_engine import TradeSignal


def clamp(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    return max(min_value, min(max_value, value))


def _clamp01(value: float) -> float:
    return clamp(value, 0.0, 1.0)


def _get_tf_setting(settings: Any, name: str, default):
    if settings is None:
        return default

    if isinstance(settings, dict):
        if name in settings:
            return settings.get(name, default)
        tf_settings = settings.get("trend_following")
        if isinstance(tf_settings, dict):
            return tf_settings.get(name, default)

    if hasattr(settings, "trend_following"):
        container = getattr(settings, "trend_following")
        if isinstance(container, dict):
            return container.get(name, default)
        if hasattr(container, name):
            return getattr(container, name)

    if hasattr(settings, name):
        return getattr(settings, name)

    return default


def _apply_soft_gate(
    confidence: float,
    core_pct: float,
    add_pct: float,
    high_conf: bool,
    settings: Any,
):
    if high_conf:
        confidence = _clamp01(confidence + _get_tf_setting(settings, "high_conf_bonus", 0.0))
        core_pct *= _get_tf_setting(settings, "high_conf_core_mult", 1.0)
        add_pct *= _get_tf_setting(settings, "high_conf_add_mult", 1.0)
        gate_tag = "high_conf"
    else:
        confidence = _clamp01(confidence - _get_tf_setting(settings, "low_conf_penalty", 0.0))
        core_pct *= _get_tf_setting(settings, "low_conf_core_mult", 1.0)
        add_pct *= _get_tf_setting(settings, "low_conf_add_mult", 1.0)
        gate_tag = "low_conf"
    return confidence, core_pct, add_pct, gate_tag


def _score_direction(snap: MarketSnapshot) -> Dict[str, float]:
    """Score long/short bias based on multi-timeframe context."""
    short_score = 0.0
    long_score = 0.0
    debug = []

    tf4h = snap.tf_4h
    tf1h = snap.tf_1h
    tf15 = snap.tf_15m
    deriv = snap.deriv

    # ---- 1. 大周期趋势 ---- #
    if tf4h.trend_label == "down":
        short_score += 0.35
        debug.append(("4h_trend_down", +0.35))
    elif tf4h.trend_label == "up":
        long_score += 0.35
        debug.append(("4h_trend_up", +0.35))

    # 1h 趋势作为次级确认
    if tf1h.trend_label == "down":
        short_score += 0.15
        debug.append(("1h_trend_down", +0.15))
    elif tf1h.trend_label == "up":
        long_score += 0.15
        debug.append(("1h_trend_up", +0.15))

    # ---- 2. 15m RSI 极端区域 ---- #
    if tf15.rsi6 >= 80:
        short_score += 0.25
        debug.append(("15m_rsi_overbought", +0.25))
    elif tf15.rsi6 <= 20:
        long_score += 0.25
        debug.append(("15m_rsi_oversold", +0.25))

    # 温和超买/超卖，给一点小分
    if 70 <= tf15.rsi6 < 80:
        short_score += 0.1
        debug.append(("15m_rsi_high", +0.1))
    if 20 < tf15.rsi6 <= 30:
        long_score += 0.1
        debug.append(("15m_rsi_low", +0.1))

    # ---- 3. 价格相对 MA99（视作阻力/支撑） ---- #
    if tf15.ma99 > 0:
        dist_to_ma99 = (tf15.close - tf15.ma99) / tf15.ma99
    else:
        dist_to_ma99 = 0.0

    # 价格在 MA99 上方且偏离不大 → 空头好位置
    if dist_to_ma99 > 0 and abs(dist_to_ma99) <= 0.02:
        short_score += 0.15
        debug.append(("price_near_resistance_ma99", +0.15))
    # 价格在 MA99 下方不远 → 多头好位置
    if dist_to_ma99 < 0 and abs(dist_to_ma99) <= 0.02:
        long_score += 0.15
        debug.append(("price_near_support_ma99", +0.15))

    # ---- 4. 流动性倾斜 ---- #
    liquidity = getattr(deriv, "liquidity_comment", "") or "balanced"
    if liquidity.startswith("asks"):
        liquidity = "asks>bids"
    elif liquidity.startswith("bids"):
        liquidity = "bids>asks"

    if liquidity == "asks>bids":
        short_score += 0.1
        debug.append(("orderbook_asks_heavier", +0.1))
    elif liquidity == "bids>asks":
        long_score += 0.1
        debug.append(("orderbook_bids_heavier", +0.1))

    # ---- 5. Funding（用百分比） ---- #
    funding_pct = (deriv.funding or 0.0) * 100.0

    # 正 funding 说明多头付费，多头 crowded，适度给空头加分
    if funding_pct > 0.02:
        short_score += 0.1
        debug.append(("funding_positive_crowded_longs", +0.1))
    elif funding_pct < -0.02:
        long_score += 0.1
        debug.append(("funding_negative_crowded_shorts", +0.1))

    # ---- 6. 波动 & 流动性异常 ---- #
    # 利用 15m vs 1h ATR 比例 + 价格偏离 + orderbook 倾斜
    atr_ratio = tf15.atr / max(tf1h.atr, 1e-6)
    price_to_ma25 = abs(tf15.close - tf15.ma25) / max(tf15.ma25, 1e-6)

    # 认为是“短周期明显放量波动”的阈值
    is_vol_spike = atr_ratio > 1.3 and price_to_ma25 > 0.007

    if is_vol_spike:
        # 如果盘口卖单明显更重，且价格在均线之上 → 偏空
        if liquidity == "asks>bids" and tf15.close >= tf15.ma25:
            short_score += 0.1
            debug.append(("vol_spike_with_heavy_asks", +0.1))

        # 如果盘口买单明显更重，且价格在均线之下 → 偏多
        if liquidity == "bids>asks" and tf15.close <= tf15.ma25:
            long_score += 0.1
            debug.append(("vol_spike_with_heavy_bids", +0.1))

    return {
        "short": short_score,
        "long": long_score,
        "debug": debug,
    }


def _high_conf_short(snap: MarketSnapshot, settings: Any) -> bool:
    tf15 = snap.tf_15m
    rsi_ok = tf15.rsi6 >= _get_tf_setting(settings, "rsi_extreme_short", 80)
    trend_ok = snap.tf_4h.trend_label == "down" and snap.tf_1h.trend_label != "up"

    if _get_tf_setting(settings, "require_liquidity_prefix_for_high_conf", True):
        liquidity_ok = (getattr(snap.deriv, "liquidity_comment", "") or "").lower().startswith(
            "asks"
        )
    else:
        liquidity_ok = True

    return trend_ok and rsi_ok and liquidity_ok


def _high_conf_long(snap: MarketSnapshot, settings: Any) -> bool:
    tf15 = snap.tf_15m
    rsi_ok = tf15.rsi6 <= _get_tf_setting(settings, "rsi_extreme_long", 20)
    trend_ok = snap.tf_4h.trend_label == "up" and snap.tf_1h.trend_label != "down"

    if _get_tf_setting(settings, "require_liquidity_prefix_for_high_conf", True):
        liquidity_ok = (getattr(snap.deriv, "liquidity_comment", "") or "").lower().startswith(
            "bids"
        )
    else:
        liquidity_ok = True

    return trend_ok and rsi_ok and liquidity_ok


def _compute_trigger(entry: float, atr: float, direction: Direction) -> float:
    if direction == "short":
        return entry - 0.3 * atr
    return entry + 0.3 * atr


def _compute_swing_sl(snap: MarketSnapshot, atr: float, direction: Direction) -> float:
    tf15 = snap.tf_15m
    tf1h = snap.tf_1h

    buffer = 0.2 * atr
    if direction == "short":
        swing_high = max(tf15.ma7, tf15.ma25, tf1h.ma25, tf1h.ma7)
        return swing_high + buffer

    swing_low = min(tf15.ma7, tf15.ma25, tf1h.ma25, tf1h.ma7)
    return swing_low - buffer


def _compute_position(confidence: float, regime: str) -> Dict[str, float]:
    if regime in ("high_vol_ranging", "low_vol_ranging"):
        if confidence >= 0.8:
            return {"core": 0.2, "add": 0.1}
        return {"core": 0.0, "add": 0.0}

    if confidence >= 0.85:
        return {"core": 0.7, "add": 0.3}
    if 0.55 <= confidence < 0.85:
        return {"core": 0.4, "add": 0.2}
    return {"core": 0.0, "add": 0.0}


def build_trend_following_signal(
    snap: MarketSnapshot,
    regime_signal: RegimeSignal,
    min_confidence: float,
    settings: Any = None,
) -> "TradeSignal":
    scores = _score_direction(snap)
    short_score = scores["short"]
    long_score = scores["long"]

    bias: Direction = "short" if short_score > long_score else "long"
    base_confidence = max(short_score, long_score)
    trend_bias = abs(long_score - short_score)
    trend_bias_conf = clamp(trend_bias / 0.3, 0.0, 1.0)
    thresholds = {
        "min_confidence": min_confidence,
        "regime_trade_min": 0.8,
        "trigger_atr_mult": 0.3,
    }
    regime = regime_signal.regime
    snap.market_mode = regime

    high_conf = (
        _high_conf_short(snap, settings)
        if bias == "short"
        else _high_conf_long(snap, settings)
    )

    position = _compute_position(base_confidence, regime)
    confidence, core_pct, add_pct, gate_tag = _apply_soft_gate(
        base_confidence, position["core"], position["add"], high_conf, settings
    )
    thresholds.update({"gate": gate_tag, "high_conf": high_conf})

    if confidence < min_confidence:
        from .signal_engine import TradeSignal

        return TradeSignal(
            symbol=snap.symbol,
            direction="none",
            confidence=round(confidence, 2),
            trade_confidence=round(confidence, 2),
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

    if regime != "trending" and confidence < 0.8:
        from .signal_engine import TradeSignal

        return TradeSignal(
            symbol=snap.symbol,
            direction="none",
            confidence=round(confidence, 2),
            trade_confidence=round(confidence, 2),
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

    tf15 = snap.tf_15m
    atr = max(tf15.atr, 1e-6)
    spot_price = tf15.close
    trigger = _compute_trigger(spot_price, atr, bias)

    price_crossed = (spot_price <= trigger) if bias == "short" else (spot_price >= trigger)
    if not price_crossed:
        from .signal_engine import TradeSignal

        return TradeSignal(
            symbol=snap.symbol,
            direction="none",
            confidence=round(confidence, 2),
            trade_confidence=round(confidence, 2),
            edge_confidence=round(trend_bias_conf, 2),
            setup_type="none",
            reason=(f"信号待确认：等待价格触发 {trigger:.2f} 以执行 {bias} 入场"),
            entry=trigger,
            snapshot=snap,
            debug_scores={**scores, "trend_bias_conf": round(trend_bias_conf, 4)},
            rejected_reasons=["price_not_triggered"],
            thresholds_snapshot=thresholds,
        )

    entry = trigger
    sl = _compute_swing_sl(snap, atr, bias)
    R = abs(sl - entry)
    if R < atr * 0.25:
        R = atr * 0.25

    if bias == "short":
        tp1 = entry - 0.5 * R
        tp2 = entry - 1.0 * R
        tp3 = entry - 2.0 * R
    else:
        tp1 = entry + 0.5 * R
        tp2 = entry + 1.0 * R
        tp3 = entry + 2.0 * R

    from .signal_engine import TradeSignal

    return TradeSignal(
        symbol=snap.symbol,
        direction=bias,
        confidence=round(confidence, 2),
        trade_confidence=round(confidence, 2),
        setup_type="trend_short" if bias == "short" else "trend_long",
        reason=(
            f"{regime} 模式下触发确认价，按 R 结构下单，TP/SL 动态；"
            f"score long={long_score:.2f} / short={short_score:.2f}"
        ),
        entry=entry,
        entry_range=[trigger],
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
