from typing import Optional, TYPE_CHECKING

from .models import MarketSnapshot
from .regime_detector import Regime

if TYPE_CHECKING:  # pragma: no cover
    from .signal_engine import TradeSignal


def _get_tf(snap: MarketSnapshot, tf: str):
    return snap.get_timeframe(tf)


def _get_nested(settings, group: str, key: str, default):
    if hasattr(settings, group):
        container = getattr(settings, group)
        if isinstance(container, dict):
            return container.get(key, default)
        if hasattr(container, key):
            return getattr(container, key)
    if isinstance(settings, dict) and group in settings:
        return settings[group].get(key, default)
    return default


LH_REQUIRED = {
    "tf.recent_high",
    "tf.recent_low",
    "tf.post_spike_small_body_count",
    "deriv.ask_wall_size",
    "deriv.bid_wall_size",
    "deriv.has_large_ask_wall",
    "deriv.has_large_bid_wall",
}


def _lh_missing_fields(snap: MarketSnapshot, tf) -> list:
    missing = []
    if tf.recent_high is None or tf.recent_low is None:
        missing.append("tf.recent_high/recent_low")
    if tf.post_spike_small_body_count is None:
        missing.append("tf.post_spike_small_body_count")

    deriv = getattr(snap, "deriv", None)
    if deriv is None:
        missing.append("snap.deriv")
        return missing

    if not hasattr(deriv, "ask_wall_size") or not hasattr(deriv, "bid_wall_size"):
        missing.append("deriv.ask_wall_size/bid_wall_size")
    if not hasattr(deriv, "has_large_ask_wall") or not hasattr(
        deriv, "has_large_bid_wall"
    ):
        missing.append("deriv.has_large_*_wall")
    return missing


def build_liquidity_hunt_signal(
    snap: MarketSnapshot, regime: Regime, settings
) -> Optional["TradeSignal"]:
    """Look for liquidity hunt setups around swing highs/lows in high vol ranges."""

    if regime != "high_vol_ranging":
        return None

    tf_name = _get_nested(settings, "liquidity_hunt", "tf", "1h")
    price_proximity_pct = _get_nested(settings, "liquidity_hunt", "price_proximity_pct", 0.004)
    min_wall_mult = _get_nested(settings, "liquidity_hunt", "min_wall_mult", 3.0)
    min_oi_spike_pct = _get_nested(settings, "liquidity_hunt", "min_oi_spike_pct", 5.0)
    post_spike_candle_count = _get_nested(settings, "liquidity_hunt", "post_spike_candle_count", 3)
    sl_buffer_pct = _get_nested(settings, "liquidity_hunt", "sl_buffer_pct", 0.0015)
    core_pct = _get_nested(settings, "liquidity_hunt", "core_position_pct", 0.5)
    add_pct = _get_nested(settings, "liquidity_hunt", "add_position_pct", 0.25)
    require_oi = _get_nested(settings, "liquidity_hunt", "require_oi", True)
    allow_oi_missing_fallback = _get_nested(
        settings, "liquidity_hunt", "allow_oi_missing_fallback", True
    )
    fallback_confidence = _get_nested(settings, "liquidity_hunt", "fallback_confidence", 0.65)
    fallback_core_position_mult = _get_nested(
        settings, "liquidity_hunt", "fallback_core_position_mult", 0.5
    )
    fallback_add_position_mult = _get_nested(
        settings, "liquidity_hunt", "fallback_add_position_mult", 0.5
    )
    allow_missing_fallback = _get_nested(
        settings, "liquidity_hunt", "allow_fallback_when_missing", True
    )
    missing_fallback_confidence = _get_nested(
        settings, "liquidity_hunt", "missing_fallback_confidence", 0.35
    )
    missing_fallback_core_mult = _get_nested(
        settings, "liquidity_hunt", "missing_fallback_core_mult", 0.5
    )
    missing_fallback_add_mult = _get_nested(
        settings, "liquidity_hunt", "missing_fallback_add_mult", 0.0
    )

    tf = _get_tf(snap, tf_name)
    missing = _lh_missing_fields(snap, tf)
    missing_reason = None
    missing_fallback_mode = False
    if missing:
        missing_reason = f"lh_missing={','.join(missing)}"
        if allow_missing_fallback:
            missing_fallback_mode = True
        else:
            snap.lh_missing_reason = missing_reason  # type: ignore[attr-defined]
            return None
    price = tf.close
    swing_high = tf.recent_high or tf.high_last_n or price
    swing_low = tf.recent_low or tf.low_last_n or price
    atr = max(tf.atr, 1e-6)

    distance_high_pct = (price - swing_high) / max(swing_high, 1e-6) * 100
    distance_low_pct = (price - swing_low) / max(swing_low, 1e-6) * 100

    near_swing_high = abs(distance_high_pct) <= price_proximity_pct * 100
    near_swing_low = abs(distance_low_pct) <= price_proximity_pct * 100

    has_large_ask_wall = snap.deriv.has_large_ask_wall or (
        (snap.deriv.ask_wall_size or 0) >= min_wall_mult * (snap.deriv.bid_wall_size or 1e-6)
    )
    has_large_bid_wall = snap.deriv.has_large_bid_wall or (
        (snap.deriv.bid_wall_size or 0) >= min_wall_mult * (snap.deriv.ask_wall_size or 1e-6)
    )

    oi_change_pct = snap.deriv.oi_change_pct
    oi_missing = oi_change_pct is None

    if oi_missing and require_oi and not allow_oi_missing_fallback:
        return None

    oi_spike = oi_change_pct is not None and oi_change_pct >= min_oi_spike_pct
    oi_flush = oi_change_pct is not None and oi_change_pct <= -min_oi_spike_pct
    small_candles_after_spike = (
        tf.post_spike_small_body_count is not None
        and tf.post_spike_small_body_count >= post_spike_candle_count
    )

    fallback_mode = (oi_missing and allow_oi_missing_fallback) or missing_fallback_mode

    if near_swing_high and has_large_ask_wall and (oi_spike or fallback_mode) and (
        small_candles_after_spike or missing_fallback_mode
    ):
        from .signal_engine import TradeSignal
        fake_breakout_high = max(
            v for v in [tf.high_last_n, swing_high, price] if v is not None
        )
        sl = fake_breakout_high * (1 + sl_buffer_pct)
        tp1 = swing_high - 0.5 * atr
        tp2 = swing_high - 1.0 * atr
        confidence = 0.75
        core_position_pct = core_pct
        add_position_pct = add_pct

        if missing_fallback_mode:
            confidence = missing_fallback_confidence
            core_position_pct *= missing_fallback_core_mult
            add_position_pct *= missing_fallback_add_mult
        elif oi_missing and allow_oi_missing_fallback:
            confidence = fallback_confidence
            core_position_pct *= fallback_core_position_mult
            add_position_pct *= fallback_add_position_mult

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

        return TradeSignal(
            symbol=snap.symbol,
            direction="short",
            confidence=confidence,
            entry=price,
            tp1=tp1,
            tp2=tp2,
            sl=sl,
            core_position_pct=core_position_pct,
            add_position_pct=add_position_pct,
            reason=reason,
            snapshot=snap,
        )

    if near_swing_low and has_large_bid_wall and (oi_flush or fallback_mode) and (
        small_candles_after_spike or missing_fallback_mode
    ):
        from .signal_engine import TradeSignal
        fake_breakout_low = min(
            v for v in [tf.low_last_n, swing_low, price] if v is not None
        )
        sl = fake_breakout_low * (1 - sl_buffer_pct)
        tp1 = swing_low + 0.5 * atr
        tp2 = swing_low + 1.0 * atr
        confidence = 0.75
        core_position_pct = core_pct
        add_position_pct = add_pct

        if missing_fallback_mode:
            confidence = missing_fallback_confidence
            core_position_pct *= missing_fallback_core_mult
            add_position_pct *= missing_fallback_add_mult
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

        return TradeSignal(
            symbol=snap.symbol,
            direction="long",
            confidence=confidence,
            entry=price,
            tp1=tp1,
            tp2=tp2,
            sl=sl,
            core_position_pct=core_position_pct,
            add_position_pct=add_position_pct,
            reason=reason,
            snapshot=snap,
        )

    return None
