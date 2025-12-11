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

    tf = _get_tf(snap, tf_name)
    price = tf.close
    swing_high = tf.recent_high or (tf.high_last_n and max(tf.high_last_n)) or price
    swing_low = tf.recent_low or (tf.low_last_n and min(tf.low_last_n)) or price
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

    oi_spike = (snap.deriv.oi_change_pct or 0.0) >= min_oi_spike_pct
    oi_flush = (snap.deriv.oi_change_pct or 0.0) <= -min_oi_spike_pct
    small_candles_after_spike = tf.post_spike_small_body_count >= post_spike_candle_count

    if near_swing_high and has_large_ask_wall and oi_spike and small_candles_after_spike:
        from .signal_engine import TradeSignal
        fake_breakout_high = max(tf.high_last_n or [swing_high, price])
        sl = fake_breakout_high * (1 + sl_buffer_pct)
        tp1 = swing_high - 0.5 * atr
        tp2 = swing_high - 1.0 * atr
        confidence = 0.75
        reason = (
            f"Liquidity hunt short near swing high {swing_high:.4f}: price={price:.4f}, "
            f"large ask wall, OI spike {snap.deriv.oi_change_pct or 0.0:.1f}%, "
            f"post-spike small candles={tf.post_spike_small_body_count}"
        )

        return TradeSignal(
            symbol=snap.symbol,
            direction="short",
            confidence=confidence,
            entry=price,
            tp1=tp1,
            tp2=tp2,
            sl=sl,
            core_position_pct=core_pct,
            add_position_pct=add_pct,
            reason=reason,
            snapshot=snap,
        )

    if near_swing_low and has_large_bid_wall and oi_flush and small_candles_after_spike:
        from .signal_engine import TradeSignal
        fake_breakout_low = min(tf.low_last_n or [swing_low, price])
        sl = fake_breakout_low * (1 - sl_buffer_pct)
        tp1 = swing_low + 0.5 * atr
        tp2 = swing_low + 1.0 * atr
        confidence = 0.75
        reason = (
            f"Liquidity hunt long near swing low {swing_low:.4f}: price={price:.4f}, "
            f"large bid wall, OI flush {snap.deriv.oi_change_pct or 0.0:.1f}%, "
            f"post-spike small candles={tf.post_spike_small_body_count}"
        )

        return TradeSignal(
            symbol=snap.symbol,
            direction="long",
            confidence=confidence,
            entry=price,
            tp1=tp1,
            tp2=tp2,
            sl=sl,
            core_position_pct=core_pct,
            add_position_pct=add_pct,
            reason=reason,
            snapshot=snap,
        )

    return None
