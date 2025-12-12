from typing import Optional, TYPE_CHECKING

from .models import MarketSnapshot, Direction
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


def build_mean_reversion_signal(
    snap: MarketSnapshot, regime: Regime, settings
) -> Optional["TradeSignal"]:
    """Generate a mean-reversion signal when the market is ranging."""

    if regime not in {"high_vol_ranging", "low_vol_ranging"}:
        return None

    tf_name = _get_nested(settings, "mean_reversion", "tf", "1h")
    rsi_oversold = _get_nested(settings, "mean_reversion", "rsi_oversold", 12)
    rsi_overbought = _get_nested(settings, "mean_reversion", "rsi_overbought", 88)
    atr_dev_mult = _get_nested(settings, "mean_reversion", "atr_dev_mult", 1.2)
    min_oi_change_pct = _get_nested(settings, "mean_reversion", "min_oi_change_pct", 3.0)
    tp_to_sl_ratio = _get_nested(settings, "mean_reversion", "tp_to_sl_ratio", 1.5)
    core_pct = _get_nested(settings, "mean_reversion", "core_position_pct", 0.5)
    add_pct = _get_nested(settings, "mean_reversion", "add_position_pct", 0.25)
    sl_buffer_mult = _get_nested(settings, "mean_reversion", "sl_buffer_mult", 0.8)
    require_oi = _get_nested(settings, "mean_reversion", "require_oi", True)
    allow_oi_missing_fallback = _get_nested(
        settings, "mean_reversion", "allow_oi_missing_fallback", True
    )
    fallback_confidence_mult = _get_nested(
        settings, "mean_reversion", "fallback_confidence_mult", 0.75
    )
    fallback_core_position_mult = _get_nested(
        settings, "mean_reversion", "fallback_core_position_mult", 0.5
    )
    fallback_add_position_mult = _get_nested(
        settings, "mean_reversion", "fallback_add_position_mult", 0.0
    )

    tf = _get_tf(snap, tf_name)
    price = tf.close
    ma25 = tf.ma25
    atr = max(tf.atr, 1e-6)
    rsi6 = tf.rsi6
    oi_change_pct = snap.deriv.oi_change_pct
    oi_missing = oi_change_pct is None

    if oi_missing and require_oi and not allow_oi_missing_fallback:
        return None

    cond_far_below_ma = price <= ma25 - atr_dev_mult * atr
    cond_oversold = rsi6 <= rsi_oversold
    cond_oi_flushing_out = (
        oi_change_pct is not None and oi_change_pct <= -min_oi_change_pct
    )

    fallback_mode = oi_missing and allow_oi_missing_fallback

    if cond_far_below_ma and cond_oversold and (cond_oi_flushing_out or fallback_mode):
        from .signal_engine import TradeSignal
        sl = price - sl_buffer_mult * atr
        tp1 = ma25
        tp2 = ma25 + 0.5 * atr
        rr = abs(tp1 - price) / max(abs(price - sl), 1e-6)
        confidence = min(0.9, 0.6 + 0.1 * rr)
        if fallback_mode:
            confidence *= fallback_confidence_mult
            core_pct *= fallback_core_position_mult
            add_pct *= fallback_add_position_mult
        reason = (
            f"Mean reversion long: price {price:.4f} < MA25 {ma25:.4f} - {atr_dev_mult} ATR, "
            f"RSI6={rsi6:.1f} oversold, OI_change={oi_change_pct if oi_change_pct is not None else float('nan'):.1f}% "
            f"{'flushing out' if cond_oi_flushing_out else 'missing'}"
        )
        if fallback_mode:
            reason += " | OI missing → fallback mode"

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

    cond_far_above_ma = price >= ma25 + atr_dev_mult * atr
    cond_overbought = rsi6 >= rsi_overbought
    cond_oi_squeezing = (
        oi_change_pct is not None and oi_change_pct >= min_oi_change_pct
    )

    if cond_far_above_ma and cond_overbought and (cond_oi_squeezing or fallback_mode):
        from .signal_engine import TradeSignal
        sl = price + sl_buffer_mult * atr
        tp1 = ma25
        tp2 = ma25 - 0.5 * atr
        rr = abs(tp1 - price) / max(abs(sl - price), 1e-6)
        confidence = min(0.9, 0.6 + 0.1 * rr)
        if fallback_mode:
            confidence *= fallback_confidence_mult
            core_pct *= fallback_core_position_mult
            add_pct *= fallback_add_position_mult
        reason = (
            f"Mean reversion short: price {price:.4f} > MA25 {ma25:.4f} + {atr_dev_mult} ATR, "
            f"RSI6={rsi6:.1f} overbought, OI_change={oi_change_pct if oi_change_pct is not None else float('nan'):.1f}% "
            f"{'squeezing' if cond_oi_squeezing else 'missing'}"
        )
        if fallback_mode:
            reason += " | OI missing → fallback mode"

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

    return None
