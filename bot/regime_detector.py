from dataclasses import dataclass
from typing import Literal, Optional

from .models import MarketSnapshot


Regime = Literal["trending", "high_vol_ranging", "low_vol_ranging", "unknown"]


@dataclass
class RegimeSignal:
    regime: Regime
    reason: str


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


def detect_regime(snap: MarketSnapshot, settings) -> RegimeSignal:
    """Decide the current market regime using MA slope, ATR and RSI behavior."""

    main_tf = _get_nested(settings, "regime", "main_tf", "1h")
    trend_ma_angle_min = _get_nested(settings, "regime", "trend_ma_angle_min", 0.0015)
    high_vol_atr_rel = _get_nested(settings, "regime", "high_vol_atr_rel", 0.015)
    low_vol_atr_rel = _get_nested(settings, "regime", "low_vol_atr_rel", 0.006)
    rsi_band = _get_nested(settings, "regime", "ranging_rsi_band", 12)
    slope_lookback = _get_nested(settings, "regime", "slope_lookback", 5)

    tf = snap.get_timeframe(main_tf)
    ma_history = tf.ma25_history
    rsi_history = tf.rsi6_history

    if ma_history and len(ma_history) > slope_lookback:
        ma_angle = (tf.ma25 - ma_history[-slope_lookback]) / max(
            slope_lookback * max(tf.close, 1e-6), 1e-6
        )
    else:
        ma_angle = 0.0

    atr_rel = tf.atr / max(tf.close, 1e-6)

    if rsi_history:
        deviations = [abs(v - 50) for v in rsi_history[-slope_lookback:]]
        rsi_avg_dev = sum(deviations) / max(len(deviations), 1)
        osc_count = sum(1 for i in range(1, len(rsi_history)) if (rsi_history[i - 1] - 50) * (rsi_history[i] - 50) < 0)
    else:
        rsi_avg_dev = abs(tf.rsi6 - 50)
        osc_count = 0

    if abs(ma_angle) >= trend_ma_angle_min:
        regime: Regime = "trending"
    else:
        if atr_rel >= high_vol_atr_rel:
            regime = "high_vol_ranging"
        elif atr_rel <= low_vol_atr_rel:
            regime = "low_vol_ranging"
        else:
            regime = "high_vol_ranging"

    if rsi_avg_dev <= rsi_band and osc_count >= max(2, slope_lookback // 2):
        if regime == "trending":
            regime = "high_vol_ranging"

    if rsi_avg_dev >= rsi_band * 1.5 and osc_count <= 1 and regime != "trending":
        regime = "trending"

    reason = (
        f"tf={main_tf} ma_angle={ma_angle:.4f} atr_rel={atr_rel:.4f} "
        f"rsi_dev={rsi_avg_dev:.2f} osc={osc_count}"
    )

    return RegimeSignal(regime=regime, reason=reason)
