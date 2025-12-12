from dataclasses import dataclass, field
from typing import List, Literal, Optional

from .models import MarketSnapshot


Regime = Literal["trending", "high_vol_ranging", "low_vol_ranging", "unknown"]


@dataclass
class RegimeSignal:
    regime: Regime
    reason: str
    ma_angle: float
    atr_rel: float
    rsi_avg_dev: float
    osc_count: int
    degraded: bool = False
    missing_fields: List[str] = field(default_factory=list)


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
    missing: List[str] = []
    degraded = False

    ma_history = getattr(tf, "ma25_history", None) or []
    rsi_history = getattr(tf, "rsi6_history", None) or []

    if len(ma_history) < 2:
        degraded = True
        missing.append("ma25_history")
        ma_angle = 0.0
    else:
        if len(ma_history) > slope_lookback:
            base = ma_history[-slope_lookback]
            ma_angle = (ma_history[-1] - base) / max(abs(base), 1e-9)
        else:
            base = ma_history[0]
            ma_angle = (ma_history[-1] - base) / max(abs(base), 1e-9)

    atr_rel = tf.atr / max(tf.close, 1e-6)

    if rsi_history:
        deviations = [abs(v - 50) for v in rsi_history[-slope_lookback:]]
        rsi_avg_dev = sum(deviations) / max(len(deviations), 1)
    else:
        degraded = True
        if "rsi6_history" not in missing:
            missing.append("rsi6_history")
        rsi_avg_dev = abs(tf.rsi6 - 50)

    if len(rsi_history) < 3:
        degraded = True
        if "rsi6_history" not in missing:
            missing.append("rsi6_history")
        osc_count = 0
    else:
        osc_count = sum(
            1
            for i in range(1, len(rsi_history))
            if (rsi_history[i - 1] - 50) * (rsi_history[i] - 50) < 0
        )

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

    dq = "" if not degraded else f" degraded=1 missing={','.join(missing)}"
    reason = (
        f"tf={main_tf} ma_angle={ma_angle:.4f} atr_rel={atr_rel:.4f} "
        f"rsi_avg_dev={rsi_avg_dev:.2f} osc_count={osc_count}{dq}"
    )

    return RegimeSignal(
        regime=regime,
        reason=reason,
        ma_angle=ma_angle,
        atr_rel=atr_rel,
        rsi_avg_dev=rsi_avg_dev,
        osc_count=osc_count,
        degraded=degraded,
        missing_fields=missing,
    )
