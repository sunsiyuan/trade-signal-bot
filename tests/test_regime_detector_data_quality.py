import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from datetime import datetime, timezone

from bot.config import Settings
from bot.models import DerivativeIndicators, MarketSnapshot, TimeframeIndicators
from bot.regime_detector import detect_regime


def make_tf() -> TimeframeIndicators:
    return TimeframeIndicators(
        timeframe="1h",
        close=100.0,
        ma7=99.0,
        ma25=98.0,
        ma99=97.0,
        rsi6=50.0,
        rsi12=50.0,
        rsi24=50.0,
        macd=0.0,
        macd_signal=0.0,
        macd_hist=0.0,
        atr=1.0,
        volume=1000.0,
    )


def make_snap(tf: TimeframeIndicators) -> MarketSnapshot:
    deriv = DerivativeIndicators(funding=0.0, open_interest=0.0, oi_change_24h=None)
    now = datetime.now(timezone.utc)
    return MarketSnapshot(
        symbol="TEST/USDC",
        ts=now,
        tf_4h=tf,
        tf_1h=tf,
        tf_15m=tf,
        deriv=deriv,
    )


def test_detect_regime_flags_degraded_when_history_missing():
    tf = make_tf()
    snap = make_snap(tf)

    signal = detect_regime(snap, Settings())

    assert signal.degraded is True
    assert "missing=" in signal.reason
    assert "ma25_history" in signal.missing_fields
    assert "rsi6_history" in signal.missing_fields


def test_detect_regime_returns_allowed_value():
    tf = make_tf()
    tf.ma25_history = [98.0, 98.2, 98.4]
    tf.rsi6_history = [52.0, 54.0, 56.0]
    snap = make_snap(tf)

    signal = detect_regime(snap, Settings())

    assert signal.regime in {
        "trending",
        "high_vol_ranging",
        "low_vol_ranging",
        "unknown",
    }
