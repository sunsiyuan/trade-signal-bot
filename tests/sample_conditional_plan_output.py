"""Generate a sample JSON event that includes a conditional plan."""

import json
from datetime import datetime, timedelta, timezone

from bot.conditional_plan import build_conditional_plan
from bot.config import Settings
from bot.logging_schema import build_signal_event
from bot.models import DerivativeIndicators, MarketSnapshot, TimeframeIndicators
from bot.signal_engine import TradeSignal


def _tf(tf: str, close: float, now: datetime, trend_label: str = "down", atr: float = 2.0):
    return TimeframeIndicators(
        timeframe=tf,
        close=close,
        ma7=close,
        ma25=close,
        ma99=close,
        rsi6=55.0,
        rsi12=50.0,
        rsi24=50.0,
        macd=0.1,
        macd_signal=0.05,
        macd_hist=-0.02,
        atr=atr,
        volume=100.0,
        trend_label=trend_label,
        last_candle_open_utc=now - timedelta(hours=4),
        last_candle_close_utc=now,
        is_last_candle_closed=True,
    )


def build_snapshot():
    now = datetime.now(timezone.utc)
    tf_4h = _tf("4h", 190.0, now, trend_label="down", atr=5.0)
    tf_1h = _tf("1h", 188.0, now, trend_label="down", atr=3.0)
    tf_15m = _tf("15m", 186.5, now - timedelta(minutes=5), trend_label="down", atr=1.5)
    deriv = DerivativeIndicators(
        funding=0.01,
        open_interest=1000.0,
        oi_change_24h=1.0,
        orderbook_asks=[],
        orderbook_bids=[],
    )
    return MarketSnapshot(
        symbol="AAVE/USDC:USDC",
        ts=now,
        tf_4h=tf_4h,
        tf_1h=tf_1h,
        tf_15m=tf_15m,
        deriv=deriv,
        regime="high_vol_ranging",
        regime_reason="range with volatility",
        bids=120.0,
        asks=100.0,
    )


def main():
    settings = Settings()
    snap = build_snapshot()
    signal = TradeSignal(
        symbol=snap.symbol,
        direction="none",
        trade_confidence=0.62,
        edge_confidence=1.0,
        setup_type="none",
        debug_scores={"long": 0.8, "short": 0.1},
    )

    plan = build_conditional_plan(signal, snap, settings)
    signal.conditional_plan = plan

    event = build_signal_event(snap, signal, settings)
    print(json.dumps(event, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
