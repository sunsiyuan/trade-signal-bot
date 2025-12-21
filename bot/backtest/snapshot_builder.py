from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

import pandas as pd

from ..config import Settings
from ..indicators import ema, compute_rsi, macd, atr, detect_trend
from ..models import MarketSnapshot, TimeframeIndicators, DerivativeIndicators


@dataclass
class DataWindow:
    tf_15m: pd.DataFrame
    tf_1h: pd.DataFrame
    tf_4h: pd.DataFrame
    oi_1h: Optional[pd.DataFrame] = None


def _ensure_dataframe(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    return df


def _select_closed(df: pd.DataFrame, ts: datetime, timeframe: str) -> pd.DataFrame:
    if df.empty:
        return df
    delta = pd.to_timedelta(timeframe)
    close_times = df["timestamp"] + delta
    mask = close_times <= ts
    return df.loc[mask].copy().reset_index(drop=True)


def _build_tf_indicators(df: pd.DataFrame, timeframe: str, settings: Settings) -> Optional[TimeframeIndicators]:
    if df.empty or len(df) < 5:
        return None

    close = df["close"]
    closes_list = close.tolist()

    ma7 = ema(close, 7)
    ma25 = ema(close, 25)
    ma99 = ema(close, 99)

    rsi6_series = compute_rsi(closes_list, settings.rsi_fast).set_axis(close.index)
    rsi12_series = compute_rsi(closes_list, settings.rsi_mid).set_axis(close.index)
    rsi24_series = compute_rsi(closes_list, settings.rsi_slow).set_axis(close.index)

    macd_line, macd_signal, macd_hist = macd(
        close,
        fast=settings.macd_fast,
        slow=settings.macd_slow,
        signal=settings.macd_signal,
    )

    atr_series = atr(df, settings.atr_period)
    trend_label = detect_trend(close, ma7, ma25, ma99)

    slope_lookback = int(settings.regime.get("slope_lookback", 5))
    hist_len = max(20, slope_lookback * 4)

    lh_cfg = getattr(settings, "liquidity_hunt", {})
    lookback = int(lh_cfg.get("swing_lookback_bars", 50))
    exclude = int(lh_cfg.get("swing_exclude_last", 1))

    window_df = None
    if len(df) > lookback + exclude:
        window_df = df.iloc[-(lookback + exclude) : -exclude]
    elif exclude > 0:
        window_df = df.iloc[:-exclude]
    else:
        window_df = df

    recent_high = None
    recent_low = None
    high_last_n = None
    low_last_n = None
    if window_df is not None and len(window_df) >= 5:
        recent_high = float(window_df["high"].max())
        recent_low = float(window_df["low"].min())
        high_last_n = recent_high
        low_last_n = recent_low

    small_body_lookback = int(lh_cfg.get("small_body_lookback", 8))
    small_body_mult = float(lh_cfg.get("small_body_atr_mult", 0.35))

    tail = df.iloc[-small_body_lookback:] if len(df) >= small_body_lookback else df
    post_spike_small_body_count = None
    if len(tail) >= 3:
        body = (tail["close"] - tail["open"]).abs()
        atr_tail = atr_series.iloc[-len(tail) :].abs()
        if len(atr_tail) == len(tail):
            small_body = body <= (atr_tail * small_body_mult)
            post_spike_small_body_count = int(small_body.sum())

    timeframe_delta = pd.to_timedelta(timeframe)
    timestamps = df["timestamp"].dt.tz_convert(timezone.utc)
    last_open = timestamps.iloc[-1].to_pydatetime()
    last_close = last_open + timeframe_delta

    expected_delta = timeframe_delta
    missing_bars = 0
    gap_list = []
    if len(timestamps) > 1:
        for prev, curr in zip(timestamps.iloc[:-1], timestamps.iloc[1:]):
            gap = curr - prev
            gap_units = gap.total_seconds() / expected_delta.total_seconds()
            if gap_units > 1.01:
                missing = int(round(gap_units - 1))
                missing_bars += missing
                gap_list.append(
                    {
                        "start_utc": (prev + expected_delta).isoformat(),
                        "end_utc": (curr - expected_delta).isoformat(),
                        "missing": missing,
                    }
                )

    high_last = float(df["high"].iloc[-1])
    low_last = float(df["low"].iloc[-1])
    price_last = float(close.iloc[-1])
    price_mid = (high_last + low_last) / 2
    typical_price = (high_last + low_last + price_last) / 3

    prev_close = float(close.iloc[-2]) if len(close) > 1 else None
    return_last = None
    if prev_close and prev_close != 0:
        return_last = (price_last - prev_close) / prev_close

    tr_last = None
    if prev_close is not None:
        tr1 = high_last - low_last
        tr2 = abs(high_last - prev_close)
        tr3 = abs(low_last - prev_close)
        tr_last = max(tr1, tr2, tr3)

    lookback_window = max(
        99,
        settings.macd_slow + settings.macd_signal,
        settings.rsi_slow + 1,
        settings.atr_period,
    )

    return TimeframeIndicators(
        timeframe=timeframe,
        close=price_last,
        ma7=float(ma7.iloc[-1]),
        ma25=float(ma25.iloc[-1]),
        ma99=float(ma99.iloc[-1]),
        ma25_history=[float(x) for x in ma25.tail(hist_len).tolist()],
        rsi6=float(rsi6_series.iloc[-1]),
        rsi6_history=[float(x) for x in rsi6_series.tail(hist_len).tolist()],
        rsi12=float(rsi12_series.iloc[-1]),
        rsi24=float(rsi24_series.iloc[-1]),
        macd=float(macd_line.iloc[-1]),
        macd_signal=float(macd_signal.iloc[-1]),
        macd_hist=float(macd_hist.iloc[-1]),
        atr=float(atr_series.iloc[-1]),
        volume=float(df["volume"].iloc[-1]),
        trend_label=trend_label,
        last_candle_open_utc=last_open,
        last_candle_close_utc=last_close,
        is_last_candle_closed=True,
        bars_used=len(df),
        lookback_window=lookback_window,
        missing_bars_count=missing_bars,
        gap_list=gap_list,
        price_last=price_last,
        price_mid=price_mid,
        typical_price=typical_price,
        return_last=return_last,
        atr_rel=float(atr_series.iloc[-1]) / max(price_last, 1e-9),
        tr_last=tr_last,
        recent_high=recent_high,
        recent_low=recent_low,
        high_last_n=high_last_n,
        low_last_n=low_last_n,
        post_spike_small_body_count=post_spike_small_body_count,
    )


def _compute_rolling_candidate(close_15m: pd.Series, settings: Settings):
    if close_15m is None or len(close_15m) == 0:
        return None, None

    window = close_15m.tail(16)
    if len(window) == 0:
        return None, None

    ma25_series = ema(window, 25)

    slope_lookback = int(getattr(settings, "regime", {}).get("slope_lookback", 5))
    if len(ma25_series) > 1:
        base_idx = max(len(ma25_series) - slope_lookback - 1, 0)
        base_val = ma25_series.iloc[base_idx]
        slope = (ma25_series.iloc[-1] - base_val) / max(abs(base_val), 1e-9)
    else:
        slope = 0.0

    _, _, macd_hist = macd(
        window,
        fast=settings.macd_fast,
        slow=settings.macd_slow,
        signal=settings.macd_signal,
    )
    macd_hist_value = float(macd_hist.iloc[-1]) if len(macd_hist) else None

    trend_ma_angle_min = float(getattr(settings, "regime", {}).get("trend_ma_angle_min", 0.0008))

    if macd_hist_value is None:
        return None, None
    if slope > trend_ma_angle_min and macd_hist_value > 0:
        return "trending", "up"
    if slope < -trend_ma_angle_min and macd_hist_value < 0:
        return "trending", "down"
    return "ranging", None


class SnapshotBuilder:
    def __init__(self, settings: Settings):
        self.settings = settings

    def build_snapshot(self, symbol: str, window: DataWindow, ts: datetime) -> Optional[MarketSnapshot]:
        tf15 = _build_tf_indicators(_select_closed(window.tf_15m, ts, "15m"), "15m", self.settings)
        tf1 = _build_tf_indicators(_select_closed(window.tf_1h, ts, "1h"), "1h", self.settings)
        tf4 = _build_tf_indicators(_select_closed(window.tf_4h, ts, "4h"), "4h", self.settings)

        if tf15 is None or tf1 is None or tf4 is None:
            return None

        oi_value = None
        oi_change_24h = None
        has_oi = False
        if window.oi_1h is not None and not window.oi_1h.empty:
            oi_df = window.oi_1h
            oi_df = oi_df.loc[oi_df["timestamp"] <= ts].copy()
            if not oi_df.empty:
                latest = oi_df.iloc[-1]
                oi_value = latest.get("open_interest")
                oi_change_24h = latest.get("oi_change_24h")
                has_oi = oi_value is not None

        deriv = DerivativeIndicators(
            funding=0.0,
            open_interest=oi_value or 0.0,
            oi_change_24h=oi_change_24h,
            mark_price=None,
            orderbook_asks=[],
            orderbook_bids=[],
            liquidity_comment="",
            ask_wall_size=None,
            bid_wall_size=None,
            ask_to_bid_ratio=None,
            has_large_ask_wall=None,
            has_large_bid_wall=None,
        )

        asks = 0.0
        bids = 0.0

        atrrel = tf1.atr / max(tf1.close, 1e-6)
        rsidev = (tf1.close - tf1.ma25) / max(tf1.ma25, 1e-6)
        rolling_candidate, rolling_dir = _compute_rolling_candidate(window.tf_15m["close"], self.settings)

        return MarketSnapshot(
            symbol=symbol,
            ts=ts,
            tf_4h=tf4,
            tf_1h=tf1,
            tf_15m=tf15,
            deriv=deriv,
            regime="unknown",
            regime_reason="",
            rsidev=rsidev,
            atrrel=atrrel,
            rsi_15m=tf15.rsi6,
            rsi_1h=tf1.rsi6,
            rolling_candidate=rolling_candidate,
            rolling_candidate_dir=rolling_dir,
            rolling_candidate_streak=1 if rolling_candidate else 0,
            asks=asks,
            bids=bids,
            data_flags={
                "has_oi": has_oi,
                "has_orderbook": False,
            },
        )
