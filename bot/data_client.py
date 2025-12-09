import ccxt
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, List

from .config import Settings
from .indicators import ema, rsi, macd, atr, detect_trend
from .models import (
    TimeframeIndicators,
    DerivativeIndicators,
    MarketSnapshot,
)


class HyperliquidDataClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.exchange = ccxt.hyperliquid({"enableRateLimit": True})

    def _fetch_ohlcv(self, timeframe: str, limit: int) -> pd.DataFrame:
        raw = self.exchange.fetch_ohlcv(
            self.settings.symbol,
            timeframe=timeframe,
            limit=limit,
        )
        df = pd.DataFrame(
            raw,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return self._drop_incomplete_last_candle(df, timeframe)

    def _drop_incomplete_last_candle(
        self, df: pd.DataFrame, timeframe: str
    ) -> pd.DataFrame:
        """
        Hyperliquid returns the latest in-progress candle. Drop it so RSI/MA are
        computed on fully closed candles, keeping values consistent with the UI.
        """

        if df.empty:
            return df

        duration = pd.to_timedelta(timeframe)
        last_start = df["timestamp"].iloc[-1].to_pydatetime().replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)

        if now < last_start + duration:
            return df.iloc[:-1]

        return df

    def fetch_all_ohlcv(self) -> Dict[str, pd.DataFrame]:
        s = self.settings
        return {
            "4h": self._fetch_ohlcv(s.tf_4h, s.candles_4h),
            "1h": self._fetch_ohlcv(s.tf_1h, s.candles_1h),
            "15m": self._fetch_ohlcv(s.tf_15m, s.candles_15m),
        }

    def fetch_derivative_indicators(self) -> DerivativeIndicators:
        """
        这里用 ticker + 订单簿模拟 funding / OI，
        真正的精确数据可以以后直接接 Hyperliquid 官方 REST。
        """
        # ticker 中有些交易所会把 funding / oi 放 info，这里做兼容写法
        ticker = self.exchange.fetch_ticker(self.settings.symbol)
        info = ticker.get("info", {})

        funding_entry = None
        funding_source = None
        raw_funding = None
        try:
            rates = self.exchange.fetch_funding_rates([self.settings.symbol])
            # ccxt returns dict keyed by symbol for funding_rates
            funding_entry = rates.get(self.settings.symbol) if isinstance(rates, dict) else None
            if funding_entry is None and isinstance(rates, list):
                funding_entry = next(
                    (r for r in rates if r.get("symbol") == self.settings.symbol), None
                )
            raw_funding = None if funding_entry is None else funding_entry.get("fundingRate")
            funding_source = "funding_rates"
        except Exception:
            # fallback to ticker info keys
            funding_source = "funding_rates_error"

        if raw_funding is None:
            raw_funding = info.get("funding") or info.get("fundingRate")
            if raw_funding is not None:
                funding_source = "ticker_info"

        funding = float(raw_funding) if raw_funding is not None else 0.0

        print(
            "RAW FUNDING:",
            {
                "ticker_info": info,
                "funding_entry": funding_entry,
                "funding_source": funding_source,
            },
        )
        print("PARSED FUNDING:", funding)
        open_interest = float(info.get("openInterest", 0.0))
        if funding_entry is not None:
            entry_info = funding_entry.get("info", {})
            open_interest = float(entry_info.get("openInterest", open_interest))

        # 很多交易所没 24h OI，需要自己再拉一次或用 info 字段；这里先简单置 0
        oi_change_24h = 0.0

        # 订单簿
        orderbook = self.exchange.fetch_order_book(self.settings.symbol, limit=20)
        asks_raw: List[List[float]] = orderbook.get("asks", [])
        bids_raw: List[List[float]] = orderbook.get("bids", [])

        orderbook_asks = [
            {"price": float(p), "size": float(s)} for p, s in asks_raw[:10]
        ]
        orderbook_bids = [
            {"price": float(p), "size": float(s)} for p, s in bids_raw[:10]
        ]

        liquidity_comment = "asks>bids" if sum(a["size"] for a in orderbook_asks) > sum(
            b["size"] for b in orderbook_bids
        ) else "bids>=asks"

        return DerivativeIndicators(
            funding=funding,
            open_interest=open_interest,
            oi_change_24h=oi_change_24h,
            orderbook_asks=orderbook_asks,
            orderbook_bids=orderbook_bids,
            liquidity_comment=liquidity_comment,
        )

    def _build_tf_indicators(self, df: pd.DataFrame, timeframe: str) -> TimeframeIndicators:
        s = self.settings

        close = df["close"]

        ma7 = ema(close, 7)
        ma25 = ema(close, 25)
        ma99 = ema(close, 99)

        rsi6 = rsi(close, s.rsi_fast)
        rsi12 = rsi(close, s.rsi_mid)
        rsi24 = rsi(close, s.rsi_slow)

        macd_line, macd_signal, macd_hist = macd(
            close,
            fast=s.macd_fast,
            slow=s.macd_slow,
            signal=s.macd_signal,
        )

        atr_series = atr(df, s.atr_period)

        trend_label = detect_trend(close, ma7, ma25, ma99)

        last = df.index[-1]
        return TimeframeIndicators(
            timeframe=timeframe,
            close=float(close.iloc[-1]),
            ma7=float(ma7.iloc[-1]),
            ma25=float(ma25.iloc[-1]),
            ma99=float(ma99.iloc[-1]),
            rsi6=float(rsi6.iloc[-1]),
            rsi12=float(rsi12.iloc[-1]),
            rsi24=float(rsi24.iloc[-1]),
            macd=float(macd_line.iloc[-1]),
            macd_signal=float(macd_signal.iloc[-1]),
            macd_hist=float(macd_hist.iloc[-1]),
            atr=float(atr_series.iloc[-1]),
            volume=float(df["volume"].iloc[-1]),
            trend_label=trend_label,
        )

    def build_market_snapshot(self) -> MarketSnapshot:
        ohlcvs = self.fetch_all_ohlcv()
        tf4 = self._build_tf_indicators(ohlcvs["4h"], "4h")
        tf1 = self._build_tf_indicators(ohlcvs["1h"], "1h")
        tf15 = self._build_tf_indicators(ohlcvs["15m"], "15m")
        deriv = self.fetch_derivative_indicators()

        ts = datetime.now(timezone.utc)

        return MarketSnapshot(
            symbol=self.settings.symbol,
            ts=ts,
            tf_4h=tf4,
            tf_1h=tf1,
            tf_15m=tf15,
            deriv=deriv,
        )
