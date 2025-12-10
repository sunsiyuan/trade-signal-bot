import ccxt
import pandas as pd
import math
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

from .config import Settings
from .indicators import ema, compute_rsi, macd, atr, detect_trend
from .models import (
    TimeframeIndicators,
    DerivativeIndicators,
    MarketSnapshot,
)


class HyperliquidDataClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.exchange = ccxt.hyperliquid({"enableRateLimit": True})
        # preload markets so symbol resolution matches Hyperliquid contracts
        self.exchange.load_markets()
        self.market = self.exchange.market(self.settings.symbol)

    @staticmethod
    def _to_float(value, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

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

    def _normalize_symbol(self, symbol: str) -> str:
        if not isinstance(symbol, str):
            return ""
        return symbol.replace(":", "/").upper()

    def _symbol_matches(self, candidate_symbol: str) -> bool:
        target_symbols = {
            self._normalize_symbol(self.settings.symbol),
            self._normalize_symbol(self.market.get("symbol")),
            self._normalize_symbol(self.market.get("id")),
        }
        return self._normalize_symbol(candidate_symbol) in target_symbols

    def _extract_funding(self, entry: Dict) -> Optional[float]:
        if not isinstance(entry, dict):
            return None

        info = entry.get("info", {}) if isinstance(entry.get("info"), dict) else {}

        candidates = [
            entry.get("fundingRate"),
            entry.get("nextFundingRate"),
            entry.get("predictedFundingRate"),
            entry.get("lastFundingRate"),
            info.get("fundingRate"),
            info.get("funding"),
            info.get("nextFundingRate"),
            info.get("predictedFundingRate"),
        ]

        return next((c for c in candidates if c is not None), None)

    def _extract_open_interest(self, entry: Dict) -> Optional[float]:
        if not isinstance(entry, dict):
            return None

        info = entry.get("info", {}) if isinstance(entry.get("info"), dict) else {}

        candidates = [
            entry.get("openInterest"),
            entry.get("openInterestAmount"),
            entry.get("openInterestValue"),
            info.get("openInterest"),
            info.get("openInterestAmount"),
            info.get("openInterestValue"),
        ]

        for candidate in candidates:
            if candidate is None:
                continue
            value = self._to_float(candidate, default=float("nan"))
            if math.isfinite(value):
                return value

        return None

    def _compute_oi_change_24h(self, current_oi: float) -> Optional[float]:
        """
        Hyperliquid 没有直接的 24h OI 变化字段。尝试通过
        fetch_open_interest_history 获取过去 24h 的首尾数据，
        返回百分比变化 (当前 - 24h 前) / 24h 前 * 100。
        失败时返回 None，不影响主流程。
        """

        if current_oi is None or not math.isfinite(current_oi):
            return None

        try:
            since_ms = int((datetime.now(timezone.utc) - timedelta(hours=24)).timestamp() * 1000)
            history = self.exchange.fetch_open_interest_history(
                self.settings.symbol,
                timeframe="1h",
                since=since_ms,
                limit=48,
            )
        except Exception:
            return None

        if not history:
            return None

        # 取最早一条作为 24h 前参考值
        earliest = history[0] if isinstance(history, list) else history
        if isinstance(history, list) and history:
            earliest = sorted(
                history,
                key=lambda x: x.get("timestamp", 0) if isinstance(x, dict) else 0,
            )[0]

        previous_oi = self._extract_open_interest(earliest)
        if previous_oi is None or previous_oi <= 0:
            return None

        return (current_oi - previous_oi) / previous_oi * 100

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
        ticker_info: Dict = {}
        ticker_error = None
        try:
            ticker = self.exchange.fetch_ticker(self.settings.symbol)
            ticker_info = ticker.get("info", {}) or {}
        except Exception as exc:  # pragma: no cover - network failure fallback
            ticker_error = str(exc)

        funding_entry = None
        funding_source = None
        raw_funding = None
        funding_rates_error = None
        open_interest = self._to_float(ticker_info.get("openInterest"))

        try:
            # Hyperliquid metaAndAssetCtxs returns perps context with "funding" and
            # "openInterest"; ccxt exposes it via fetch_funding_rates(). Guard
            # against non-dict payloads (e.g., error strings) so we don't call
            # .get on a str and override valid ticker funding.
            rates = self.exchange.fetch_funding_rates()

            candidate = None
            if isinstance(rates, list):
                candidate = next(
                    (
                        r
                        for r in rates
                        if isinstance(r, dict)
                        and self._symbol_matches(r.get("symbol"))
                    ),
                    None,
                )
            elif isinstance(rates, dict) and rates.get("symbol"):
                candidate = (
                    rates
                    if self._symbol_matches(rates.get("symbol"))
                    else None
                )

            if candidate:
                funding_entry = candidate
                raw_funding = self._extract_funding(candidate)
                funding_source = "fetch_funding_rates"
                entry_info = candidate.get("info", {}) if isinstance(candidate, dict) else {}
                if isinstance(entry_info, dict):
                    open_interest = self._to_float(
                        entry_info.get("openInterest"), default=open_interest
                    )
            elif rates is not None:
                funding_rates_error = f"unexpected funding type: {type(rates).__name__}"
        except Exception as exc:  # pragma: no cover - network failure fallback
            funding_rates_error = str(exc)

        if raw_funding is None:
            raw_funding = ticker_info.get("funding") or ticker_info.get("fundingRate")
            if raw_funding is not None:
                funding_source = "ticker_info"

        funding = self._to_float(raw_funding, default=0.0)

        oi_change_24h = self._compute_oi_change_24h(open_interest) or 0.0

        # 订单簿
        orderbook = self.exchange.fetch_order_book(self.settings.symbol, limit=20)
        asks_raw: List[List[float]] = orderbook.get("asks", [])
        bids_raw: List[List[float]] = orderbook.get("bids", [])

        def _format_price(price: float) -> float:
            try:
                return float(
                    self.exchange.price_to_precision(self.settings.symbol, price)
                )
            except Exception:
                return float(price)

        orderbook_asks = [
            {"price": _format_price(p), "size": float(s)} for p, s in asks_raw[:10]
        ]
        orderbook_bids = [
            {"price": _format_price(p), "size": float(s)} for p, s in bids_raw[:10]
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
        closes_list = close.tolist()

        ma7 = ema(close, 7)
        ma25 = ema(close, 25)
        ma99 = ema(close, 99)

        rsi6_series = compute_rsi(closes_list, s.rsi_fast).set_axis(close.index)
        rsi12_series = compute_rsi(closes_list, s.rsi_mid).set_axis(close.index)
        rsi24_series = compute_rsi(closes_list, s.rsi_slow).set_axis(close.index)

        rsi6_value = float(rsi6_series.iloc[-1])
        rsi12_value = float(rsi12_series.iloc[-1])
        rsi24_value = float(rsi24_series.iloc[-1])

        if (
            self.settings.debug_rsi
            and self.settings.symbol == "HYPE/USDC:USDC"
            and timeframe == "15m"
        ):
            print(
                f"[RSI_DEBUG] tf=15m symbol={self.settings.symbol} "
                f"last_closes={closes_list[-10:]} "
                f"rsi6={rsi6_value:.2f} rsi12={rsi12_value:.2f} rsi24={rsi24_value:.2f}"
            )

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
            rsi6=rsi6_value,
            rsi12=rsi12_value,
            rsi24=rsi24_value,
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
