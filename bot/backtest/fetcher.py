from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional

import ccxt

from .types import Candle


@dataclass
class FetchResult:
    candles: List[Candle]
    requests: int
    backoff_events: int


class MarketDataFetcher:
    def __init__(self, exchange: Optional[ccxt.hyperliquid] = None):
        self.exchange = exchange or ccxt.hyperliquid({"enableRateLimit": True})
        if exchange is None:
            self.exchange.load_markets()

    def _call_with_backoff(self, func, *, max_retries: int = 6, base_delay: float = 1.0):
        backoff_events = 0
        for attempt in range(max_retries):
            try:
                return func(), backoff_events
            except Exception:
                backoff_events += 1
                sleep_for = base_delay * (2 ** attempt) + random.uniform(0, 0.4)
                time.sleep(sleep_for)
        raise RuntimeError("API call failed after retries")

    def fetch_ohlcv_page(
        self,
        symbol: str,
        timeframe: str,
        since_ms: int,
        limit: int = 500,
    ) -> List[Candle]:
        def _fetch():
            return self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=limit)

        raw, _ = self._call_with_backoff(_fetch)
        return [
            Candle(
                ts_open_ms=int(row[0]),
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
                volume=float(row[5]),
            )
            for row in raw
        ]

    def fetch_ohlcv_range(
        self,
        symbol: str,
        timeframe: str,
        since_ms: int,
        until_ms: Optional[int] = None,
        limit: int = 500,
    ) -> FetchResult:
        candles: List[Candle] = []
        backoff_events = 0
        requests = 0
        next_since = since_ms

        while True:
            def _fetch():
                return self.exchange.fetch_ohlcv(
                    symbol,
                    timeframe=timeframe,
                    since=next_since,
                    limit=limit,
                )

            raw, backoff = self._call_with_backoff(_fetch)
            backoff_events += backoff
            requests += 1

            if not raw:
                break

            page = [
                Candle(
                    ts_open_ms=int(row[0]),
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5]),
                )
                for row in raw
            ]

            candles.extend(page)
            last_ts = page[-1].ts_open_ms

            if until_ms is not None and last_ts >= until_ms:
                break

            next_since = last_ts + 1
            if len(page) < limit:
                break

        return FetchResult(candles=candles, requests=requests, backoff_events=backoff_events)


class TokenBucket:
    def __init__(self, rate_per_minute: int):
        self.capacity = max(rate_per_minute, 1)
        self.tokens = self.capacity
        self.updated_at = time.time()

    def take(self, tokens: int = 1) -> None:
        while True:
            now = time.time()
            elapsed = now - self.updated_at
            refill = elapsed * (self.capacity / 60.0)
            if refill > 0:
                self.tokens = min(self.capacity, self.tokens + refill)
                self.updated_at = now
            if self.tokens >= tokens:
                self.tokens -= tokens
                return
            time.sleep(0.05)


def flatten_candles(records: Iterable[Candle]) -> List[Candle]:
    seen = {}
    for candle in records:
        seen[candle.ts_open_ms] = candle
    return [seen[k] for k in sorted(seen.keys())]
