from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta, timezone

from bot.backtest.data_store import JSONLDataStore
from bot.backtest.fetcher import MarketDataFetcher, TokenBucket, flatten_candles


def _parse_timeframes(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_symbols(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Incrementally fetch market data with rate limit handling")
    parser.add_argument("--symbols", required=True, help="Comma-separated symbols")
    parser.add_argument("--timeframes", default="15m,1h,4h", help="Comma-separated timeframes")
    parser.add_argument("--data-dir", default="data/market_data", help="Data directory")
    parser.add_argument("--lookback_hours", type=int, default=48, help="Lookback buffer in hours")
    parser.add_argument("--max_requests_per_minute", type=int, default=60, help="Soft rate limit")

    args = parser.parse_args()

    symbols = _parse_symbols(args.symbols)
    timeframes = _parse_timeframes(args.timeframes)
    data_store = JSONLDataStore(base_dir=args.data_dir)
    fetcher = MarketDataFetcher()
    limiter = TokenBucket(rate_per_minute=args.max_requests_per_minute)

    now = datetime.now(timezone.utc)

    for symbol in symbols:
        for timeframe in timeframes:
            max_ts = data_store.get_max_ts(symbol, timeframe)
            lookback = now - timedelta(hours=args.lookback_hours)
            since_ms = int((lookback if max_ts is None else datetime.fromtimestamp(max_ts / 1000, tz=timezone.utc) - timedelta(hours=args.lookback_hours)).timestamp() * 1000)
            until_ms = int(now.timestamp() * 1000)

            start_time = time.time()
            limiter.take()
            result = fetcher.fetch_ohlcv_range(symbol, timeframe, since_ms, until_ms=until_ms)
            candles = flatten_candles(result.candles)
            inserted = data_store.upsert_candles(symbol, timeframe, candles)
            elapsed = time.time() - start_time

            latest_ts = data_store.get_max_ts(symbol, timeframe)
            latest_dt = datetime.fromtimestamp(latest_ts / 1000, tz=timezone.utc) if latest_ts else None

            print(
                "[fetch_market_data] symbol={symbol} tf={timeframe} inserted={inserted} "
                "total={total} latest={latest} backoff={backoff} elapsed_sec={elapsed:.2f}".format(
                    symbol=symbol,
                    timeframe=timeframe,
                    inserted=inserted,
                    total=len(candles),
                    latest=latest_dt.isoformat() if latest_dt else "n/a",
                    backoff=result.backoff_events,
                    elapsed=elapsed,
                )
            )


if __name__ == "__main__":
    main()
