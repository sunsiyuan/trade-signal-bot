from __future__ import annotations

import argparse
from datetime import datetime, timezone

from bot.backtest.data_store import JSONLDataStore
from bot.backtest.fetcher import MarketDataFetcher, TokenBucket, flatten_candles


def _parse_timeframes(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_symbols(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill historical candles")
    parser.add_argument("--symbols", required=True, help="Comma-separated symbols")
    parser.add_argument("--timeframes", default="15m,1h,4h", help="Comma-separated timeframes")
    parser.add_argument("--start", required=True, help="ISO start datetime")
    parser.add_argument("--end", required=True, help="ISO end datetime")
    parser.add_argument("--data-dir", default="data/market_data", help="Data directory")
    parser.add_argument("--max_requests_per_minute", type=int, default=60, help="Soft rate limit")

    args = parser.parse_args()

    symbols = _parse_symbols(args.symbols)
    timeframes = _parse_timeframes(args.timeframes)
    start_dt = datetime.fromisoformat(args.start).astimezone(timezone.utc)
    end_dt = datetime.fromisoformat(args.end).astimezone(timezone.utc)

    data_store = JSONLDataStore(base_dir=args.data_dir)
    fetcher = MarketDataFetcher()
    limiter = TokenBucket(rate_per_minute=args.max_requests_per_minute)

    since_ms = int(start_dt.timestamp() * 1000)
    until_ms = int(end_dt.timestamp() * 1000)

    for symbol in symbols:
        for timeframe in timeframes:
            limiter.take()
            result = fetcher.fetch_ohlcv_range(symbol, timeframe, since_ms, until_ms=until_ms)
            candles = flatten_candles(result.candles)
            inserted = data_store.upsert_candles(symbol, timeframe, candles)

            latest_ts = data_store.get_max_ts(symbol, timeframe)
            latest_dt = datetime.fromtimestamp(latest_ts / 1000, tz=timezone.utc) if latest_ts else None

            print(
                "[backfill_market_data] symbol={symbol} tf={timeframe} inserted={inserted} "
                "total={total} latest={latest} backoff={backoff}".format(
                    symbol=symbol,
                    timeframe=timeframe,
                    inserted=inserted,
                    total=len(candles),
                    latest=latest_dt.isoformat() if latest_dt else "n/a",
                    backoff=result.backoff_events,
                )
            )


if __name__ == "__main__":
    main()
