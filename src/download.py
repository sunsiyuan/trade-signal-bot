"""Download historical candle data for local research.

This module provides a CLI and callable helpers to generate (or fetch)
candle data for a given symbol and timeframe. It writes the results to
CSV and emits a metadata JSON file containing basic integrity
information.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List

import pandas as pd


logger = logging.getLogger(__name__)


SUPPORTED_TIMEFRAMES = {
    "1m": timedelta(minutes=1),
    "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "1h": timedelta(hours=1),
    "4h": timedelta(hours=4),
    "1d": timedelta(days=1),
}


@dataclass
class Candle:
    """Simple container for a single candle."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    funding: float | None = None
    oi: float | None = None


def timeframe_to_timedelta(timeframe: str) -> timedelta:
    """Return the expected timedelta for a timeframe string."""

    if timeframe not in SUPPORTED_TIMEFRAMES:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    return SUPPORTED_TIMEFRAMES[timeframe]


def compute_sha256(file_path: Path) -> str:
    """Compute the SHA256 checksum for a given file."""

    import hashlib

    hasher = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _build_date_range(start: str, end: str, timeframe: str) -> List[datetime]:
    delta = timeframe_to_timedelta(timeframe)
    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(end).replace(tzinfo=timezone.utc) + timedelta(days=1)
    dates = []
    current = start_dt
    while current < end_dt:
        dates.append(current)
        current += delta
    return dates


def _synthetic_candles(symbol: str, timeframe: str, start: str, end: str) -> List[Candle]:
    """Generate deterministic synthetic candles for offline use."""

    dates = _build_date_range(start, end, timeframe)
    if not dates:
        return []

    seed = abs(hash((symbol, timeframe))) % 10_000
    rng = pd.Series(range(len(dates)), dtype=float)
    base = rng.rolling(window=5, min_periods=1).mean() + (seed % 100) / 10

    candles: List[Candle] = []
    for idx, ts in enumerate(dates):
        noise = (idx % 7) * 0.1
        open_price = float(base.iloc[idx] + noise)
        close_price = open_price + ((-1) ** idx) * 0.05
        high = max(open_price, close_price) + 0.02
        low = min(open_price, close_price) - 0.02
        volume = 100 + idx * 0.5
        candles.append(
            Candle(
                timestamp=ts,
                open=open_price,
                high=high,
                low=low,
                close=close_price,
                volume=volume,
                funding=None,
                oi=None,
            )
        )
    return candles


def validate_candles(candles: Iterable[Candle], timeframe: str) -> bool:
    """Validate candle continuity and ordering.

    Returns True if gaps were detected.
    """

    delta = timeframe_to_timedelta(timeframe)
    timestamps: List[datetime] = []
    for candle in candles:
        timestamps.append(candle.timestamp)

    if not timestamps:
        logger.warning("No candles to validate.")
        return False

    if any(timestamps[i] >= timestamps[i + 1] for i in range(len(timestamps) - 1)):
        raise ValueError("Timestamps are not strictly increasing")

    expected = timestamps[0]
    has_gaps = False
    for ts in timestamps[1:]:
        expected += delta
        if ts != expected:
            has_gaps = True
            logger.warning("Gap detected between %s and %s", expected, ts)
            expected = ts
    return has_gaps


def _write_csv(path: Path, candles: Iterable[Candle]) -> None:
    fieldnames = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "funding",
        "oi",
    ]
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for candle in candles:
            writer.writerow(
                {
                    "timestamp": candle.timestamp.isoformat().replace("+00:00", "Z"),
                    "open": candle.open,
                    "high": candle.high,
                    "low": candle.low,
                    "close": candle.close,
                    "volume": candle.volume,
                    "funding": candle.funding,
                    "oi": candle.oi,
                }
            )


def download_data(symbol: str, timeframe: str, start: str, end: str, out_dir: str | Path = "data/raw") -> dict:
    """Download or generate candles and persist them locally.

    Returns the metadata dictionary for downstream consumption.
    """

    candles = _synthetic_candles(symbol, timeframe, start, end)
    if not candles:
        raise ValueError("No candles were generated for the requested range")

    has_gaps = validate_candles(candles, timeframe)
    base_dir = Path(out_dir) / symbol
    base_dir.mkdir(parents=True, exist_ok=True)

    csv_path = base_dir / f"{symbol}_{timeframe}.csv"
    _write_csv(csv_path, candles)

    checksum = compute_sha256(csv_path)
    metadata = {
        "symbol": symbol,
        "timeframe": timeframe,
        "source": "Synthetic generator (offline)",
        "start": candles[0].timestamp.isoformat().replace("+00:00", "Z"),
        "end": candles[-1].timestamp.isoformat().replace("+00:00", "Z"),
        "rows": len(candles),
        "downloaded_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "checksum_sha256": checksum,
        "has_gaps": has_gaps,
    }

    metadata_path = base_dir / f"{symbol}_{timeframe}_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return metadata


def parse_args(args: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download historical candles")
    parser.add_argument("--symbol", required=True, help="Trading symbol, e.g., HYPE")
    parser.add_argument("--timeframe", required=True, choices=SUPPORTED_TIMEFRAMES.keys())
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--out-dir", default="data/raw", help="Output directory root")
    return parser.parse_args(args)


def main(argv: List[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args(argv)
    try:
        metadata = download_data(
            symbol=args.symbol,
            timeframe=args.timeframe,
            start=args.start,
            end=args.end,
            out_dir=args.out_dir,
        )
        logger.info("Download completed: %s", metadata)
        return 0
    except Exception as exc:  # pragma: no cover - defensive CLI handling
        logger.error("Download failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
