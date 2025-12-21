from __future__ import annotations

import argparse

from ..config import Settings
from .data_store import JSONLDataStore
from .simulator import run_backtest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run backtests on historical data")
    parser.add_argument("--symbol", required=True, help="Symbol like BTC/USDC:USDC")
    parser.add_argument("--data-dir", default="data/market_data", help="Base dir for historical data")
    parser.add_argument(
        "--mode",
        default="execute_now_only",
        choices=["execute_now_only", "execute_now_and_limit4h"],
        help="Execution mode",
    )
    parser.add_argument("--start", default=None, help="ISO start datetime")
    parser.add_argument("--end", default=None, help="ISO end datetime")
    parser.add_argument("--output-dir", default="data/backtest_output", help="Output folder")

    args = parser.parse_args()

    settings = Settings(symbol=args.symbol)
    data_store = JSONLDataStore(base_dir=args.data_dir)

    run_backtest(
        symbol=args.symbol,
        data_store=data_store,
        output_dir=args.output_dir,
        mode=args.mode,
        settings=settings,
        start=args.start,
        end=args.end,
    )


if __name__ == "__main__":
    main()
