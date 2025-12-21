from __future__ import annotations

from typing import Optional

from ..config import Settings
from .data_store import JSONLDataStore
from .simulator import run_backtest as run_simulator_backtest
from .types import BacktestResult


def run_backtest(
    *,
    symbol: str,
    data_dir: str,
    output_dir: str,
    mode: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    settings: Optional[Settings] = None,
) -> BacktestResult:
    settings = settings or Settings(symbol=symbol)
    data_store = JSONLDataStore(base_dir=data_dir)

    return run_simulator_backtest(
        symbol=symbol,
        data_store=data_store,
        output_dir=output_dir,
        mode=mode,
        settings=settings,
        start=start,
        end=end,
    )
