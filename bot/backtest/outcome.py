from __future__ import annotations

from typing import Dict, List

from .types import TradeRecord


def _max_drawdown(equity_curve: List[float]) -> float:
    peak = equity_curve[0] if equity_curve else 0.0
    max_dd = 0.0
    for value in equity_curve:
        if value > peak:
            peak = value
        drawdown = peak - value
        if drawdown > max_dd:
            max_dd = drawdown
    return max_dd


def summarize_trades(trades: List[TradeRecord], mode: str) -> Dict[str, object]:
    filled_trades = [t for t in trades if t.filled_ts is not None and t.pnl_abs is not None]
    wins = [t for t in filled_trades if t.pnl_abs is not None and t.pnl_abs > 0]

    total_pnl = sum(t.pnl_abs or 0.0 for t in filled_trades)
    equity_curve = []
    running = 0.0
    for trade in filled_trades:
        running += trade.pnl_abs or 0.0
        equity_curve.append(running)

    max_dd = _max_drawdown(equity_curve)
    fill_rate = len(filled_trades) / max(len([t for t in trades if not t.duplicate_skipped]), 1)

    time_to_fill = []
    for trade in trades:
        if trade.filled_ts and trade.order_created_ts:
            time_to_fill.append((trade.filled_ts - trade.order_created_ts) / 1000)

    avg_time_to_fill = sum(time_to_fill) / len(time_to_fill) if time_to_fill else 0.0

    oi_flags = [t.data_coverage.get("has_oi", False) for t in trades]
    ob_flags = [t.data_coverage.get("has_orderbook", False) for t in trades]

    data_coverage = {
        "has_oi_ratio": sum(1 for f in oi_flags if f) / max(len(oi_flags), 1),
        "has_orderbook_ratio": sum(1 for f in ob_flags if f) / max(len(ob_flags), 1),
    }

    return {
        "mode": mode,
        "metrics": {
            "trade_count": len(filled_trades),
            "win_rate": len(wins) / max(len(filled_trades), 1),
            "total_pnl": total_pnl,
            "max_drawdown": max_dd,
            "fill_rate": fill_rate,
            "avg_time_to_fill_sec": avg_time_to_fill,
        },
        "data_coverage": data_coverage,
    }
