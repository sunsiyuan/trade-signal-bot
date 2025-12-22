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


def _rate(numerator: int, denominator: int) -> float:
    return numerator / max(denominator, 1)


def summarize_trades(
    trades: List[TradeRecord], mode: str, gate_stats: Dict[str, object]
) -> Dict[str, object]:
    closed_trades = [t for t in trades if t.pnl_abs is not None]
    wins = [t for t in closed_trades if t.pnl_abs is not None and t.pnl_abs > 0]

    total_pnl = sum(t.pnl_abs or 0.0 for t in closed_trades)
    equity_curve: List[float] = []
    running = 0.0
    for trade in closed_trades:
        running += trade.pnl_abs or 0.0
        equity_curve.append(running)

    max_dd = _max_drawdown(equity_curve)
    signal_count = int(gate_stats.get("signal_count", 0))
    execute_decision_count = int(gate_stats.get("execute_decision_count", 0))
    order_created_count = int(gate_stats.get("order_created_count", 0))
    filled_count = int(gate_stats.get("filled_count", 0))
    closed_trade_count = int(gate_stats.get("closed_trade_count", 0))
    signal_to_execute_rate = _rate(execute_decision_count, signal_count)
    execute_to_order_rate = _rate(order_created_count, execute_decision_count)
    order_fill_rate = _rate(filled_count, order_created_count)
    signal_fill_rate = _rate(filled_count, signal_count)
    closed_trade_rate = _rate(closed_trade_count, filled_count)

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
            "trade_count": closed_trade_count,
            "win_rate": _rate(len(wins), closed_trade_count),
            "total_pnl": total_pnl,
            "max_drawdown": max_dd,
            "fill_rate": order_fill_rate,
            "avg_time_to_fill_sec": avg_time_to_fill,
        },
        "data_coverage": data_coverage,
        "gate_stats": {
            "signal_count": signal_count,
            "decision_count": int(gate_stats.get("decision_count", 0)),
            "execute_decision_count": execute_decision_count,
            "order_created_count": order_created_count,
            "filled_count": filled_count,
            "closed_trade_count": closed_trade_count,
            "skipped_count": int(gate_stats.get("skipped_count", 0)),
            "skipped_by_reason": gate_stats.get("skipped_by_reason", {}),
            "forced_close_count": int(gate_stats.get("forced_close_count", 0)),
            "cooldown_blocked_count": int(gate_stats.get("cooldown_blocked_count", 0)),
            "dedup_blocked_count": int(gate_stats.get("dedup_blocked_count", 0)),
            "in_position_blocked_count": int(gate_stats.get("in_position_blocked_count", 0)),
            "rates": {
                "signal_to_execute_rate": signal_to_execute_rate,
                "execute_to_order_rate": execute_to_order_rate,
                "order_fill_rate": order_fill_rate,
                "signal_fill_rate": signal_fill_rate,
                "closed_trade_rate": closed_trade_rate,
            },
        },
    }
