from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Literal


Direction = Literal["long", "short", "none"]
PlanType = Literal["EXECUTE_NOW", "PLACE_LIMIT_4H", "WATCH_ONLY"]
OrderStatus = Literal["open", "filled", "expired"]
ExitReason = Literal["tp", "sl", "expired", "manual"]


@dataclass
class Candle:
    ts_open_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class OptionalOI:
    ts_ms: int
    open_interest: Optional[float]
    oi_change_24h: Optional[float] = None


@dataclass
class Order:
    order_id: str
    signal_id: str
    symbol: str
    plan_type: PlanType
    direction: Direction
    entry_price: Optional[float]
    created_ts: int
    valid_until_ts: int
    sl: Optional[float] = None
    tp: Optional[float] = None
    setup_type: str = ""
    decision_trace: Optional[Dict] = None
    data_coverage: Dict[str, bool] = field(default_factory=dict)
    status: OrderStatus = "open"
    filled_ts: Optional[int] = None
    filled_price: Optional[float] = None
    expired_ts: Optional[int] = None


@dataclass
class Fill:
    order_id: str
    filled_ts: int
    filled_price: float
    qty: float


@dataclass
class Position:
    trade_id: str
    signal_id: str
    symbol: str
    direction: Direction
    entry_ts: int
    entry_price: float
    sl: Optional[float]
    tp: Optional[float]
    status: Literal["open", "closed"] = "open"
    exit_ts: Optional[int] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[ExitReason] = None


@dataclass
class TradeRecord:
    trade_id: str
    symbol: str
    signal_id: str
    plan_type: PlanType
    direction: Direction
    setup_type: str
    order_created_ts: Optional[int]
    filled_ts: Optional[int]
    filled_price: Optional[float]
    expired: bool
    exit_ts: Optional[int]
    exit_price: Optional[float]
    exit_reason: Optional[ExitReason]
    pnl_abs: Optional[float]
    pnl_pct: Optional[float]
    decision_trace: Optional[Dict]
    data_coverage: Dict[str, bool]
    duplicate_skipped: bool = False
    duplicate_reason: Optional[str] = None
    first_exec_ts: Optional[int] = None
    cooldown_skipped: bool = False
    cooldown_remaining_sec: Optional[float] = None
    in_position: bool = False
    forced_close: bool = False
    gate_decision: Optional[str] = None
    scope_key: Optional[str] = None
    dedup_key: Optional[str] = None


@dataclass
class BacktestResult:
    symbol: str
    mode: str
    trades: List[TradeRecord] = field(default_factory=list)
    summary: Dict[str, object] = field(default_factory=dict)
