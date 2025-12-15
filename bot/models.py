from dataclasses import dataclass, field
from typing import Literal, Optional, List, Dict
from datetime import datetime

Direction = Literal["long", "short", "none"]


@dataclass
class ExecutionIntent:
    symbol: str
    direction: Direction                 # long / short / none

    # --- 理想入场（单点） ---
    entry_price: Optional[float]         # 理想 entry（非当前价也可）
    entry_reason: str                    # "TF_trigger" / "MR_MA25" / "LH_sweep"

    # --- 结构性失效 ---
    invalidation_price: Optional[float]  # 结构破坏位（通常 = SL）

    # --- 执行参数 ---
    ttl_hours: int = 4                   # 固定 4h
    allow_execute_now: bool = True       # 是否允许当前价立即成交

    # --- 风控引用 ---
    atr_4h: Optional[float] = None       # 用于 execution gate（不用于改 SL）

    # --- Debug ---
    reason: str = ""
    debug: Optional[Dict] = None


@dataclass
class ConditionalPlan:
    execution_mode: Literal[
        "EXECUTE_NOW",
        "PLACE_LIMIT_4H",
        "WATCH_ONLY",
    ]

    direction: Direction
    entry_price: Optional[float]
    valid_until_utc: Optional[str]
    cancel_if: Dict[str, bool]
    explain: str


@dataclass
class TimeframeIndicators:
    """单个时间周期的关键指标快照，比如 4H / 1H / 15m。"""
    timeframe: str              # "4h" / "1h" / "15m"
    close: float
    ma7: float
    ma25: float
    ma99: float

    rsi6: float
    rsi12: float
    rsi24: float

    macd: float                 # DIF
    macd_signal: float          # DEA
    macd_hist: float            # 柱子

    atr: float                  # 当前 ATR（可用 1H/15m）
    volume: float               # 最新一根K线的成交量
    vwap: Optional[float] = None  # 如果你后面算 VWAP 可以塞进来

    trend_label: str = "range"  # "up" / "down" / "range"

    # 数据质量与窗口信息
    last_candle_open_utc: Optional[datetime] = None
    last_candle_close_utc: Optional[datetime] = None
    is_last_candle_closed: bool = True
    bars_used: int = 0
    lookback_window: int = 0
    missing_bars_count: int = 0
    gap_list: List[Dict] = field(default_factory=list)

    # 价格基准与波动（量纲校验）
    price_last: Optional[float] = None
    price_mid: Optional[float] = None
    typical_price: Optional[float] = None
    return_last: Optional[float] = None
    atr_rel: Optional[float] = None
    tr_last: Optional[float] = None

    # 历史与形态辅助
    ma25_history: List[float] = field(default_factory=list)
    rsi6_history: List[float] = field(default_factory=list)
    recent_high: Optional[float] = None
    recent_low: Optional[float] = None

    # Optional debug window extremes
    high_last_n: Optional[float] = None
    low_last_n: Optional[float] = None

    # For LH "post OI spike small bodies" confirmation
    post_spike_small_body_count: Optional[int] = None


@dataclass
class DerivativeIndicators:
    """衍生品情绪相关指标：Funding / OI / 盘口等。"""
    funding: float                  # 当前 funding rate
    open_interest: float            # 当前 OI
    oi_change_24h: Optional[float]  # 最近24h OI 变化 %，拉取失败时为 None
    oi_change_pct: Optional[float] = None  # 近几根K的 OI 变化百分比
    mark_price: Optional[float] = None      # 最新标记价格
    orderbook_asks: List[Dict] = field(default_factory=list)  # 顶部卖单墙 [{price, size}, ...]
    orderbook_bids: List[Dict] = field(default_factory=list)  # 底部买单墙
    liquidity_comment: str = ""     # 对流动性的简单文字判断（可选）

    # Orderbook 墙体辅助信息
    ask_wall_size: float = 0.0
    bid_wall_size: float = 0.0
    ask_to_bid_ratio: Optional[float] = None
    has_large_ask_wall: bool = False
    has_large_bid_wall: bool = False


@dataclass
class MarketSnapshot:
    """多周期 + 衍生品指标准备好之后的完整输入。"""
    symbol: str
    ts: datetime
    tf_4h: TimeframeIndicators
    tf_1h: TimeframeIndicators
    tf_15m: TimeframeIndicators
    deriv: DerivativeIndicators
    regime: str = "unknown"
    regime_reason: str = ""
    rsidev: float = 0.0
    atrrel: float = 0.0
    rsi_15m: Optional[float] = None
    rsi_1h: Optional[float] = None
    asks: float = 0.0
    bids: float = 0.0

    def get_timeframe(self, tf: str) -> TimeframeIndicators:
        tf = tf.lower()
        if tf in {"4h", "4hour", "tf_4h"}:
            return self.tf_4h
        if tf in {"1h", "1hour", "tf_1h"}:
            return self.tf_1h
        if tf in {"15m", "15min", "tf_15m"}:
            return self.tf_15m
        raise ValueError(f"Unsupported timeframe: {tf}")
