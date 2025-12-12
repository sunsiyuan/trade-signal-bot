from dataclasses import dataclass, field
from typing import Literal, Optional, List, Dict
from datetime import datetime

Direction = Literal["long", "short", "none"]


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

    # 历史与形态辅助
    ma25_history: List[float] = field(default_factory=list)
    rsi6_history: List[float] = field(default_factory=list)
    recent_high: Optional[float] = None
    recent_low: Optional[float] = None
    high_last_n: List[float] = field(default_factory=list)
    low_last_n: List[float] = field(default_factory=list)
    post_spike_small_body_count: int = 0


@dataclass
class DerivativeIndicators:
    """衍生品情绪相关指标：Funding / OI / 盘口等。"""
    funding: float                  # 当前 funding rate
    open_interest: float            # 当前 OI
    oi_change_24h: Optional[float]  # 最近24h OI 变化 %，拉取失败时为 None
    oi_change_pct: Optional[float] = None  # 近几根K的 OI 变化百分比
    orderbook_asks: List[Dict] = field(default_factory=list)  # 顶部卖单墙 [{price, size}, ...]
    orderbook_bids: List[Dict] = field(default_factory=list)  # 底部买单墙
    liquidity_comment: str = ""     # 对流动性的简单文字判断（可选）

    # Orderbook 墙体辅助信息
    ask_wall_size: Optional[float] = None
    bid_wall_size: Optional[float] = None
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
    market_mode: str = "range"
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
