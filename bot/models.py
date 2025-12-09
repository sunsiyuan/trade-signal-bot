from dataclasses import dataclass
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


@dataclass
class DerivativeIndicators:
    """衍生品情绪相关指标：Funding / OI / 盘口等。"""
    funding: float                  # 当前 funding rate
    open_interest: float            # 当前 OI
    oi_change_24h: float            # 最近24h OI 变化 %
    orderbook_asks: List[Dict]      # 顶部卖单墙 [{price, size}, ...]
    orderbook_bids: List[Dict]      # 底部买单墙
    liquidity_comment: str = ""     # 对流动性的简单文字判断（可选）


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
