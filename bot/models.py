from dataclasses import dataclass, field
from typing import Literal, Optional, List, Dict
from datetime import datetime

# 统一方向枚举，避免字符串漂移
Direction = Literal["long", "short", "none"]


# ============================================================
# ExecutionIntent
# ============================================================
@dataclass
class ExecutionIntent:
    """
    「策略层 → 执行层」的最小执行意图描述。
    核心问题：如果要交易，这一单“应该怎么下”？
    """
    symbol: str
    direction: Direction                 # 方向：多 / 空 / 不交易

    # --- 理想入场（不是必须立即成交） ---
    entry_price: Optional[float]         # 策略认为的“理想价位”
    entry_reason: str                    # 入场触发来源（用于解释/调试）

    # --- 结构性失效 ---
    invalidation_price: Optional[float]  # 结构破坏位（通常等价于 SL）

    # --- 执行约束 ---
    ttl_hours: int = 4                   # 执行窗口（与 LIMIT_4H 对齐）
    allow_execute_now: bool = True       # 是否允许当前价直接成交

    # --- 风控引用（只用于 gate，不反向修改结构） ---
    atr_4h: Optional[float] = None

    # --- Debug / 可读性 ---
    reason: str = ""
    debug: Optional[Dict] = None


# ============================================================
# ConditionalPlan
# ============================================================
@dataclass
class ConditionalPlan:
    """
    执行层的「可落地计划」：
    ExecutionIntent 被解析后，转成具体执行模式。
    """
    execution_mode: Literal[
        "EXECUTE_NOW",     # 立即执行
        "PLACE_LIMIT_4H",  # 放 4h 条件单
        "WATCH_ONLY",      # 仅观察，不执行
    ]

    direction: Direction
    entry_price: Optional[float]
    valid_until_utc: Optional[str]       # 条件单/计划失效时间
    cancel_if: Dict[str, bool]            # 失效条件（如结构破坏、regime 变化）
    explain: str                          # 给人的一句话解释
    debug: Optional[Dict] = None


# ============================================================
# TimeframeIndicators
# ============================================================
@dataclass
class TimeframeIndicators:
    """
    单一时间周期（4H / 1H / 15m）的完整技术快照。
    用于：
    - regime 判断
    - setup 打分
    - debug / 输出展示
    """
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
    macd_hist: float            # Histogram

    atr: float                  # 当前周期 ATR
    volume: float               # 最新一根 K 的成交量
    vwap: Optional[float] = None

    trend_label: str = "range"  # 该周期自身的趋势标签（不等于全局 regime）

    # --- 数据质量 / 时间窗口 ---
    last_candle_open_utc: Optional[datetime] = None
    last_candle_close_utc: Optional[datetime] = None
    is_last_candle_closed: bool = True
    bars_used: int = 0
    lookback_window: int = 0
    missing_bars_count: int = 0
    gap_list: List[Dict] = field(default_factory=list)

    # --- 价格尺度 / 波动归一 ---
    price_last: Optional[float] = None
    price_mid: Optional[float] = None
    typical_price: Optional[float] = None
    return_last: Optional[float] = None
    atr_rel: Optional[float] = None
    tr_last: Optional[float] = None

    # --- 形态 / 历史辅助 ---
    ma25_history: List[float] = field(default_factory=list)
    rsi6_history: List[float] = field(default_factory=list)
    recent_high: Optional[float] = None
    recent_low: Optional[float] = None

    # --- Debug 用极值 ---
    high_last_n: Optional[float] = None
    low_last_n: Optional[float] = None

    # --- LH / 流动性形态辅助 ---
    post_spike_small_body_count: Optional[int] = None


# ============================================================
# DerivativeIndicators
# ============================================================
@dataclass
class DerivativeIndicators:
    """
    衍生品 / 情绪维度输入。
    这些字段通常用于：
    - gate（是否允许进场）
    - LH / squeeze / OI 相关策略
    """
    funding: float
    open_interest: float
    oi_change_24h: Optional[float]        # 可能缺失，因此 Optional
    mark_price: Optional[float] = None

    # 盘口简化结构（只保留“墙”级别信息）
    orderbook_asks: List[Dict] = field(default_factory=list)
    orderbook_bids: List[Dict] = field(default_factory=list)
    liquidity_comment: str = ""

    # 派生的盘口摘要
    ask_wall_size: Optional[float] = 0.0
    bid_wall_size: Optional[float] = 0.0
    ask_to_bid_ratio: Optional[float] = None
    has_large_ask_wall: Optional[bool] = False
    has_large_bid_wall: Optional[bool] = False


# ============================================================
# MarketSnapshot
# ============================================================
@dataclass
class MarketSnapshot:
    """
    系统中最核心的「只读输入对象」。
    特点：
    - 聚合多周期 + 衍生品
    - 策略层不应修改它，只基于它做判断
    """
    symbol: str
    ts: datetime

    tf_4h: TimeframeIndicators
    tf_1h: TimeframeIndicators
    tf_15m: TimeframeIndicators
    deriv: DerivativeIndicators

    # --- 全局市场状态 ---
    regime: str = "unknown"
    regime_reason: str = ""

    # --- 跨周期聚合指标（便于策略直接用） ---
    rsidev: float = 0.0
    atrrel: float = 0.0
    rsi_15m: Optional[float] = None
    rsi_1h: Optional[float] = None

    # --- Rolling regime candidate（不替代正式 regime） ---
    rolling_candidate: Optional[str] = None  # 'trending' | 'ranging' | None
    rolling_candidate_dir: Optional[str] = None  # 'up' | 'down' | None
    rolling_candidate_streak: int = 0

    # --- 盘口汇总 ---
    asks: float = 0.0
    bids: float = 0.0

    # --- 数据覆盖标记（用于回测缺失降级） ---
    data_flags: Optional[Dict[str, bool]] = None

    def get_timeframe(self, tf: str) -> TimeframeIndicators:
        """
        统一的 timeframe 访问器，避免上游到处 if/else。
        """
        tf = tf.lower()
        if tf in {"4h", "4hour", "tf_4h"}:
            return self.tf_4h
        if tf in {"1h", "1hour", "tf_1h"}:
            return self.tf_1h
        if tf in {"15m", "15min", "tf_15m"}:
            return self.tf_15m
        raise ValueError(f"Unsupported timeframe: {tf}")
