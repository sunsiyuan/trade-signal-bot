# =========================
# signal_engine.py
# =========================

# 标准库：数据类 & 类型标注
from dataclasses import dataclass
from typing import Optional, List, Dict

# 项目内依赖
from .config import Settings                     # 全局配置（阈值、策略参数等）
from .models import (
    ConditionalPlan,
    ExecutionIntent,
    MarketSnapshot,
    Direction,
)
from .conditional_plan import (
    build_conditional_plan_from_intent,
    resolve_atr_4h,
)
from .regime_detector import detect_regime, RegimeSignal
from .strategy_trend_following import (
    build_execution_intent_tf,
    build_trend_following_signal,
)


# =========================
# TradeSignal：最终输出给“执行层 / Telegram / 人工 review”的对象
# =========================
@dataclass
class TradeSignal:
    """最终要输出的执行方案（单一币种、单一方向）。"""

    # ---- 基本信息 ----
    symbol: str
    direction: Direction            # "long" / "short" / "none"

    # ---- 解释 & 置信度 ----
    reason: str = ""                # 人类可读的解释
    trade_confidence: float = 0.0   # 是否值得“采取行动”（胜率感）
    edge_confidence: float = 0.0    # 当前是否处在“好位置 / 边缘”（机会强度）

    # ---- 策略类型标识 ----
    setup_type: str = "none"        # trend_long / range_short / none 等

    # ---- 交易点位 ----
    entry: Optional[float] = None
    tp1: Optional[float] = None
    tp2: Optional[float] = None
    tp3: Optional[float] = None
    sl: Optional[float] = None

    # ---- 仓位建议 ----
    core_position_pct: float = 0.0  # 核心仓位
    add_position_pct: float = 0.0   # 加仓仓位

    # ---- Debug / Explainability ----
    snapshot: Optional[MarketSnapshot] = None
    debug_scores: Optional[Dict] = None
    rejected_reasons: Optional[List[str]] = None
    thresholds_snapshot: Optional[Dict] = None

    # ---- 条件单计划 ----
    execution_intent: Optional[ExecutionIntent] = None
    conditional_plan: Optional[ConditionalPlan] = None
    conditional_plan_debug: Optional[Dict] = None


# =========================
# SignalEngine：整个系统的“中枢大脑”
# =========================
class SignalEngine:
    """把 MarketSnapshot + Regime → TradeSignal 的决策引擎。"""

    def __init__(self, settings: Optional[Settings] = None):
        """
        settings:
        - 推荐传 Settings 实例
        - 也允许 dict（方便测试 / 临时 override）
        """
        self.settings = settings or {}
        self.min_confidence = self._get_setting("min_confidence", 0.3)

    # -------------------------
    # 通用 settings 读取
    # -------------------------
    def _get_setting(self, name: str, default):
        """
        统一读取 settings：
        - Settings.xxx
        - 或 dict["xxx"]
        """
        if hasattr(self.settings, name):
            return getattr(self.settings, name)
        if isinstance(self.settings, dict):
            return self.settings.get(name, default)
        return default

    def _build_execution_intent(
        self, signal: TradeSignal, snap: MarketSnapshot, regime_signal: RegimeSignal
    ) -> ExecutionIntent:
        if signal is None:
            return ExecutionIntent(
                symbol=snap.symbol,
                direction="none",
                entry_price=None,
                entry_reason="no_signal",
                invalidation_price=None,
                atr_4h=resolve_atr_4h(snap),
                reason="",
                debug=None,
            )
        if getattr(signal, "execution_intent", None):
            return signal.execution_intent  # type: ignore[return-value]

        setup_type = getattr(signal, "setup_type", "") or ""
        if regime_signal.regime == "trending":
            return build_execution_intent_tf(snap, regime_signal, signal)

        if regime_signal.regime in ("high_vol_ranging", "low_vol_ranging"):
            if setup_type.startswith("lh"):
                from .strategy_liquidity_hunt import build_execution_intent_lh

                return build_execution_intent_lh(snap, signal)

            if setup_type.startswith("mr"):
                from .strategy_mean_reversion import build_execution_intent_mr

                return build_execution_intent_mr(snap, signal)

        return ExecutionIntent(
            symbol=snap.symbol,
            direction=signal.direction,
            entry_price=getattr(signal, "entry", None),
            entry_reason=setup_type or "unknown",
            invalidation_price=signal.sl if hasattr(signal, "sl") else None,
            atr_4h=resolve_atr_4h(snap),
            reason=signal.reason,
            debug=signal.debug_scores,
        )

    # =========================
    # 核心入口：根据 regime 分流
    # =========================
    def decide(self, snap: MarketSnapshot, regime_signal: RegimeSignal) -> TradeSignal:
        regime = regime_signal.regime

        # 把 regime 写回 snapshot（方便下游 debug / 输出）
        snap.regime = regime
        snap.regime_reason = regime_signal.reason

        if regime == "trending":
            # 趋势行情 → 趋势跟随策略
            return build_trend_following_signal(
                snap,
                regime_signal,
                min_confidence=self.min_confidence,
                settings=self.settings,
            )

        if regime in ("high_vol_ranging", "low_vol_ranging"):
            # 震荡行情 → range router
            return self._decide_range(snap)

        # 兜底：未知 regime 也按趋势处理（防止系统无输出）
        return build_trend_following_signal(
            snap,
            regime_signal,
            min_confidence=self.min_confidence,
            settings=self.settings,
        )

    # =========================
    # Range router：LH / MR
    # =========================
    def _decide_range(self, snap: MarketSnapshot) -> TradeSignal:
        from .strategy_liquidity_hunt import build_liquidity_hunt_signal
        from .strategy_mean_reversion import build_mean_reversion_signal

        regime = getattr(snap, "regime", None)

        # --- 高波动震荡：优先 LH ---
        if regime == "high_vol_ranging":
            lh = build_liquidity_hunt_signal(snap, regime, self.settings)
            if lh:
                return lh

        # --- 所有震荡：MR ---
        if regime in ("high_vol_ranging", "low_vol_ranging"):
            mr = build_mean_reversion_signal(snap, regime, self.settings)
            if mr:
                return mr

        # --- 无策略命中 ---
        return TradeSignal(
            symbol=snap.symbol,
            direction="none",
            trade_confidence=0.0,
            edge_confidence=0.5,
            setup_type="none",
            reason="Range regime but no LH/MR trigger",
            snapshot=snap,
            rejected_reasons=["no_range_strategy_triggered"],
            debug_scores={"long": 0.0, "short": 0.0},
        )

    # =========================
    # 对外主入口
    # =========================
    def generate_signal(self, snap: MarketSnapshot) -> TradeSignal:
        """
        一次完整信号生成流程：
        detect_regime → decide → conditional_plan → reason 整理
        """
        regime_signal = detect_regime(snap, self.settings)

        snap.regime = regime_signal.regime
        snap.regime_reason = regime_signal.reason

        signal = self.decide(snap, regime_signal)

        intent = self._build_execution_intent(signal, snap, regime_signal)
        signal.execution_intent = intent
        signal.conditional_plan = build_conditional_plan_from_intent(intent, snap)

        # --- reason 统一前缀 ---
        if signal and signal.reason:
            regime_label = snap.regime or "unknown"
            regime_reason = snap.regime_reason or ""

            if regime_reason:
                signal.reason = f"[{regime_label}] {signal.reason} | {regime_reason}"
            else:
                signal.reason = f"[{regime_label}] {signal.reason}"

        return signal
