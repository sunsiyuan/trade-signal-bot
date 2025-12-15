# =========================
# signal_engine.py
# =========================

# 标准库：数据类 & 类型标注
from dataclasses import dataclass
from typing import Optional, List, Dict

# 项目内依赖
from .config import Settings                     # 全局配置（阈值、策略参数等）
from .models import MarketSnapshot, Direction    # 市场快照 & 方向枚举
from .conditional_plan import build_conditional_plan
from .regime_detector import detect_regime, RegimeSignal
from .strategy_trend_following import build_trend_following_signal


# -------------------------
# 工具函数：数值裁剪
# -------------------------
def clamp(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    """
    将 value 裁剪到 [min_value, max_value] 区间内
    常用于把各种指标归一化到 0~1
    """
    return max(min_value, min(max_value, value))


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
    conditional_plan: Optional[Dict] = None
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

    # -------------------------
    # regime 子配置读取
    # -------------------------
    def _get_regime_setting(self, name: str, default):
        """
        专门读取 settings.regime.xxx
        """
        if hasattr(self.settings, "regime"):
            container = getattr(self.settings, "regime")
            if isinstance(container, dict):
                return container.get(name, default)
            if hasattr(container, name):
                return getattr(container, name)
        if isinstance(self.settings, dict):
            regime_settings = self.settings.get("regime")
            if isinstance(regime_settings, dict):
                return regime_settings.get(name, default)
        return default

    # -------------------------
    # regime 覆写逻辑（弱趋势 → 震荡）
    # -------------------------
    def _apply_regime_override(self, regime_signal: RegimeSignal) -> RegimeSignal:
        """
        如果 detect_regime 判为 trending，
        但趋势强度不足，则强制降级为 high_vol_ranging
        """
        if regime_signal.regime != "trending":
            return regime_signal

        min_trend_ma_angle = self._get_regime_setting(
            "min_trend_ma_angle", 0.001
        )
        max_trend_osc = self._get_regime_setting("max_trend_osc", 1)

        weak_trend = (
            abs(regime_signal.ma_angle) < min_trend_ma_angle
            or regime_signal.osc_count <= max_trend_osc
        )

        if weak_trend:
            return RegimeSignal(
                regime="high_vol_ranging",
                reason=regime_signal.reason,
                ma_angle=regime_signal.ma_angle,
                atr_rel=regime_signal.atr_rel,
                rsi_avg_dev=regime_signal.rsi_avg_dev,
                osc_count=regime_signal.osc_count,
            )

        return regime_signal

    # =========================
    # 核心入口：根据 regime 分流
    # =========================
    def decide(self, snap: MarketSnapshot, regime_signal: RegimeSignal) -> TradeSignal:
        regime_signal = self._apply_regime_override(regime_signal)
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
    # Range regime：机会评分器
    # =========================
    def _range_setup_score(self, snap: MarketSnapshot) -> Dict[str, float]:
        """
        只做“机会强度评估”，不直接下单
        """

        # --- 1h RSI 偏离 50：判断是否靠近区间边缘 ---
        rsi1h = snap.rsi_1h if snap.rsi_1h is not None else snap.tf_1h.rsi6
        abs_dev = abs(rsi1h - 50.0)
        edge = clamp((abs_dev - 5.0) / (15.0 - 5.0), 0.0, 1.0)

        # --- 15m RSI：短周期超买 / 超卖 ---
        rsi15 = snap.rsi_15m if snap.rsi_15m is not None else snap.tf_15m.rsi6
        oversold = clamp((35 - rsi15) / 15, 0.0, 1.0)
        overbought = clamp((rsi15 - 65) / 15, 0.0, 1.0)

        # --- 1h RSI 二次确认 ---
        confirm_long = clamp((45 - rsi1h) / 15, 0.0, 1.0)
        confirm_short = clamp((rsi1h - 55) / 15, 0.0, 1.0)

        # --- 波动率（ATR relative） ---
        if snap.atrrel is None:
            atrrel = None
            tape = 0.5                # 缺失时给中性，不偏多
            atrrel_missing = True
            tape_reason = "atrrel_missing_fallback"
        else:
            atrrel = snap.atrrel
            tape = clamp((0.02 - atrrel) / (0.02 - 0.008), 0.0, 1.0)
            atrrel_missing = False
            tape_reason = "atrrel_based"

        # --- 订单簿倾斜 ---
        asks = snap.asks if snap.asks is not None else 0.0
        bids = snap.bids if snap.bids is not None else 0.0
        ob = clamp((bids - asks) / (bids + asks + 1e-9), -1.0, 1.0)
        ob_long = clamp(ob, 0.0, 1.0)
        ob_short = clamp(-ob, 0.0, 1.0)

        # --- 中位惩罚（靠近中线 = 不做） ---
        mid_penalty = 1 - edge

        # --- 综合评分 ---
        long_score = edge * (
            0.45 * oversold +
            0.20 * confirm_long +
            0.20 * tape +
            0.15 * ob_long
        )
        short_score = edge * (
            0.45 * overbought +
            0.20 * confirm_short +
            0.20 * tape +
            0.15 * ob_short
        )

        return {
            "edge": round(edge, 4),
            "long": round(long_score, 4),
            "short": round(short_score, 4),
            "mid_penalty": round(mid_penalty, 4),
            "abs_dev": round(abs_dev, 4),
            "tape": round(tape, 4),
            "ob": round(ob, 4),
            "atrrel": atrrel,
            "atrrel_missing": atrrel_missing,
            "tape_reason": tape_reason,
        }

    # =========================
    # Range router：LH / MR
    # =========================
    def _decide_range(self, snap: MarketSnapshot) -> TradeSignal:
        from .strategy_liquidity_hunt import build_liquidity_hunt_signal
        from .strategy_mean_reversion import build_mean_reversion_signal

        scores = self._range_setup_score(snap)
        edge = scores["edge"]
        best = max(scores["long"], scores["short"])
        regime = getattr(snap, "regime", None)

        thresholds = {"edge_min": 0.35, "best_min": 0.55}

        # --- 高波动震荡：优先 LH ---
        if regime == "high_vol_ranging":
            lh = build_liquidity_hunt_signal(snap, regime, self.settings)
            if lh:
                lh.edge_confidence = max(edge, getattr(lh, "edge_confidence", 0.0), best)
                lh.debug_scores = scores
                lh.thresholds_snapshot = thresholds
                return lh

        # --- 所有震荡：MR ---
        if regime in ("high_vol_ranging", "low_vol_ranging"):
            mr = build_mean_reversion_signal(snap, regime, self.settings)
            if mr:
                mr.edge_confidence = max(edge, getattr(mr, "edge_confidence", 0.0), best)
                mr.debug_scores = scores
                mr.thresholds_snapshot = thresholds
                return mr

        # --- 无策略命中 ---
        return TradeSignal(
            symbol=snap.symbol,
            direction="none",
            trade_confidence=0.0,
            edge_confidence=max(edge, best),
            setup_type="none",
            reason=(
                "Range regime but no LH/MR trigger | "
                f"edge={edge:.2f} best={best:.2f}"
            ),
            snapshot=snap,
            debug_scores=scores,
            rejected_reasons=["no_range_strategy_triggered"],
            thresholds_snapshot=thresholds,
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

        # --- 条件单（分批、回踩、突破） ---
        conditional_plan = build_conditional_plan(signal, snap, self.settings)
        if conditional_plan:
            signal.conditional_plan = conditional_plan

        # --- reason 统一前缀 ---
        if signal and signal.reason:
            regime_label = snap.regime or "unknown"
            regime_reason = snap.regime_reason or ""

            if regime_reason:
                signal.reason = f"[{regime_label}] {signal.reason} | {regime_reason}"
            else:
                signal.reason = f"[{regime_label}] {signal.reason}"

        return signal
