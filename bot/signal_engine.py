# bot/signal_engine.py
from dataclasses import dataclass
from typing import Optional, List, Dict

from .config import Settings
from .models import MarketSnapshot, Direction
from .regime_detector import detect_regime, RegimeSignal
from .strategy_trend_following import build_trend_following_signal


def clamp(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    return max(min_value, min(max_value, value))


@dataclass
class TradeSignal:
    """最终要输出的执行方案。"""
    symbol: str
    direction: Direction            # "long" / "short" / "none"
    reason: str = ""                # 文字解释，方便你人工 review
    confidence: float = 0.0         # 兼容旧字段，映射到 trade_confidence
    trade_confidence: float = 0.0   # 0~1，用来映射“胜率区间”
    edge_confidence: float = 0.0    # 0~1，区间边缘机会强度
    setup_type: str = "none"        # "trend_long" / "trend_short" / "range_long" / "range_short" / "none"

    # 点位
    entry: Optional[float] = None
    entry_range: Optional[List[float]] = None
    tp1: Optional[float] = None
    tp2: Optional[float] = None
    tp3: Optional[float] = None
    sl: Optional[float] = None

    # 仓位建议
    core_position_pct: float = 0.0
    add_position_pct: float = 0.0

    # 为了 debug，把关键指标也带出来（可选）
    snapshot: Optional[MarketSnapshot] = None
    debug_scores: Optional[Dict] = None
    rejected_reasons: Optional[List[str]] = None
    thresholds_snapshot: Optional[Dict] = None

    def __post_init__(self):
        # 保持与旧 confidence 字段的兼容性
        if self.trade_confidence == 0 and self.confidence:
            self.trade_confidence = self.confidence
        if self.confidence == 0 and self.trade_confidence:
            self.confidence = self.trade_confidence


class SignalEngine:
    """把指标 → 交易决策 的大脑。"""

    def __init__(self, settings: Optional[Settings] = None):
        """Initialize the signal engine with optional overrides.

        A ``Settings`` instance remains the preferred input, but a plain
        dictionary can also be passed in tests to override thresholds.
        """
        self.settings = settings or {}
        self.min_confidence = self._get_setting("min_confidence", 0.3)

    def _get_setting(self, name: str, default):
        if hasattr(self.settings, name):
            return getattr(self.settings, name)
        if isinstance(self.settings, dict):
            return self.settings.get(name, default)
        return default

    def decide(self, snap: MarketSnapshot, regime_signal: RegimeSignal) -> TradeSignal:
        regime = regime_signal.regime

        # override: if regime says trending but structure is flat, route to ranging
        if regime == "trending" and (
            abs(getattr(snap, "maangle", 0.0)) < 1e-4 or getattr(snap, "osc", 0) == 0
        ):
            regime = "high_vol_ranging"

        snap.regime = regime

        if regime == "trending":
            return build_trend_following_signal(
                snap, regime_signal, min_confidence=self.min_confidence
            )
        if regime in ("high_vol_ranging", "low_vol_ranging"):
            return self._decide_range(snap)
        return build_trend_following_signal(
            snap, regime_signal, min_confidence=self.min_confidence
        )

    def _range_setup_score(self, snap: MarketSnapshot) -> Dict[str, float]:
        # edge proxy: distance from midline (RSI=50) on 1h
        rsi1h = snap.rsi_1h if snap.rsi_1h is not None else snap.tf_1h.rsi6
        abs_dev = abs(rsi1h - 50.0)
        edge = clamp((abs_dev - 5.0) / (15.0 - 5.0), 0.0, 1.0)

        rsi15 = snap.rsi_15m if snap.rsi_15m is not None else snap.tf_15m.rsi6

        oversold = clamp((35 - rsi15) / 15, 0.0, 1.0)
        overbought = clamp((rsi15 - 65) / 15, 0.0, 1.0)
        confirm_long = clamp((45 - rsi1h) / 15, 0.0, 1.0)
        confirm_short = clamp((rsi1h - 55) / 15, 0.0, 1.0)

        atrrel = snap.atrrel if snap.atrrel is not None else 0.0
        tape = clamp((0.02 - atrrel) / (0.02 - 0.008), 0.0, 1.0)

        asks = snap.asks if snap.asks is not None else 0.0
        bids = snap.bids if snap.bids is not None else 0.0
        ob = clamp((bids - asks) / (bids + asks + 1e-9), -1.0, 1.0)
        ob_long = clamp(ob, 0.0, 1.0)
        ob_short = clamp(-ob, 0.0, 1.0)

        mid_penalty = 1 - edge

        long_score = edge * (
            0.45 * oversold + 0.20 * confirm_long + 0.20 * tape + 0.15 * ob_long
        )
        short_score = edge * (
            0.45 * overbought + 0.20 * confirm_short + 0.20 * tape + 0.15 * ob_short
        )

        return {
            "edge": round(edge, 4),
            "long": round(long_score, 4),
            "short": round(short_score, 4),
            "mid_penalty": round(mid_penalty, 4),
            "abs_dev": round(abs_dev, 4),
            "tape": round(tape, 4),
            "ob": round(ob, 4),
        }

    def _decide_range(self, snap: MarketSnapshot) -> TradeSignal:
        from .strategy_liquidity_hunt import build_liquidity_hunt_signal
        from .strategy_mean_reversion import build_mean_reversion_signal

        scores = self._range_setup_score(snap)
        edge = scores["edge"]
        best = max(scores["long"], scores["short"])
        regime = getattr(snap, "regime", None)
        thresholds = {"edge_min": 0.35, "best_min": 0.55}

        if regime == "high_vol_ranging":
            lh = build_liquidity_hunt_signal(snap, regime, self.settings)
            if lh:
                lh.edge_confidence = max(edge, getattr(lh, "edge_confidence", 0.0), best)
                lh.debug_scores = scores
                lh.thresholds_snapshot = thresholds
                return lh

        if regime in ("high_vol_ranging", "low_vol_ranging"):
            mr = build_mean_reversion_signal(snap, regime, self.settings)
            if mr:
                mr.edge_confidence = max(edge, getattr(mr, "edge_confidence", 0.0), best)
                mr.debug_scores = scores
                mr.thresholds_snapshot = thresholds
                return mr

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

    def generate_signal(self, snap: MarketSnapshot) -> TradeSignal:
        regime_signal = detect_regime(snap, self.settings)
        snap.regime = regime_signal.regime
        snap.regime_reason = regime_signal.reason

        signal = self.decide(snap, regime_signal)

        if signal and signal.reason:
            reason_prefix = "unknown"
            if snap.regime in ("high_vol_ranging", "low_vol_ranging"):
                reason_prefix = "range"
            elif snap.regime == "trending":
                reason_prefix = "trending"
            signal.reason = (
                f"[{reason_prefix}] {signal.reason} | regime={snap.regime} | {regime_signal.reason}"
            )

        return signal
