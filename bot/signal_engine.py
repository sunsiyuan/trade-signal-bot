# bot/signal_engine.py
from dataclasses import dataclass
from typing import Optional, List, Dict

from .config import Settings
from .models import MarketSnapshot, Direction


@dataclass
class TradeSignal:
    """最终要输出的执行方案。"""
    symbol: str
    direction: Direction            # "long" / "short" / "none"
    confidence: float               # 0~1，用来映射“胜率区间”
    reason: str                     # 文字解释，方便你人工 review

    # 点位
    entry: Optional[float] = None
    entry_range: Optional[List[float]] = None
    tp1: Optional[float] = None
    tp2: Optional[float] = None
    sl: Optional[float] = None

    # 仓位建议
    core_position_pct: float = 0.0
    add_position_pct: float = 0.0

    # 为了 debug，把关键指标也带出来（可选）
    snapshot: Optional[MarketSnapshot] = None
    debug_scores: Optional[Dict] = None


class SignalEngine:
    """把指标 → 交易决策 的大脑。"""

    def __init__(self, settings: Optional[Settings] = None):
        """Initialize the signal engine with optional overrides.

        A ``Settings`` instance remains the preferred input, but a plain
        dictionary can also be passed in tests to override thresholds.
        """
        self.settings = settings or {}
        self.min_confidence = self._get_setting("min_confidence", 0.3)

        # ATR 参数（可以外部配置）
        self.atr_sl_mult = self._get_setting("atr_sl_mult", 0.8)
        self.atr_tp1_mult = self._get_setting("atr_tp1_mult", 0.8)
        self.atr_tp2_mult = self._get_setting("atr_tp2_mult", 1.6)

    def _get_setting(self, name: str, default):
        if hasattr(self.settings, name):
            return getattr(self.settings, name)
        if isinstance(self.settings, dict):
            return self.settings.get(name, default)
        return default

    def _score_direction(self, snap: MarketSnapshot) -> Dict[str, float]:
        """对 long / short 两个方向分别打分，并给出 debug 信息。"""
        short_score = 0.0
        long_score = 0.0
        debug = []

        tf4h = snap.tf_4h
        tf1h = snap.tf_1h
        tf15 = snap.tf_15m
        deriv = snap.deriv

        # ---- 1. 大周期趋势 ---- #
        if tf4h.trend_label == "down":
            short_score += 0.35
            debug.append(("4h_trend_down", +0.35))
        elif tf4h.trend_label == "up":
            long_score += 0.35
            debug.append(("4h_trend_up", +0.35))

        # 1h 趋势作为次级确认
        if tf1h.trend_label == "down":
            short_score += 0.15
            debug.append(("1h_trend_down", +0.15))
        elif tf1h.trend_label == "up":
            long_score += 0.15
            debug.append(("1h_trend_up", +0.15))

        # ---- 2. 15m RSI 极端区域 ---- #
        if tf15.rsi6 >= 80:
            short_score += 0.25
            debug.append(("15m_rsi_overbought", +0.25))
        elif tf15.rsi6 <= 20:
            long_score += 0.25
            debug.append(("15m_rsi_oversold", +0.25))

        # 温和超买/超卖，给一点小分
        if 70 <= tf15.rsi6 < 80:
            short_score += 0.1
            debug.append(("15m_rsi_high", +0.1))
        if 20 < tf15.rsi6 <= 30:
            long_score += 0.1
            debug.append(("15m_rsi_low", +0.1))

        # ---- 3. 价格相对 MA99（视作阻力/支撑） ---- #
        if tf15.ma99 > 0:
            dist_to_ma99 = (tf15.close - tf15.ma99) / tf15.ma99
        else:
            dist_to_ma99 = 0.0

        # 价格在 MA99 上方且偏离不大 → 空头好位置
        if dist_to_ma99 > 0 and abs(dist_to_ma99) <= 0.02:
            short_score += 0.15
            debug.append(("price_near_resistance_ma99", +0.15))
        # 价格在 MA99 下方不远 → 多头好位置
        if dist_to_ma99 < 0 and abs(dist_to_ma99) <= 0.02:
            long_score += 0.15
            debug.append(("price_near_support_ma99", +0.15))

        # ---- 4. 流动性倾斜 ---- #
        liquidity = getattr(deriv, "liquidity_comment", "") or "balanced"
        if liquidity.startswith("asks"):
            liquidity = "asks>bids"
        elif liquidity.startswith("bids"):
            liquidity = "bids>asks"

        if liquidity == "asks>bids":
            short_score += 0.1
            debug.append(("orderbook_asks_heavier", +0.1))
        elif liquidity == "bids>asks":
            long_score += 0.1
            debug.append(("orderbook_bids_heavier", +0.1))

        # ---- 5. Funding（用百分比） ---- #
        funding_pct = (deriv.funding or 0.0) * 100.0

        # 正 funding 说明多头付费，多头 crowded，适度给空头加分
        if funding_pct > 0.02:
            short_score += 0.1
            debug.append(("funding_positive_crowded_longs", +0.1))
        elif funding_pct < -0.02:
            long_score += 0.1
            debug.append(("funding_negative_crowded_shorts", +0.1))

        # ---- 6. OI 变化：杠杆堆积方向 ---- #
        if deriv.oi_change_24h > 0.1:
            if tf15.rsi6 >= 80:
                short_score += 0.1
                debug.append(("oi_up_with_overbought", +0.1))
            if tf15.rsi6 <= 20:
                long_score += 0.1
                debug.append(("oi_up_with_oversold", +0.1))

        return {
            "short": short_score,
            "long": long_score,
            "debug": debug,
        }

    def generate_signal(self, snap: MarketSnapshot) -> TradeSignal:
        scores = self._score_direction(snap)
        short_score = scores["short"]
        long_score = scores["long"]
        debug = scores["debug"]

        tf15 = snap.tf_15m
        atr = max(tf15.atr, 1e-6)

        # ---- 高胜率做空：4h下跌 + 15m严重超买 + 卖墙更重 ---- #
        high_conf_short = (
            snap.tf_4h.trend_label == "down"
            and snap.tf_1h.trend_label != "up"
            and tf15.rsi6 >= 80
            and (getattr(snap.deriv, "liquidity_comment", "") or "").startswith(
                "asks"
            )
        )

        # ---- 高胜率做多：4h上升 + 15m严重超卖 + 买墙更重 ---- #
        high_conf_long = (
            snap.tf_4h.trend_label == "up"
            and snap.tf_1h.trend_label != "down"
            and tf15.rsi6 <= 20
            and (getattr(snap.deriv, "liquidity_comment", "") or "").startswith(
                "bids"
            )
        )

        direction: Direction = "none"
        confidence = 0.0
        reason = "多空得分不足，继续观望"
        entry = tp1 = tp2 = sl = None

        if high_conf_short:
            direction = "short"
            confidence = max(0.8, min(0.95, short_score))
            reason = (
                "4h 下跌趋势 + 15m 严重超买 + 上方卖盘更重，"
                "典型反弹尾段做空窗口"
            )

        elif high_conf_long:
            direction = "long"
            confidence = max(0.8, min(0.95, long_score))
            reason = (
                "4h 上升趋势 + 15m 严重超卖 + 下方买盘更重，"
                "典型回调尾段做多窗口"
            )

        else:
            best_score = max(short_score, long_score)
            if best_score < self.min_confidence:
                return TradeSignal(
                    symbol=snap.symbol,
                    direction="none",
                    confidence=best_score,
                    reason=(
                        f"多空得分不足（long={long_score:.2f}, "
                        f"short={short_score:.2f}），继续观望"
                    ),
                    snapshot=snap,
                    debug_scores=scores,
                )
            if short_score > long_score:
                direction = "short"
                confidence = best_score
                reason = (
                    f"空头得分更高（short={short_score:.2f} > "
                    f"long={long_score:.2f}），偏空执行"
                )
            else:
                direction = "long"
                confidence = best_score
                reason = (
                    f"多头得分更高（long={long_score:.2f} > "
                    f"short={short_score:.2f}），偏多执行"
                )

        entry = tf15.close

        if direction == "short":
            sl = entry + atr * self.atr_sl_mult
            tp1 = entry - atr * self.atr_tp1_mult
            tp2 = entry - atr * self.atr_tp2_mult
        elif direction == "long":
            sl = entry - atr * self.atr_sl_mult
            tp1 = entry + atr * self.atr_tp1_mult
            tp2 = entry + atr * self.atr_tp2_mult

        core_pct = 0.7 if direction in ("long", "short") else 0.0
        add_pct = 0.3 if direction in ("long", "short") else 0.0

        return TradeSignal(
            symbol=snap.symbol,
            direction=direction,
            confidence=round(confidence, 2),
            reason=reason,
            entry=entry,
            tp1=tp1,
            tp2=tp2,
            sl=sl,
            core_position_pct=core_pct,
            add_position_pct=add_pct,
            snapshot=snap,
            debug_scores=scores,
        )
