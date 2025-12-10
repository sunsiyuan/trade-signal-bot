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
    tp3: Optional[float] = None
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

        # ---- 6. 波动 & 流动性异常 ---- #
        # 利用 15m vs 1h ATR 比例 + 价格偏离 + orderbook 倾斜
        atr_ratio = tf15.atr / max(tf1h.atr, 1e-6)
        price_to_ma25 = abs(tf15.close - tf15.ma25) / max(tf15.ma25, 1e-6)

        # 认为是“短周期明显放量波动”的阈值
        is_vol_spike = atr_ratio > 1.3 and price_to_ma25 > 0.007

        if is_vol_spike:
            # 如果盘口卖单明显更重，且价格在均线之上 → 偏空
            if liquidity == "asks>bids" and tf15.close >= tf15.ma25:
                short_score += 0.1
                debug.append(("vol_spike_with_heavy_asks", +0.1))

            # 如果盘口买单明显更重，且价格在均线之下 → 偏多
            if liquidity == "bids>asks" and tf15.close <= tf15.ma25:
                long_score += 0.1
                debug.append(("vol_spike_with_heavy_bids", +0.1))

        return {
            "short": short_score,
            "long": long_score,
            "debug": debug,
        }

    def _detect_market_mode(self, snap: MarketSnapshot) -> str:
        """基于 ATR 扩张与趋势标签做出行情分类。"""
        tf15 = snap.tf_15m
        tf1h = snap.tf_1h
        tf4h = snap.tf_4h

        atr_ratio = tf15.atr / max(tf1h.atr, 1e-6)
        price_to_ma25 = abs(tf15.close - tf15.ma25) / max(tf15.ma25, 1e-6)

        if atr_ratio > 1.2 and price_to_ma25 > 0.005:
            return "breakout"

        if tf15.rsi6 >= 78 or tf15.rsi6 <= 22:
            if (tf4h.trend_label == "down" and tf15.rsi6 >= 78) or (
                tf4h.trend_label == "up" and tf15.rsi6 <= 22
            ):
                return "sfp"

        if atr_ratio < 0.85 and tf4h.trend_label == "range":
            return "squeeze"

        if tf4h.trend_label == "range" and tf1h.trend_label == "range":
            return "range"

        return "trend"

    def _high_conf_short(self, snap: MarketSnapshot) -> bool:
        tf15 = snap.tf_15m
        return (
            snap.tf_4h.trend_label == "down"
            and snap.tf_1h.trend_label != "up"
            and tf15.rsi6 >= 80
            and (getattr(snap.deriv, "liquidity_comment", "") or "").startswith(
                "asks"
            )
        )

    def _high_conf_long(self, snap: MarketSnapshot) -> bool:
        tf15 = snap.tf_15m
        return (
            snap.tf_4h.trend_label == "up"
            and snap.tf_1h.trend_label != "down"
            and tf15.rsi6 <= 20
            and (getattr(snap.deriv, "liquidity_comment", "") or "").startswith(
                "bids"
            )
        )

    def _compute_trigger(self, entry: float, atr: float, direction: Direction) -> float:
        if direction == "short":
            return entry - 0.3 * atr
        return entry + 0.3 * atr

    def _compute_swing_sl(self, snap: MarketSnapshot, atr: float, direction: Direction) -> float:
        tf15 = snap.tf_15m
        tf1h = snap.tf_1h

        buffer = 0.2 * atr
        if direction == "short":
            swing_high = max(tf15.ma7, tf15.ma25, tf1h.ma25, tf1h.ma7)
            return swing_high + buffer

        swing_low = min(tf15.ma7, tf15.ma25, tf1h.ma25, tf1h.ma7)
        return swing_low - buffer

    def _compute_position(self, confidence: float, regime: str) -> Dict[str, float]:
        if regime == "range":
            if confidence >= 0.8:
                return {"core": 0.2, "add": 0.1}
            return {"core": 0.0, "add": 0.0}

        if confidence >= 0.85:
            return {"core": 0.7, "add": 0.3}
        if 0.55 <= confidence < 0.85:
            return {"core": 0.4, "add": 0.2}
        return {"core": 0.0, "add": 0.0}

    def generate_signal(self, snap: MarketSnapshot) -> TradeSignal:
        scores = self._score_direction(snap)
        short_score = scores["short"]
        long_score = scores["long"]
        debug = scores["debug"]

        bias: Direction = "short" if short_score > long_score else "long"
        confidence = max(short_score, long_score)

        if confidence < self.min_confidence:
            return TradeSignal(
                symbol=snap.symbol,
                direction="none",
                confidence=round(confidence, 2),
                reason=(
                    f"多空得分不足（long={long_score:.2f}, "
                    f"short={short_score:.2f}），继续观望"
                ),
                snapshot=snap,
                debug_scores=scores,
            )

        regime = self._detect_market_mode(snap)
        snap.market_mode = regime

        if regime != "trend" and confidence < 0.8:
            return TradeSignal(
                symbol=snap.symbol,
                direction="none",
                confidence=round(confidence, 2),
                reason=f"行情模式为 {regime}，信号强度 {confidence:.2f} 不足以出手",
                snapshot=snap,
                debug_scores=scores,
            )

        if bias == "short" and not self._high_conf_short(snap):
            return TradeSignal(
                symbol=snap.symbol,
                direction="none",
                confidence=round(confidence, 2),
                reason="做空条件未满足高胜率模板，等待更好的 setup",
                snapshot=snap,
                debug_scores=scores,
            )

        if bias == "long" and not self._high_conf_long(snap):
            return TradeSignal(
                symbol=snap.symbol,
                direction="none",
                confidence=round(confidence, 2),
                reason="做多条件未满足高胜率模板，等待更好的 setup",
                snapshot=snap,
                debug_scores=scores,
            )

        tf15 = snap.tf_15m
        atr = max(tf15.atr, 1e-6)
        spot_price = tf15.close
        trigger = self._compute_trigger(spot_price, atr, bias)

        price_crossed = (spot_price <= trigger) if bias == "short" else (spot_price >= trigger)
        if not price_crossed:
            return TradeSignal(
                symbol=snap.symbol,
                direction="none",
                confidence=round(confidence, 2),
                reason=(
                    f"信号待确认：等待价格触发 {trigger:.2f} 以执行 {bias} 入场"
                ),
                entry=trigger,
                snapshot=snap,
                debug_scores=scores,
            )

        entry = trigger
        sl = self._compute_swing_sl(snap, atr, bias)
        R = abs(sl - entry)
        if R < atr * 0.25:
            R = atr * 0.25

        if bias == "short":
            tp1 = entry - 0.5 * R
            tp2 = entry - 1.0 * R
            tp3 = entry - 2.0 * R
        else:
            tp1 = entry + 0.5 * R
            tp2 = entry + 1.0 * R
            tp3 = entry + 2.0 * R

        position = self._compute_position(confidence, regime)

        return TradeSignal(
            symbol=snap.symbol,
            direction=bias,
            confidence=round(confidence, 2),
            reason=(
                f"{regime} 模式下触发确认价，按 R 结构下单，TP/SL 动态；"
                f"score long={long_score:.2f} / short={short_score:.2f}"
            ),
            entry=entry,
            entry_range=[trigger],
            tp1=tp1,
            tp2=tp2,
            tp3=tp3,
            sl=sl,
            core_position_pct=position["core"],
            add_position_pct=position["add"],
            snapshot=snap,
            debug_scores={**scores, "regime": regime, "trigger": trigger, "R": R},
        )
