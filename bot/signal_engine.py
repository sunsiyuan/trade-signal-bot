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


class SignalEngine:
    """把指标 → 交易决策 的大脑。"""

    def __init__(self, settings: Settings):
        self.settings = settings

    def _score_direction(self, snap: MarketSnapshot) -> Dict[str, float]:
        """
        根据多周期趋势 + RSI + MACD + 衍生品情绪
        给 long / short 各打一套分数，0~1。
        """
        long_score = 0.0
        short_score = 0.0

        tf4 = snap.tf_4h
        tf1 = snap.tf_1h
        tf15 = snap.tf_15m
        d = snap.deriv
        s = self.settings

        # 1️⃣ 趋势维度（多周期统一方向）
        if tf4.trend_label == tf1.trend_label == tf15.trend_label == "down":
            short_score += 0.35
        elif tf4.trend_label == tf1.trend_label == tf15.trend_label == "up":
            long_score += 0.35

        # 2️⃣ RSI 6/12/24 共振（用配置里的阈值）
        short_trig = s.rsi_short_trigger      # 默认 60
        long_trig = s.rsi_long_trigger        # 默认 30

        # 空：高位一致偏强
        if (
            tf15.rsi6 >= short_trig
            and tf15.rsi12 >= short_trig - 5
            and tf15.rsi24 >= short_trig - 10
        ):
            short_score += 0.25

        # 多：低位一致偏弱
        if (
            tf15.rsi6 <= long_trig
            and tf15.rsi12 <= long_trig + 5
            and tf15.rsi24 <= long_trig + 10
        ):
            long_score += 0.25

        # 3️⃣ MACD 动能（以 1H 为主）
        if tf1.macd < tf1.macd_signal and tf1.macd_hist < 0:
            short_score += 0.15
        if tf1.macd > tf1.macd_signal and tf1.macd_hist > 0:
            long_score += 0.15

        # 4️⃣ 衍生品情绪：Funding & OI
        if d.funding > 0.02 and d.oi_change_24h > 0:
            short_score += 0.15
        if d.funding < -0.02 and d.oi_change_24h > 0:
            long_score += 0.15

        # 5️⃣ 盘口流动性（简单版：上方卖墙大 → 偏空，下方买墙大 → 偏多）
        ask_size = sum(w["size"] for w in d.orderbook_asks)
        bid_size = sum(w["size"] for w in d.orderbook_bids)
        if ask_size > bid_size * 2:
            short_score += 0.10
        if bid_size > ask_size * 2:
            long_score += 0.10

        return {
            "long": min(long_score, 1.0),
            "short": min(short_score, 1.0),
        }

    def generate_signal(self, snap: MarketSnapshot) -> TradeSignal:
        scores = self._score_direction(snap)
        long_score = scores["long"]
        short_score = scores["short"]

        # 胜率<0.6 直接不开单
        best = max(long_score, short_score)
        if best < 0.6:
            return TradeSignal(
                symbol=snap.symbol,
                direction="none",
                confidence=best,
                reason=f"多空得分不足（long={long_score:.2f}, short={short_score:.2f}），继续观望",
                snapshot=snap,
            )

        if short_score > long_score:
            direction: Direction = "short"
            score = short_score
        else:
            direction = "long"
            score = long_score

        # 简单用 15m ATR 做点位
        tf15 = snap.tf_15m
        price = tf15.close
        atr = tf15.atr if tf15.atr > 0 else 0.5  # 防止除0

        atr_sl = self.settings.atr_sl_mult * atr
        atr_tp1 = self.settings.atr_tp1_mult * atr
        atr_tp2 = self.settings.atr_tp2_mult * atr

        if direction == "short":
            entry_low = price
            entry_high = price + 0.25 * atr
            sl = entry_high + atr_sl
            tp1 = price - atr_tp1
            tp2 = price - atr_tp2
        else:
            entry_low = price - 0.25 * atr
            entry_high = price
            sl = entry_low - atr_sl
            tp1 = price + atr_tp1
            tp2 = price + atr_tp2

        confidence = score  # 0~1

        return TradeSignal(
            symbol=snap.symbol,
            direction=direction,
            confidence=confidence,
            reason=f"direction={direction}, long={long_score:.2f}, short={short_score:.2f}",
            entry=round(price, 4),
            entry_range=[round(entry_low, 4), round(entry_high, 4)],
            tp1=round(tp1, 4),
            tp2=round(tp2, 4),
            sl=round(sl, 4),
            core_position_pct=self.settings.core_position_pct,
            add_position_pct=self.settings.add_position_pct,
            snapshot=snap,
        )
