# data_client.py
# ============================================================
# 数据连接/特征构建层（Data Client / Feature Builder）
#
# 这个文件的职责可以分成 4 层：
#
# A) 交易所连接层（ccxt.hyperliquid）
#    - fetch_ohlcv 拉K线（多周期）
#    - fetch_ticker / fetch_funding_rates 拉衍生品信息（funding / OI / mark）
#    - fetch_order_book 拉盘口（用于 liquidity hunt 的 wall 判断）
#
# B) 数据清洗与对齐（核心是：只用“已收盘K线”）
#    - Hyperliquid 会返回“正在形成的最后一根K线”，如果拿来算 RSI/MA 会导致：
#      1) 指标抖动  2) 和 UI 显示不一致  3) 策略触发来回翻
#    - 因此这里会 drop 最后一根未收盘K线（_drop_incomplete_last_candle）
#
# C) 特征计算（Indicators + Microstructure）
#    - 多周期：MA(7/25/99), RSI(6/12/24), MACD, ATR
#    - 趋势标签：detect_trend（基于 MA 关系）
#    - 震荡/波动特征：atr_rel、return_last、tr_last、rsidev 等
#    - LH 特征：recent_high/low, post_spike_small_body_count, orderbook walls
#    - OI 特征：open_interest + 24h change（API优先，CSV兜底）
#
# D) 输出统一结构 MarketSnapshot（策略层只读 snapshot，不关心数据细节）
#    - tf_4h/tf_1h/tf_15m: TimeframeIndicators
#    - deriv: DerivativeIndicators
#    - 一些跨特征：atrrel / rsidev / asks / bids 等
# ============================================================

import ccxt
import pandas as pd
import math
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

from .config import Settings
from .oi_history import load_oi_history, _parse_timestamp
from .indicators import ema, compute_rsi, macd, atr, detect_trend
from .models import (
    TimeframeIndicators,
    DerivativeIndicators,
    MarketSnapshot,
)


class HyperliquidDataClient:
    """
    Hyperliquid 的数据客户端（ccxt 封装）。
    多币种模式下建议：
    - 在 main.py 里只初始化一次 exchange 和 funding_rates
    - 这里通过注入 exchange/funding_rates 来复用，避免重复 API 调用。
    """

    def __init__(
        self,
        settings: Settings,
        exchange: Optional[ccxt.hyperliquid] = None,
        funding_rates: Optional[Dict] = None,
    ):
        self.settings = settings

        # exchange 可注入（多币种复用），否则自己创建一个带 rate limit 的实例。
        self.exchange = exchange or ccxt.hyperliquid({"enableRateLimit": True})

        # 如果 exchange 是这里新建的，需要 load_markets：
        # 否则 symbol 的解析可能不一致（Hyperliquid 的合约/别名映射）
        if exchange is None:
            self.exchange.load_markets()

        # market 是 ccxt 对该 symbol 的市场结构（包含 id/symbol 等）
        # 用于后面 funding_rates 的 symbol 匹配。
        self.market = self.exchange.market(self.settings.symbol)

        # funding_rates 缓存（在 main.py 里 fetch 一次后注入进来）
        self.funding_rates = funding_rates

    # ------------------------------------------------------------
    # 小工具：安全转 float
    # ------------------------------------------------------------
    @staticmethod
    def _to_float(value, default: float = 0.0) -> float:
        """
        将 value 转 float；失败则返回 default。
        用途：处理 ccxt 返回的 str/None/异常类型，避免指标链路中断。
        """
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    # ------------------------------------------------------------
    # A) OHLCV 获取 + 清洗：只保留已收盘K线
    # ------------------------------------------------------------
    def _fetch_ohlcv(self, timeframe: str, limit: int) -> pd.DataFrame:
        """
        从交易所拉 OHLCV，并转成 DataFrame。
        关键点：
        - timestamp 转 UTC（pd.to_datetime(..., utc=True)）
        - sort + reset_index 保证时序正确
        - 最后调用 _drop_incomplete_last_candle：丢掉“未收盘最后一根”
        """
        raw = self.exchange.fetch_ohlcv(
            self.settings.symbol,
            timeframe=timeframe,
            limit=limit,
        )
        df = pd.DataFrame(
            raw,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return self._drop_incomplete_last_candle(df, timeframe)

    def _drop_incomplete_last_candle(
        self, df: pd.DataFrame, timeframe: str
    ) -> pd.DataFrame:
        """
        Hyperliquid 会返回最新一根“正在形成中的 candle”。
        如果不 drop，会带来两个常见问题：
        1) 指标不稳定：RSI/MA/MACD 会随着价格实时跳动（你会看到频繁的 no action / action 抖动）
        2) 与 UI 不一致：你肉眼看盘的K线收盘后才确认，但 bot 提前拿未收盘数据算，会偏差

        这里的判定方法：
        - 取最后一行 timestamp（视为该 candle 的 open 时间）
        - duration = pd.to_timedelta(timeframe) 例如 "15m"/"1h"/"4h"
        - 如果 now < last_start + duration：说明 candle 还没收盘 -> drop 最后一行
        """
        if df.empty:
            return df

        duration = pd.to_timedelta(timeframe)
        last_start = df["timestamp"].iloc[-1].to_pydatetime().replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)

        if now < last_start + duration:
            return df.iloc[:-1]

        return df

    # ------------------------------------------------------------
    # funding_rates 匹配：不同 payload 下的 symbol 兼容
    # ------------------------------------------------------------
    def _normalize_symbol(self, symbol: str) -> str:
        """
        用于把不同格式的 symbol（可能带 : 或大小写差异）统一到同一个可比较形式。
        - ":" 替换成 "/"
        - upper
        """
        if not isinstance(symbol, str):
            return ""
        return symbol.replace(":", "/").upper()

    def _symbol_matches(self, candidate_symbol: str) -> bool:
        """
        判断 candidate_symbol 是否“等价于”当前 settings.symbol：
        - settings.symbol（原始配置）
        - market['symbol']（ccxt 标准符号）
        - market['id']（交易所内部 id）
        三者都 normalize 后放进 target_symbols。
        """
        target_symbols = {
            self._normalize_symbol(self.settings.symbol),
            self._normalize_symbol(self.market.get("symbol")),
            self._normalize_symbol(self.market.get("id")),
        }
        return self._normalize_symbol(candidate_symbol) in target_symbols

    # ------------------------------------------------------------
    # B) Orderbook Walls：给 Liquidity Hunt 策略用的“墙体”特征
    # ------------------------------------------------------------
    def _compute_orderbook_walls(
        self, orderbook_asks: List[Dict], orderbook_bids: List[Dict]
    ):
        """
        目标：计算盘口在“离 mid 一定深度内”的累计卖墙/买墙 size，并判断是否存在“显著大墙”。

        参数来自 settings.liquidity_hunt：
        - wall_depth_bps：在 mid 附近多深算“墙体区域”（bps=万分比）
          例如 30 bps = 0.30% 深度
        - wall_size_threshold：大墙判定阈值（相对倍数）
          例如 ask_wall >= 3 * bid_wall -> has_large_ask_wall=True

        返回：
        - ask_wall_size / bid_wall_size：深度内累计 size
        - has_large_ask_wall / has_large_bid_wall：是否存在显著墙体
        """
        lh_cfg = getattr(self.settings, "liquidity_hunt", {})
        wall_depth_bps = float(lh_cfg.get("wall_depth_bps", 30))
        wall_size_threshold = float(lh_cfg.get("wall_size_threshold", 3.0))

        if not orderbook_asks or not orderbook_bids:
            return 0.0, 0.0, False, False

        try:
            best_ask = float(orderbook_asks[0]["price"])
            best_bid = float(orderbook_bids[0]["price"])
        except (KeyError, TypeError, ValueError):
            return 0.0, 0.0, False, False

        mid_price = (best_ask + best_bid) / 2 if best_ask and best_bid else None
        if not mid_price or mid_price <= 0:
            return 0.0, 0.0, False, False

        depth_ratio = wall_depth_bps / 10000.0

        def _in_depth(price: float, side: str) -> bool:
            """
            判断某个价位是否在“mid 附近指定深度”内：
            - ask：price >= mid 且 (price-mid)/mid <= depth_ratio
            - bid：price <= mid 且 (mid-price)/mid <= depth_ratio
            """
            if side == "ask":
                return price >= mid_price and (price - mid_price) / mid_price <= depth_ratio
            return price <= mid_price and (mid_price - price) / mid_price <= depth_ratio

        # 累加深度范围内的 size
        ask_wall_size = sum(
            float(ask.get("size", 0))
            for ask in orderbook_asks
            if _in_depth(float(ask.get("price", 0)), "ask")
        )
        bid_wall_size = sum(
            float(bid.get("size", 0))
            for bid in orderbook_bids
            if _in_depth(float(bid.get("price", 0)), "bid")
        )

        # 大墙判定（倍数对比）
        has_large_ask_wall = ask_wall_size >= wall_size_threshold * max(
            bid_wall_size, 1e-6
        )
        has_large_bid_wall = bid_wall_size >= wall_size_threshold * max(
            ask_wall_size, 1e-6
        )

        return ask_wall_size, bid_wall_size, has_large_ask_wall, has_large_bid_wall

    # ------------------------------------------------------------
    # C) Funding / OI 提取：兼容多种 ccxt payload
    # ------------------------------------------------------------
    def _extract_funding(self, entry: Dict) -> Optional[float]:
        """
        从 fetch_funding_rates 的 entry 里提取 funding。
        因为不同交易所/不同 ccxt 版本字段不一致，所以这里做“候选字段列表”。
        返回第一个非 None 的候选值。
        """
        if not isinstance(entry, dict):
            return None

        info = entry.get("info", {}) if isinstance(entry.get("info"), dict) else {}

        candidates = [
            entry.get("fundingRate"),
            entry.get("nextFundingRate"),
            entry.get("predictedFundingRate"),
            entry.get("lastFundingRate"),
            info.get("fundingRate"),
            info.get("funding"),
            info.get("nextFundingRate"),
            info.get("predictedFundingRate"),
        ]

        return next((c for c in candidates if c is not None), None)

    def _extract_open_interest(self, entry: Dict) -> Optional[float]:
        """
        同样做 open interest 的候选字段兼容。
        并且：
        - candidate -> float
        - 只接受 math.isfinite 的值（过滤 nan/inf）
        """
        if not isinstance(entry, dict):
            return None

        info = entry.get("info", {}) if isinstance(entry.get("info"), dict) else {}

        candidates = [
            entry.get("openInterest"),
            entry.get("openInterestAmount"),
            entry.get("openInterestValue"),
            info.get("openInterest"),
            info.get("openInterestAmount"),
            info.get("openInterestValue"),
        ]

        for candidate in candidates:
            if candidate is None:
                continue
            value = self._to_float(candidate, default=float("nan"))
            if math.isfinite(value):
                return value

        return None

    def _compute_oi_change_24h(self, current_oi: float) -> Optional[float]:
        """
        计算 24h OI 百分比变化（%）。

        主路径（API 优先）：
        - exchange.fetch_open_interest_history(symbol, timeframe="1h", since=now-24h)
        - 找最早一条作为 24h 前 OI
        - change = (current - previous) / previous * 100

        失败/缺失时的兜底路径（本地 CSV 累积数据）：
        - load_oi_history(symbol, hours=30)
        - 找到“<= 目标时间点（整点对齐的 now-24h）”的最新一行作为 previous
        - 同样算百分比变化

        设计权衡：
        - 不把 OI 变化当作“强依赖”：失败返回 None，不中断主流程
        - 让策略层自己决定 require_oi / fallback 的降级逻辑
        """
        if current_oi is None or not math.isfinite(current_oi):
            return None

        try:
            since_ms = int((datetime.now(timezone.utc) - timedelta(hours=24)).timestamp() * 1000)
            history = self.exchange.fetch_open_interest_history(
                self.settings.symbol,
                timeframe="1h",
                since=since_ms,
                limit=48,
            )
        except Exception:
            history = None

        def _compute_change(previous: Optional[float]) -> Optional[float]:
            if previous is None or previous <= 0:
                return None
            return (current_oi - previous) / previous * 100

        # --- API 路径 ---
        if history:
            earliest = history[0] if isinstance(history, list) else history
            if isinstance(history, list) and history:
                earliest = sorted(
                    history,
                    key=lambda x: x.get("timestamp", 0) if isinstance(x, dict) else 0,
                )[0]

            previous_oi = self._extract_open_interest(earliest)
            change = _compute_change(previous_oi)
            if change is not None:
                return change

        # --- CSV 兜底路径（按小时快照） ---
        history_rows = load_oi_history(self.settings.symbol, hours=30)
        if len(history_rows) >= 2:
            target_ts = (
                datetime.now(timezone.utc)
                .replace(minute=0, second=0, microsecond=0)
                - timedelta(hours=24)
            )

            fallback_row = None
            # rows 升序；从尾部开始找 <= target_ts 的最近行，保证“至少 24h”而不是“刚好差一点”
            for row in reversed(history_rows):
                ts = _parse_timestamp(str(row.get("timestamp_utc", "")))
                if ts is None:
                    continue
                if ts <= target_ts:
                    fallback_row = row
                    break

            if fallback_row is None:
                return None

            try:
                previous_oi = float(fallback_row.get("oi", "nan"))
                if math.isfinite(previous_oi):
                    return _compute_change(previous_oi)
            except (TypeError, ValueError):
                return None

        return None

    # ------------------------------------------------------------
    # A') 多周期 OHLCV 一次拉齐
    # ------------------------------------------------------------
    def fetch_all_ohlcv(self) -> Dict[str, pd.DataFrame]:
        """
        拉取 4h/1h/15m 三套 OHLCV（各自 limit 在 Settings 里配置）。
        输出 dict 用固定 key："4h","1h","15m"（和 Settings.tf_* 对齐）。
        """
        s = self.settings
        return {
            "4h": self._fetch_ohlcv(s.tf_4h, s.candles_4h),
            "1h": self._fetch_ohlcv(s.tf_1h, s.candles_1h),
            "15m": self._fetch_ohlcv(s.tf_15m, s.candles_15m),
        }

    # ------------------------------------------------------------
    # D) 衍生品指标：funding / OI / orderbook / mark
    # ------------------------------------------------------------
    def fetch_derivative_indicators(self) -> DerivativeIndicators:
        """
        构造 DerivativeIndicators（衍生品维度特征）。

        数据来源：
        - fetch_ticker：可能带 mark/last/openInterest/funding 等（不保证）
        - fetch_funding_rates：更像“perps context”，一般更可靠，但也可能失败/类型异常
        - fetch_order_book：盘口前 N 档，用于 LH wall 计算与 liquidity comment

        注意：
        - 这层做的是“尽力而为”的聚合；任何一步失败都尽量兜底，而不是让 bot crash。
        - 真正高精度/强一致的数据，可未来换成 Hyperliquid 官方 REST/WebSocket。
        """
        ticker_info: Dict = {}
        ticker_error = None
        try:
            ticker = self.exchange.fetch_ticker(self.settings.symbol)
            ticker_info = ticker.get("info", {}) or {}
        except Exception as exc:  # pragma: no cover
            ticker_error = str(exc)

        def _first_finite(candidates):
            """候选列表里找到第一个 finite 的 float。"""
            for candidate in candidates:
                if candidate is None:
                    continue
                value = self._to_float(candidate, default=float("nan"))
                if math.isfinite(value):
                    return value
            return None

        # mark_price：优先 ticker 里的 mark/last，再尝试 info 的 markPrice/markPx/last 等
        mark_price = None
        if isinstance(ticker, dict):
            mark_price = _first_finite(
                [
                    ticker.get("mark"),
                    ticker.get("last"),
                    ticker_info.get("markPrice"),
                    ticker_info.get("markPx"),
                    ticker_info.get("last"),
                    ticker_info.get("lastPrice"),
                ]
            )

        funding_entry = None
        funding_source = None
        raw_funding = None
        funding_rates_error = None

        # open_interest：先从 ticker_info 的 openInterest 读（可能是 None/0）
        open_interest = self._to_float(ticker_info.get("openInterest"))

        # --- funding_rates 路径：优先复用 self.funding_rates，否则现场 fetch ---
        try:
            rates = self.funding_rates
            if rates is None:
                rates = self.exchange.fetch_funding_rates()

            candidate = None
            # rates 可能是 list[dict] 或 dict
            if isinstance(rates, list):
                candidate = next(
                    (
                        r
                        for r in rates
                        if isinstance(r, dict)
                        and self._symbol_matches(r.get("symbol"))
                    ),
                    None,
                )
            elif isinstance(rates, dict) and rates.get("symbol"):
                candidate = rates if self._symbol_matches(rates.get("symbol")) else None

            if candidate:
                funding_entry = candidate
                raw_funding = self._extract_funding(candidate)
                funding_source = "fetch_funding_rates"

                # 同时尝试从 candidate.info 里补 open_interest（更准确）
                entry_info = candidate.get("info", {}) if isinstance(candidate, dict) else {}
                if isinstance(entry_info, dict):
                    open_interest = self._to_float(
                        entry_info.get("openInterest"), default=open_interest
                    )
            elif rates is not None:
                funding_rates_error = f"unexpected funding type: {type(rates).__name__}"
        except Exception as exc:  # pragma: no cover
            funding_rates_error = str(exc)

        # funding 兜底：如果 funding_rates 提取失败，看看 ticker_info 里有没有
        if raw_funding is None:
            raw_funding = ticker_info.get("funding") or ticker_info.get("fundingRate")
            if raw_funding is not None:
                funding_source = "ticker_info"

        # 最终 funding：缺失时落到 0.0（注意：0.0 并不等于“真的为0”，只是“缺数据兜底”）
        funding = self._to_float(raw_funding, default=0.0)

        # 计算 OI 24h change（可能返回 None）
        oi_change_24h = self._compute_oi_change_24h(open_interest)

        # --- 盘口 ---
        orderbook = self.exchange.fetch_order_book(self.settings.symbol, limit=20)
        asks_raw: List[List[float]] = orderbook.get("asks", [])
        bids_raw: List[List[float]] = orderbook.get("bids", [])

        def _format_price(price: float) -> float:
            """
            用 ccxt 的 price_to_precision 把价格格式化成交易所允许精度。
            失败时直接 float(price)。
            """
            try:
                return float(
                    self.exchange.price_to_precision(self.settings.symbol, price)
                )
            except Exception:
                return float(price)

        # 只取前10档（上层 LH 判断是“近端墙体”，不需要太深）
        orderbook_asks = [
            {"price": _format_price(p), "size": float(s)} for p, s in asks_raw[:10]
        ]
        orderbook_bids = [
            {"price": _format_price(p), "size": float(s)} for p, s in bids_raw[:10]
        ]

        # 如果 ticker 没拿到 mark，则用 best ask/bid 的 mid 做一个近似 mark
        if mark_price is None and orderbook_asks and orderbook_bids:
            mid = (orderbook_asks[0]["price"] + orderbook_bids[0]["price"]) / 2
            mark_price = mid

        # liquidity_comment：一个非常粗的“买卖盘大小对比”标签（只看前10档的 size 总和）
        liquidity_comment = "asks>bids" if sum(a["size"] for a in orderbook_asks) > sum(
            b["size"] for b in orderbook_bids
        ) else "bids>=asks"

        # 墙体计算（为 LH 策略服务）
        (
            ask_wall_size,
            bid_wall_size,
            has_large_ask_wall,
            has_large_bid_wall,
        ) = self._compute_orderbook_walls(orderbook_asks, orderbook_bids)

        # ask_to_bid_ratio：便于策略层做连续值判断（而不是 only bool）
        ask_to_bid_ratio = None
        if bid_wall_size > 0:
            ask_to_bid_ratio = ask_wall_size / bid_wall_size

        return DerivativeIndicators(
            funding=funding,
            open_interest=open_interest,
            oi_change_24h=oi_change_24h,
            orderbook_asks=orderbook_asks,
            orderbook_bids=orderbook_bids,
            liquidity_comment=liquidity_comment,
            ask_wall_size=ask_wall_size,
            bid_wall_size=bid_wall_size,
            ask_to_bid_ratio=ask_to_bid_ratio,
            has_large_ask_wall=has_large_ask_wall,
            has_large_bid_wall=has_large_bid_wall,
            mark_price=mark_price,
        )

    # ------------------------------------------------------------
    # C') 单个 timeframe 的指标构建：TimeframeIndicators
    # ------------------------------------------------------------
    def _build_tf_indicators(self, df: pd.DataFrame, timeframe: str) -> TimeframeIndicators:
        """
        给某个周期的 OHLCV df 计算指标，并输出 TimeframeIndicators。

        你可以把 TimeframeIndicators 看成策略层的“最小特征向量”：
        - 趋势：MA(7/25/99) + trend_label
        - 动量：RSI(6/12/24) + MACD(hist)
        - 波动：ATR + atr_rel + tr_last
        - 数据质量：missing_bars_count + gap_list + is_last_candle_closed
        - LH：recent_high/low、post_spike_small_body_count
        - 价格特征：price_last/mid/typical/return_last
        """
        s = self.settings

        close = df["close"]
        closes_list = close.tolist()

        # --- MA（用于趋势判断和 rsidev 等） ---
        ma7 = ema(close, 7)
        ma25 = ema(close, 25)
        ma99 = ema(close, 99)

        # --- RSI：使用 closes_list + compute_rsi（自实现/封装） ---
        rsi6_series = compute_rsi(closes_list, s.rsi_fast).set_axis(close.index)
        rsi12_series = compute_rsi(closes_list, s.rsi_mid).set_axis(close.index)
        rsi24_series = compute_rsi(closes_list, s.rsi_slow).set_axis(close.index)

        rsi6_value = float(rsi6_series.iloc[-1])
        rsi12_value = float(rsi12_series.iloc[-1])
        rsi24_value = float(rsi24_series.iloc[-1])

        # Debug：只在指定条件下打印，避免 Actions log 被刷爆
        if (
            self.settings.debug_rsi
            and self.settings.symbol == "HYPE/USDC:USDC"
            and timeframe == "15m"
        ):
            print(
                f"[RSI_DEBUG] tf=15m symbol={self.settings.symbol} "
                f"last_closes={closes_list[-10:]} "
                f"rsi6={rsi6_value:.2f} rsi12={rsi12_value:.2f} rsi24={rsi24_value:.2f}"
            )

        # --- MACD：返回 line / signal / hist ---
        macd_line, macd_signal, macd_hist = macd(
            close,
            fast=s.macd_fast,
            slow=s.macd_slow,
            signal=s.macd_signal,
        )

        # --- ATR：基于 OHLC，输出 series ---
        atr_series = atr(df, s.atr_period)

        # --- 趋势标签：一般基于 MA 排列关系/斜率 ---
        trend_label = detect_trend(close, ma7, ma25, ma99)

        # history 长度：用于后续 slope/角度等计算或 debug 展示
        slope_lookback = int(self.settings.regime.get("slope_lookback", 5))
        hist_len = max(20, slope_lookback * 4)

        # --- LH：swing high/low（用于判断价格是否接近关键位） ---
        lh_cfg = getattr(self.settings, "liquidity_hunt", {})
        lookback = int(lh_cfg.get("swing_lookback_bars", 50))
        exclude = int(lh_cfg.get("swing_exclude_last", 1))

        # window_df：用于计算 recent_high/low 的窗口（排除最后 exclude 根，避免用到未确认结构）
        window_df = None
        if len(df) > lookback + exclude:
            window_df = df.iloc[-(lookback + exclude) : -exclude]
        elif exclude > 0:
            window_df = df.iloc[:-exclude]
        else:
            window_df = df

        recent_high = None
        recent_low = None
        high_last_n = None
        low_last_n = None
        if window_df is not None and len(window_df) >= 5:
            recent_high = float(window_df["high"].max())
            recent_low = float(window_df["low"].min())
            high_last_n = recent_high
            low_last_n = recent_low

        # --- LH：post_spike_small_body_count（吸筹/停顿感） ---
        small_body_lookback = int(lh_cfg.get("small_body_lookback", 8))
        small_body_mult = float(lh_cfg.get("small_body_atr_mult", 0.35))

        tail = df.iloc[-small_body_lookback:] if len(df) >= small_body_lookback else df
        post_spike_small_body_count = None
        if len(tail) >= 3:
            body = (tail["close"] - tail["open"]).abs()
            atr_tail = atr_series.iloc[-len(tail) :].abs()
            if len(atr_tail) == len(tail):
                small_body = body <= (atr_tail * small_body_mult)
                post_spike_small_body_count = int(small_body.sum())

        # --- 数据质量：时间戳对齐 + gap 检测（缺K线统计） ---
        timeframe_delta = pd.to_timedelta(timeframe)
        timestamps = df["timestamp"].dt.tz_convert(timezone.utc)

        last_open = timestamps.iloc[-1].to_pydatetime()
        last_close = last_open + timeframe_delta
        now = datetime.now(timezone.utc)

        expected_delta = timeframe_delta
        missing_bars = 0
        gap_list = []
        if len(timestamps) > 1:
            for prev, curr in zip(timestamps.iloc[:-1], timestamps.iloc[1:]):
                gap = curr - prev
                gap_units = gap.total_seconds() / expected_delta.total_seconds()
                # 大于 1.01 倍才认为缺失（给数据源/时钟一点误差空间）
                if gap_units > 1.01:
                    missing = int(round(gap_units - 1))
                    missing_bars += missing
                    gap_list.append(
                        {
                            "start_utc": (prev + expected_delta).isoformat(),
                            "end_utc": (curr - expected_delta).isoformat(),
                            "missing": missing,
                        }
                    )

        # --- 价格衍生特征：mid/typical/return/TR ---
        high_last = float(df["high"].iloc[-1])
        low_last = float(df["low"].iloc[-1])
        price_last = float(close.iloc[-1])
        price_mid = (high_last + low_last) / 2
        typical_price = (high_last + low_last + price_last) / 3

        prev_close = float(close.iloc[-2]) if len(close) > 1 else None
        return_last = None
        if prev_close and prev_close != 0:
            return_last = (price_last - prev_close) / prev_close

        # TR（True Range）最后一根：给你 debug/策略增强用（可反映单根波动）
        tr_last = None
        if prev_close is not None:
            tr1 = high_last - low_last
            tr2 = abs(high_last - prev_close)
            tr3 = abs(low_last - prev_close)
            tr_last = max(tr1, tr2, tr3)

        # lookback_window：表示“这些指标至少需要这么多 bars 才稳定”
        # 用于日志/调试/防止过短数据导致异常。
        lookback_window = max(
            99,
            s.macd_slow + s.macd_signal,
            s.rsi_slow + 1,
            s.atr_period,
        )

        return TimeframeIndicators(
            timeframe=timeframe,
            close=price_last,

            # MA
            ma7=float(ma7.iloc[-1]),
            ma25=float(ma25.iloc[-1]),
            ma99=float(ma99.iloc[-1]),
            ma25_history=[float(x) for x in ma25.tail(hist_len).tolist()],

            # RSI
            rsi6=rsi6_value,
            rsi6_history=[float(x) for x in rsi6_series.tail(hist_len).tolist()],
            rsi12=rsi12_value,
            rsi24=rsi24_value,

            # MACD
            macd=float(macd_line.iloc[-1]),
            macd_signal=float(macd_signal.iloc[-1]),
            macd_hist=float(macd_hist.iloc[-1]),

            # 波动/量
            atr=float(atr_series.iloc[-1]),
            volume=float(df["volume"].iloc[-1]),

            # 趋势标签
            trend_label=trend_label,

            # 时间对齐信息（非常重要：让你能确认 bot 用的是收盘K）
            last_candle_open_utc=last_open,
            last_candle_close_utc=last_close,
            is_last_candle_closed=now >= last_close,

            # 数据质量
            bars_used=len(df),
            lookback_window=lookback_window,
            missing_bars_count=missing_bars,
            gap_list=gap_list,

            # 价格统计
            price_last=price_last,
            price_mid=price_mid,
            typical_price=typical_price,
            return_last=return_last,

            # 相对波动：ATR / price
            atr_rel=float(atr_series.iloc[-1]) / max(price_last, 1e-9),
            tr_last=tr_last,

            # LH swing 信息
            recent_high=recent_high,
            recent_low=recent_low,
            high_last_n=high_last_n,
            low_last_n=low_last_n,
            post_spike_small_body_count=post_spike_small_body_count,
        )

    # ------------------------------------------------------------
    # D') 统一输出 MarketSnapshot：给策略层的唯一入口
    # ------------------------------------------------------------
    def build_market_snapshot(self) -> MarketSnapshot:
        """
        将所有数据拼成一个 MarketSnapshot：
        - 三周期 K线指标：tf_4h / tf_1h / tf_15m
        - 衍生品指标：deriv（funding/OI/orderbook/mark）
        - 跨周期汇总：atrrel / rsidev
        - 盘口汇总：asks / bids（前10档 size 总和）

        这样策略层只需要读 snapshot，不需要碰 ccxt/pandas。
        """
        ohlcvs = self.fetch_all_ohlcv()
        tf4 = self._build_tf_indicators(ohlcvs["4h"], "4h")
        tf1 = self._build_tf_indicators(ohlcvs["1h"], "1h")
        tf15 = self._build_tf_indicators(ohlcvs["15m"], "15m")
        deriv = self.fetch_derivative_indicators()

        ts = datetime.now(timezone.utc)

        # asks/bids：这里用 deriv.orderbook_* 的 size 总和（只统计前10档）
        asks = sum(order.get("size", 0) for order in deriv.orderbook_asks)
        bids = sum(order.get("size", 0) for order in deriv.orderbook_bids)

        # atrrel：你在日志里常用的“市场波动相对强度”（这里用 1h ATR / 1h close）
        atrrel = tf1.atr / max(tf1.close, 1e-6)

        # rsidev：价格相对 MA25 的偏离度（(close - ma25)/ma25），可作为震荡/趋势强度辅助特征
        rsidev = (tf1.close - tf1.ma25) / max(tf1.ma25, 1e-6)

        return MarketSnapshot(
            symbol=self.settings.symbol,
            ts=ts,
            tf_4h=tf4,
            tf_1h=tf1,
            tf_15m=tf15,
            deriv=deriv,

            # cross features
            atrrel=atrrel,
            rsidev=rsidev,

            # convenience fields（策略层常用）
            rsi_15m=tf15.rsi6,
            rsi_1h=tf1.rsi6,

            # microstructure summary
            asks=asks,
            bids=bids,
        )
