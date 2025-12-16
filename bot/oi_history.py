from __future__ import annotations

import csv
import math
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import ccxt

from .config import Settings


# ============================================================
# 这个模块在系统中的角色（非常关键）
#
# 你在 data_client.py 里想算 oi_change_24h，但 API 不稳定 / 有时缺字段。
# 所以这里提供一条“稳态兜底数据源”：
# - 每小时抓一次 OI / funding / mark price
# - 以 CSV 的形式落盘（并可选择 commit 到 repo）
#
# 这样做的工程收益：
# - 只要 GitHub Actions 稳定跑，你就能长期累积 OI 历史
# - data_client 在算 oi_change_24h 时，API 失败就回退用 CSV
# - 策略层可以继续坚持 require_oi，但缺数据时也有 fallback 机会
# ============================================================

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "oi_history"
# 说明：默认写到 repo 内的 data/oi_history/
# - 好处：Actions 环境易读写、且可 git commit
# - 代价：repo 会增长（不过是 CSV，体量还算可控）


def _normalize_symbol(symbol: str) -> str:
    """把 symbol 统一格式：把 ":" 替换为 "/" 并 upper，方便做等价比较。"""
    return symbol.replace(":", "/").upper()


def _parse_timestamp(value: Optional[str]) -> Optional[datetime]:
    """
    解析 CSV 中存的 timestamp_utc（ISO 字符串）为 datetime。

    兼容点：
    - datetime.fromisoformat 对以 'Z' 结尾的字符串不总是友好
      所以这里做一个小兜底：如果 endswith('Z')，去掉 Z 并补 tzinfo=UTC。

    注意：
    - 若解析失败返回 None（读文件时不会 crash）
    - 返回的 datetime 可能有 tzinfo，也可能没有；但你的写入是 isoformat(UTC) 形式，通常是带 tz 的。
    """
    if not value:
        return None

    try:
        return datetime.fromisoformat(value)
    except ValueError:
        # Some ISO strings may end with Z; handle gracefully.
        try:
            if value.endswith("Z"):
                return datetime.fromisoformat(value[:-1]).replace(tzinfo=timezone.utc)
        except Exception:
            return None
    return None


def _to_float(value: object) -> Optional[float]:
    """
    安全 float 转换：
    - 转不出来 -> None
    - nan/inf -> None
    目的：保证后续计算不会被“字符串/空值/非有限数”污染。
    """
    try:
        number = float(value)
        if math.isfinite(number):
            return number
    except (TypeError, ValueError):
        return None
    return None


@dataclass
class OIHistoryStore:
    """
    负责“每个 symbol 一张 CSV”的存取与 git commit。

    设计选择：
    - 文件按 symbol 拆分：方便 tail 读、也避免一个大文件越来越难管理
    - append_snapshot 会做“时间单调递增”校验，防止 Actions 重跑/乱序重复写
    """

    base_dir: Path = DEFAULT_DATA_DIR
    exchange: str = "hyperliquid"

    def _csv_path(self, symbol: str) -> Path:
        """
        symbol -> csv 文件名
        - "/" 和 ":" 替换为 "_"
        - 文件名格式：{exchange}__{sanitized}.csv
        """
        sanitized = symbol.replace("/", "_").replace(":", "_")
        return self.base_dir / f"{self.exchange}__{sanitized}.csv"

    def _read_last_timestamp(self, path: Path) -> Optional[datetime]:
        """
        读取 CSV 最后一行的 timestamp_utc，作为 append 的去重依据。

        注意：
        - 这里是“顺序扫到最后一行”，对非常大的 CSV 会 O(n)；
          但按小时写入、一币一年也就 ~8760 行，仍然可接受。
        - 如果未来要扩展到几十币 * 多年，可以考虑优化成 seek 尾部读取。
        """
        if not path.exists():
            return None

        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            last_row = None
            for row in reader:
                last_row = row

        if not last_row:
            return None

        return _parse_timestamp(last_row.get("timestamp_utc"))

    def append_snapshot(self, row: Dict[str, object]) -> bool:
        """
        追加一行快照（如果 timestamp 比文件尾部更新）。

        返回值语义：
        - True：写入了新行
        - False：跳过（通常因为：
            a) row.timestamp_utc 解析失败
            b) 文件已存在且 last_ts >= ts（重复/乱序）
          ）

        这里的“单调性”校验非常重要：
        - GitHub Actions 可能同一小时跑多次（或 retry）
        - 你希望每个小时只写一行，避免重复数据污染 oi_change_24h 的估计
        """
        self.base_dir.mkdir(parents=True, exist_ok=True)

        path = self._csv_path(str(row.get("symbol", "")))
        ts = _parse_timestamp(str(row.get("timestamp_utc", "")))
        if ts is None:
            return False

        last_ts = self._read_last_timestamp(path)
        if last_ts is not None and last_ts >= ts:
            return False

        file_exists = path.exists()
        with path.open("a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "timestamp_utc",
                    "exchange",
                    "symbol",
                    "oi",
                    "funding_rate",
                    "mark_price",
                ],
            )
            if not file_exists:
                writer.writeheader()

            writer.writerow(row)

        return True

    def load(self, symbol: str, hours: int = 24, now: Optional[datetime] = None) -> List[Dict]:
        """
        读取最近 N 小时的历史（按 timestamp 升序返回）。

        实现策略：
        - 全量读入 rows，然后按 cutoff 过滤（对当前规模足够）
        - 对 oi / funding_rate / mark_price 三列尝试转 float
          转不了就保留原值（通常是 ""）

        注意：
        - cutoff 采用 now_utc - hours
        - 如果行里 timestamp_utc 解析失败，会用 datetime.min 兜底参与比较（基本会被过滤掉）
        """
        now_utc = now or datetime.now(timezone.utc)
        cutoff = now_utc - timedelta(hours=hours)
        path = self._csv_path(symbol)

        if not path.exists():
            return []

        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        def _row_ts(r: Dict[str, str]) -> Optional[datetime]:
            return _parse_timestamp(r.get("timestamp_utc"))

        filtered = [r for r in rows if (_row_ts(r) or datetime.min.replace(tzinfo=timezone.utc)) >= cutoff]
        for row in filtered:
            for field in ("oi", "funding_rate", "mark_price"):
                value = row.get(field)
                parsed = _to_float(value)
                row[field] = parsed if parsed is not None else value

        filtered.sort(key=lambda r: _row_ts(r) or datetime.min.replace(tzinfo=timezone.utc))
        return filtered

    def has_pending_changes(self) -> bool:
        """
        判断 base_dir 目录下是否有 git 未提交的改动。

        用途：
        - CLI main() 里跑完 snapshot 后，若 changed 且有 pending changes，就 commit 一次。
        """
        result = subprocess.run(
            ["git", "status", "--porcelain", str(self.base_dir)],
            capture_output=True,
            text=True,
            check=False,
        )
        return bool(result.stdout.strip())

    def commit(self) -> None:
        """
        将 base_dir 下的 CSV add 并 commit。

        注意：
        - 这里不 push（Actions 是否 push 取决于 workflow 权限/配置）
        - commit message 固定：data: update hourly oi snapshot
        - 如果没有 csv_files，就直接 return（避免空提交）
        """
        csv_files = list(self.base_dir.glob("*.csv"))
        if not csv_files:
            return

        subprocess.run(["git", "add", *[str(f) for f in csv_files]], check=False)
        subprocess.run(
            ["git", "commit", "-m", "data: update hourly oi snapshot"],
            check=False,
        )


# ============================================================
# funding_rates / ticker 的提取工具：兼容 ccxt 返回结构差异
# ============================================================

def _select_funding_entry(funding_rates, symbol: str) -> Optional[Dict]:
    """
    从 fetch_funding_rates 的返回中找到某个 symbol 对应的 entry。
    - funding_rates 可能是 list[dict] 或 dict
    - 用 normalize_symbol 做等价比较
    """
    if isinstance(funding_rates, list):
        for entry in funding_rates:
            if isinstance(entry, dict) and _normalize_symbol(entry.get("symbol", "")) == _normalize_symbol(symbol):
                return entry
    elif isinstance(funding_rates, dict) and funding_rates.get("symbol"):
        if _normalize_symbol(funding_rates.get("symbol")) == _normalize_symbol(symbol):
            return funding_rates
    return None


def _extract_funding_rate(entry: Dict[str, object]) -> Optional[float]:
    """
    提取 funding rate（优先从 entry 顶层字段，其次从 entry['info'] 里找）。
    候选字段列表体现的是“ccxt/交易所字段不统一”的现实。
    """
    if not isinstance(entry, dict):
        return None

    candidates = [
        entry.get("fundingRate"),
        entry.get("nextFundingRate"),
        entry.get("predictedFundingRate"),
    ]

    info = entry.get("info", {}) if isinstance(entry.get("info"), dict) else {}
    candidates.extend([
        info.get("fundingRate"),
        info.get("nextFundingRate"),
        info.get("predictedFundingRate"),
        info.get("funding"),
    ])

    for candidate in candidates:
        value = _to_float(candidate)
        if value is not None:
            return value
    return None


def _extract_open_interest(entry: Dict[str, object]) -> Optional[float]:
    """
    提取 open interest（同样兼容不同字段名）。
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
        value = _to_float(candidate)
        if value is not None:
            return value
    return None


def _extract_mark_price(ticker: Dict[str, object]) -> Optional[float]:
    """
    提取 mark price（优先 ticker 顶层，再到 info 里找）。
    注意：如果 ticker 只给 last，不给 mark，这里会用 last 兜底。
    """
    info = ticker.get("info", {}) if isinstance(ticker.get("info"), dict) else {}
    candidates = [
        ticker.get("mark"),
        ticker.get("last"),
        info.get("markPrice"),
        info.get("markPx"),
        info.get("last"),
        info.get("lastPrice"),
    ]

    for candidate in candidates:
        value = _to_float(candidate)
        if value is not None:
            return value
    return None


class HourlyOISnapshot:
    """
    每小时采集 OI / funding / mark 并写入 CSV。

    这是一个“可被 GitHub Actions 定时触发”的小任务模块：
    - run() 会遍历 settings.tracked_symbols
    - 每个币构造一行 snapshot，写入对应 CSV
    - 是否 commit 由 main()（CLI）控制
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        store: Optional[OIHistoryStore] = None,
        exchange: Optional[ccxt.hyperliquid] = None,
    ):
        self.settings = settings or Settings()
        self.store = store or OIHistoryStore()

        # 可以注入 exchange（例如复用），否则自己建一个
        self.exchange = exchange or ccxt.hyperliquid({"enableRateLimit": True})
        if exchange is None:
            self.exchange.load_markets()

    def _current_hour(self) -> datetime:
        """
        返回“当前小时”的整点 UTC。
        这样每小时只会写一条，且 timestamp_utc 可用于对齐 24h 变化计算。
        """
        return datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

    def _fetch_ticker(self, symbol: str) -> Optional[Dict]:
        """
        拉 ticker（网络不稳定时返回 None，而不是抛异常中断）。
        这符合该任务的定位：采集任务尽量多拿到一些币的数据，个别失败不影响整体。
        """
        try:
            return self.exchange.fetch_ticker(symbol)
        except Exception as exc:  # pragma: no cover - network failure fallback
            print(f"[warn] fetch_ticker failed for {symbol}: {exc}")
            return None

    def _fetch_funding_rates(self):
        """
        funding_rates 通常一次 fetch 可以覆盖所有币，所以 run() 里会先 fetch 一次再复用。
        """
        try:
            return self.exchange.fetch_funding_rates()
        except Exception as exc:  # pragma: no cover - network failure fallback
            print(f"[warn] fetch_funding_rates failed: {exc}")
            return None

    def _build_snapshot(self, symbol: str, funding_rates) -> Optional[Dict[str, object]]:
        """
        构造单币的一行 snapshot。

        数据优先级（尽量用更“专门”的来源）：
        - funding_rate：优先 funding_rates 里的 entry，其次 ticker.info 兜底
        - open_interest：优先 funding_entry（若有），否则 ticker（某些交易所会在 ticker.info 带）
        - mark_price：从 ticker 提取（mark 不一定存在，会 fallback last）

        输出字段：
        - timestamp_utc：整点
        - exchange/symbol：维度标识
        - oi/funding_rate/mark_price：缺失则写空字符串（""）
          这样 CSV 永远有列，不会因为 None 导致写入异常
        """
        ticker = self._fetch_ticker(symbol)
        if not isinstance(ticker, dict):
            return None

        funding_entry = _select_funding_entry(funding_rates, symbol)
        funding_rate = _extract_funding_rate(funding_entry) if funding_entry else None
        open_interest = _extract_open_interest(funding_entry or ticker)
        mark_price = _extract_mark_price(ticker)

        # funding_rate 兜底：如果 funding_entry 没给，看看 ticker.info
        if funding_rate is None:
            info = ticker.get("info", {}) if isinstance(ticker.get("info"), dict) else {}
            funding_rate = _to_float(info.get("funding") or info.get("fundingRate"))

        ts = self._current_hour().isoformat()

        return {
            "timestamp_utc": ts,
            "exchange": self.store.exchange,
            "symbol": symbol,
            "oi": open_interest if open_interest is not None else "",
            "funding_rate": funding_rate if funding_rate is not None else "",
            "mark_price": mark_price if mark_price is not None else "",
        }

    def run(self) -> bool:
        """
        执行一次采集：
        - 先取 funding_rates（若失败则 funding_rates=None，后续 _build_snapshot 会退化）
        - 遍历 tracked_symbols
        - append_snapshot：只有当 ts 比文件尾部新时才写入
        - wrote：只要任意一个币写入成功就 True

        这个返回值用于 main() 决定是否 commit。
        """
        funding_rates = self._fetch_funding_rates()
        wrote = False
        for symbol in self.settings.tracked_symbols:
            snapshot = self._build_snapshot(symbol, funding_rates)
            if snapshot is None:
                continue
            wrote = self.store.append_snapshot(snapshot) or wrote
        return wrote


def load_oi_history(
    symbol: str,
    hours: int = 24,
    *,
    base_dir: Optional[Path] = None,
    exchange: str = "hyperliquid",
    now: Optional[datetime] = None,
) -> List[Dict]:
    """
    对外暴露的读取函数（data_client 会用它作为 oi_change_24h 的 CSV 兜底）。

    语义：
    - 从 CSV 读取最近 N 小时的历史，按 timestamp 升序返回
    - 若文件不存在/数据不足，则返回可用部分（可能是 []）

    参数：
    - base_dir 可覆盖（便于测试或本地换目录）
    - exchange 用于文件名前缀（如果未来接入别的交易所，也可以复用这套结构）
    - now 允许你在回测/单测里固定时间点
    """
    store = OIHistoryStore(base_dir=base_dir or DEFAULT_DATA_DIR, exchange=exchange)
    return store.load(symbol, hours=hours, now=now)


def main() -> None:  # pragma: no cover - CLI entry
    """
    CLI 入口：用于本地或 Actions 直接跑 `python -m ...oi_history` 之类。

    逻辑：
    - snapshotter.run() 采集
    - 若 changed 且 git status 显示 data_dir 有改动，则 commit
      * 这里还会设置 git identity（Actions 环境常需要）
    """
    snapshotter = HourlyOISnapshot()
    changed = snapshotter.run()

    if changed and snapshotter.store.has_pending_changes():
        print("[info] New hourly OI snapshots collected, committing changes...")
        # Configure git identity if not already set (useful in CI).
        subprocess.run(["git", "config", "user.name", "github-actions"], check=False)
        subprocess.run(["git", "config", "user.email", "github-actions@users.noreply.github.com"], check=False)
        snapshotter.store.commit()
    else:
        print("[info] No new hourly data; skipping commit.")


if __name__ == "__main__":  # pragma: no cover
    main()
