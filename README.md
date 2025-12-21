# Hyperliquid Basic Signal Bot

一个最基础的 **“是否开单 + 给点位”** 信号脚本：

- 数据源：Hyperliquid（通过 `ccxt.hyperliquid`）
- 逻辑：4H / 1H / 15m 三周期 EMA21 + 15m RSI + 1H ATR
- 输出：Long / Short / DoNothing + Entry 区间、TP1/TP2、SL 建议
- 运行方式：本地命令行 / GitHub Actions 定时执行

> **Strategy decides “what & why”. ConditionalPlan decides “how & when”.**

## 本地运行

```bash
git clone <your-repo>
cd hyperliquid-basic-bot

python -m venv venv
source venv/bin/activate   # Windows 用 venv\Scripts\activate

pip install -r requirements.txt

python -m bot.main
```

> 提示：本地运行时会自动读取当前目录下的 `.env` 文件，因此可以将下面提到的环境变量写入 `.env` 以便独立配置 Telegram、Server酱或 webhook 渠道。

## 历史数据与回测

### 数据结构（JSONL，必需）

回测/历史数据存储位于 `data/market_data/`，每个 symbol 一个目录。

**OHLCV candles（必须）**

文件：`data/market_data/<symbol>/candles_<timeframe>.jsonl`

每行字段：

```json
{
  "ts_open_ms": 1720000000000,
  "open": 123.4,
  "high": 130.0,
  "low": 121.0,
  "close": 128.0,
  "volume": 10000.0
}
```

支持时间周期：`15m`（驱动回测）、`1h`、`4h`。如果 `1h/4h` 不存在，会从 `15m` 重采样。

**可选 OI（允许缺失）**

文件：`data/market_data/<symbol>/open_interest_1h.jsonl`

```json
{
  "ts_ms": 1720000000000,
  "open_interest": 123456.0,
  "oi_change_24h": -0.08
}
```

缺失时会自动降级运行（策略会降低信心/缩仓或跳过相关 gating）。

> 目前仓库自带 `data/oi_history/*.csv`（OI 历史），回测默认不会直接读取该目录。如需回测使用 OI，可考虑把 CSV 转换为上述 JSONL 格式（后续可扩展脚本）。

### 定时增量拉取（candles）

```bash
python scripts/fetch_market_data.py \
  --symbols BTC/USDC:USDC,ETH/USDC:USDC \
  --timeframes 15m,1h,4h \
  --lookback_hours 48 \
  --max_requests_per_minute 60
```

**行为说明**：
- 读取本地 `meta.json` 的 max ts，带 `lookback_hours` 补洞。
- 统一指数退避 + jitter。
- token bucket 软限流。

### 一次性回填历史（candles）

```bash
python scripts/backfill_market_data.py \
  --symbols BTC/USDC:USDC \
  --timeframes 15m,1h,4h \
  --start 2024-01-01T00:00:00+00:00 \
  --end 2024-02-01T00:00:00+00:00
```

### 回测执行

**执行模式 A：仅 EXECUTE_NOW**

```bash
python -m bot.backtest.cli \
  --symbol BTC/USDC:USDC \
  --mode execute_now_only \
  --data-dir data/market_data \
  --output-dir data/backtest_output
```

**执行模式 B：EXECUTE_NOW + PLACE_LIMIT_4H**

```bash
python -m bot.backtest.cli \
  --symbol BTC/USDC:USDC \
  --mode execute_now_and_limit4h \
  --data-dir data/market_data \
  --output-dir data/backtest_output
```

### 回测输出

输出目录：`data/backtest_output/`

- `trades.jsonl`：逐笔回测记录（包含 `duplicate_skipped`、`duplicate_reason`、`first_exec_ts`）。
- `summary.json`：汇总指标（交易数、胜率、总收益、回撤、fill_rate、平均成交时长、数据覆盖率）。

### py_compile 自检

```bash
python -m py_compile bot/backtest/*.py scripts/*.py
```

## 通知配置

如果希望在本地或 GitHub Actions 运行后推送到 Telegram、微信 Server酱或自定义 webhook，预先设置对应的环境变量即可：

- `TELEGRAM_ACTION_BOT_TOKEN`、`TELEGRAM_ACTION_CHAT_ID`
- `TELEGRAM_SUMMARY_BOT_TOKEN`、`TELEGRAM_SUMMARY_CHAT_ID`
- `TELEGRAM_BOT_TOKEN`、`TELEGRAM_CHAT_ID`
- `FTQQ_KEY`
- `WEBHOOK_URL`

当至少配置了一个渠道时，`bot.main` 会自动发送信号摘要；GitHub Actions 失败时也会尝试推送失败提醒。
