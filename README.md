# Hyperliquid Basic Signal Bot

一个最基础的 **“是否开单 + 给点位”** 信号脚本：

- 数据源：Hyperliquid（通过 `ccxt.hyperliquid`）
- 逻辑：4H / 1H / 15m 三周期 EMA21 + 15m RSI + 1H ATR
- 输出：Long / Short / DoNothing + Entry 区间、TP1/TP2、SL 建议
- 运行方式：本地命令行 / GitHub Actions 定时执行

## 本地运行

```bash
git clone <your-repo>
cd hyperliquid-basic-bot

python -m venv venv
source venv/bin/activate   # Windows 用 venv\Scripts\activate

pip install -r requirements.txt

python -m bot.main

## 通知配置

如果希望在本地或 GitHub Actions 运行后推送到 Telegram、微信 Server酱或自定义 webhook，预先设置对应的环境变量即可：

- `TELEGRAM_BOT_TOKEN`、`TELEGRAM_CHAT_ID`
- `FTQQ_KEY`
- `WEBHOOK_URL`

当至少配置了一个渠道时，`bot.main` 会自动发送信号摘要；GitHub Actions 失败时也会尝试推送失败提醒。
