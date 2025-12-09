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
