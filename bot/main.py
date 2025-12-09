# bot/main.py
from .config import Settings
from .data_client import HyperliquidDataClient
from .signal_engine import SignalEngine


def print_signal(signal):
    snap = signal.snapshot

    print("====== TRADE SIGNAL ======")
    print(f"Time:   {snap.ts.isoformat()}")
    print(f"Symbol: {signal.symbol}")
    print()

    # 1）多周期指标
    def tf_row(tf):
        return (
            f"{tf.timeframe:>4} | "
            f"close={tf.close:.4f} | "
            f"MA7={tf.ma7:.2f}, MA25={tf.ma25:.2f}, MA99={tf.ma99:.2f} | "
            f"RSI6={tf.rsi6:.2f}, RSI12={tf.rsi12:.2f}, RSI24={tf.rsi24:.2f} | "
            f"MACD={tf.macd:.4f}, SIG={tf.macd_signal:.4f}, HIST={tf.macd_hist:.4f} | "
            f"ATR={tf.atr:.4f} | Trend={tf.trend_label}"
        )

    print("=== Timeframe Indicators ===")
    print(tf_row(snap.tf_4h))
    print(tf_row(snap.tf_1h))
    print(tf_row(snap.tf_15m))
    print()

    # 2）衍生品指标
    d = snap.deriv
    print("=== Derivative Indicators ===")
    print(
        f"Funding={d.funding:.6f}, "
        f"OI={d.open_interest:.2f}, "
        f"OI_24h_change={d.oi_change_24h:.2f}, "
        f"Liquidity={d.liquidity_comment}"
    )
    print("Top Asks (price,size):", [(w["price"], w["size"]) for w in d.orderbook_asks])
    print("Top Bids  (price,size):", [(w["price"], w["size"]) for w in d.orderbook_bids])
    print()

    # 3）信号结果
    print("=== Trade Signal ===")
    print(f"Direction : {signal.direction}")
    print(f"Confidence: {signal.confidence:.2f}")
    print(f"Reason    : {signal.reason}")

    if signal.direction != "none":
        print(f"Entry range: {signal.entry_range}")
        print(f"TP1: {signal.tp1} | TP2: {signal.tp2} | SL: {signal.sl}")
        print(
            f"Position: core={signal.core_position_pct * 100:.0f}% "
            f"+ add={signal.add_position_pct * 100:.0f}%"
        )
    else:
        print("No trade. Waiting for next setup.")
    print("===============================")


def main():
    settings = Settings()
    client = HyperliquidDataClient(settings)
    snapshot = client.build_market_snapshot()

    engine = SignalEngine(settings)
    signal = engine.generate_signal(snapshot)

    print_signal(signal)


if __name__ == "__main__":
    main()
