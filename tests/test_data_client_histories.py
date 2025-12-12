import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

from bot.config import Settings
from bot.data_client import HyperliquidDataClient


class DummyExchange:
    def market(self, symbol):
        return {"symbol": symbol, "id": symbol}


def make_sample_df(rows: int = 60) -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=rows, freq="1h", tz="UTC")
    base_price = 100
    data = {
        "timestamp": timestamps,
        "open": [base_price + i * 0.1 for i in range(rows)],
        "high": [base_price + i * 0.1 + 1 for i in range(rows)],
        "low": [base_price + i * 0.1 - 1 for i in range(rows)],
        "close": [base_price + i * 0.1 + 0.5 for i in range(rows)],
        "volume": [1000 + i for i in range(rows)],
    }
    return pd.DataFrame(data)


def test_build_tf_indicators_includes_histories():
    settings = Settings()
    client = HyperliquidDataClient(settings, exchange=DummyExchange(), funding_rates={})

    df = make_sample_df()
    tf = client._build_tf_indicators(df, "1h")

    assert len(tf.ma25_history) >= 2
    assert len(tf.rsi6_history) >= 3
