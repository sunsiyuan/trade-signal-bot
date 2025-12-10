import math
import sys
from pathlib import Path

# Ensure repo root is on path for local imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

from bot.indicators import compute_rsi


def wilder_reference(closes, period):
    if len(closes) < period + 1:
        return math.nan

    gains = []
    losses = []
    for i in range(1, period + 1):
        change = closes[i] - closes[i - 1]
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    rsi_value = 100.0 if avg_loss == 0 else 100.0 - (100.0 / (1.0 + (avg_gain / avg_loss)))

    for i in range(period + 1, len(closes)):
        change = closes[i] - closes[i - 1]
        gain = max(change, 0.0)
        loss = max(-change, 0.0)

        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

        if avg_loss == 0:
            rsi_value = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi_value = 100.0 - (100.0 / (1.0 + rs))

    return rsi_value


def test_compute_rsi_matches_reference():
    prices = [44, 44.15, 43.9, 44.35, 44.6, 45.1, 45.0, 45.2, 45.1]
    period = 6

    rsi_series = compute_rsi(prices, period)
    result = rsi_series.iloc[-1]

    reference_value = wilder_reference(prices, period)

    assert math.isclose(result, reference_value, rel_tol=0, abs_tol=1e-6)
