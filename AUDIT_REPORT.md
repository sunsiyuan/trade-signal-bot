# Repository Audit (2025-12-16)

## Observations
- Logging schema previously emitted placeholder price fields (`mark`, `index`) while omitting real-time values already available from `MarketSnapshot` (mark price, snapshot timestamp, orderbook wall sums, RSI rollups). This caused downstream logs to diverge from production data and hid latency/quality context.
- Market-level block did not include the snapshot timestamp or aggregated orderbook sizes (`asks`/`bids`) computed in `HyperliquidDataClient`, making replay and latency analysis harder.
- Regime diagnostics logged only ATR/RSI deviation aggregates; per-timeframe RSI values stored on the snapshot were never serialized.
- Timeframe blocks skipped volume and trend labels while carrying unused placeholders (`mark`, `index`), resulting in partially empty indicator payloads.

## Dead code / unreachable paths
- No completely unreachable modules or functions were identified during this pass. Placeholder logging fields that were never populated (timeframe `mark`, `index`) have been removed from the schema and replaced with live metrics.

## Recommendations
- Add automated schema drift checks that compare `MarketSnapshot` / `TradeSignal` fields with serialized output to prevent future omissions.
- Consider extending alignment diagnostics to include per-timeframe latency and volume completeness thresholds for alerting.
- If new derivatives data (e.g., funding forecasts or index prices) become available, wire them through `DerivativeIndicators` so schema changes can remain additive.
