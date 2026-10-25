# Retail Demand Forecasting (PySpark) - leak-free features & rolling-origin backtesting

Forecasts daily unit sales **14 days ahead** for 400 store-product series (20 stores x 20 products,
2 years, 292k rows). Features are built with Spark window functions, models are evaluated with a
rolling-origin backtest against honest baselines, and forecasts come with empirically calibrated
80% prediction intervals.

## Quick start
```bash
pip install -r requirements.txt        # needs Java 17+
python scripts/generate_data.py        # simulate sales (promos, holidays, seasonality, trend, over-dispersed counts)
python scripts/run_backtest.py         # ~9 min locally (GBT x 3 folds); writes predictions + feature importance
python scripts/report.py               # intervals, bias correction, charts in reports/
pytest -q                              # 6 tests, incl. a no-future-leakage proof
```

## Results (3 folds x 14 days = 42 test days x 400 series = 16,800 forecasts)
| Method | WAPE | MAE | Bias |
|---|---|---|---|
| Naive (same weekday, 2 weeks ago) | 0.557 | 6.82 | +1.0% |
| Avg of last 3 same weekdays | 0.459 | 5.62 | +3.1% |
| 28-day moving average | 0.450 | 5.51 | +4.8% |
| Linear regression (log scale) | 0.389 | 4.77 | -12.1% |
| **Gradient-boosted trees (log scale)** | **0.376** | **4.61** | -11.4% |

GBT is **32% better than naive and 16% better than the moving average**, and wins in every fold
(0.377 / 0.369 / 0.382). It is also better on promo days (0.354 vs 0.381 on normal days): promo is a
known-in-advance feature and promo days have higher volume, so relative error is lower.

### Bias: found, diagnosed, and corrected
Both ML models under-forecast total demand by ~11-12%. Cause: training on `log1p(sales)` and
inverting with `expm1` gives something nearer the *median* than the *mean*, and over-dispersed
counts have mean > median. That is good for absolute error (WAPE) but bad for inventory planning,
where totals matter. Fix: scale each fold's forecasts by actual/forecast from the **previous fold**
(strictly past data). On folds 2-3: bias **-11.7% -> -1.1%**, WAPE 0.376 -> 0.378, and fold totals
land within ~2% of actual (e.g. 67.3k vs 67.1k units).

### Prediction intervals
80% intervals from the 10th/90th percentile of the previous fold's log residuals: **79.8% empirical
coverage** on folds 2-3, data the calibration never saw.

## Why the features can be trusted
Every sales-derived feature for date `d` uses only sales on or before `d - 14` (lags 14/21/28;
rolling mean/std over windows ending at d-14). This enables a *direct* 14-day forecast with no
recursive prediction. Promo, price, holiday and calendar are planned in advance, so they are
legitimately known for the future. `tests/test_forecasting.py` proves it: rewriting **all** sales newer
than d-14 leaves every feature unchanged, while changing day d-14 does change them (so the test can fail).

## Other design decisions
- **WAPE, not MAPE**: ~2.6% of days have zero sales, which makes MAPE undefined/explosive.
- **Baselines first**: naive, seasonal averages and a moving average. The ML model must beat them.
- **Rolling origin, not a random split**: train on all data up to a cutoff, forecast the next 14 days,
  repeat three times. Random splits would leak the future into training.
- **Features** (25): log lags, rolling level/volatility, promo, price ratio, holiday/neighbour-day flags,
  day-of-week, seasonal sin/cos, category. Top GBT feature: recent 28-day level (63% of importance).

## Limitations (please read)
- **Synthetic data.** Real demand has stockouts (zeros that aren't demand), price elasticity quirks,
  new products with no history, and structural breaks. Expect lower absolute accuracy.
- **Test windows are only the last 6 weeks (Nov 19 - Dec 30)** - a holiday-heavy period. Performance in
  other seasons is untested; run longer backtests before trusting these numbers year-round.
- Flat accuracy across horizon days 1-14 is **by construction** (all features are >=14 days old), not
  evidence that accuracy doesn't decay with horizon.
- Single simulated dataset and seed; no confidence intervals on WAPE. GBT vs linear (0.376 vs 0.389) is a
  small gap, and GBT's advantage over the 28-day moving average is the more robust finding.
- GBT was trimmed to 40 iterations for runtime and not tuned; the bias correction and intervals use
  only one calibration fold each.
- Cold-start series (<42 days of history) are dropped by the warm-up filter, not forecast.

## Ideas to extend
Tune GBT / try quantile or Tweedie objectives (directly targets the mean, removing the bias at source);
hierarchical reconciliation (store/category totals); promo-lift features (days since promo start);
price elasticity; per-series error monitoring and drift alerts.

## Layout
```
scripts/generate_data.py  simulator      src/features.py  leak-free window-function features
scripts/run_backtest.py   run folds      src/backtest.py  folds, baselines, GBT + linear pipelines
scripts/report.py         charts         src/metrics.py   WAPE / bias / coverage
tests/test_forecasting.py                 reports/         accuracy.png, forecasts.png
```
