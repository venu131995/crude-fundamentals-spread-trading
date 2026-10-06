# Crude Fundamentals & Brent–WTI Spread Trading Simulator

A fundamentals-to-trading workflow for the **transatlantic crude arbitrage**. It reads the EIA's weekly
US petroleum data, tests which fundamentals actually move prices and physical flows, and turns the one
robust relationship into a **rule-based Brent–WTI spread strategy**. The strategy is calibrated on
2012–2017 and tested out of sample on 2018–2026, with trading costs.

The core question a crude desk asks is: **when US crude is cheap relative to Brent, do barrels leave the
US, and does that close the gap?**

## Data

| Source | Series | Frequency |
|---|---|---|
| EIA Weekly Petroleum Status Report | US commercial crude stocks, Cushing (OK) stocks, US production, US exports | weekly, to 25 Sep 2026 |
| FRED | Dated Brent, WTI Cushing spot | daily, to 29 Sep 2026 |

EIA data for the week ending Friday is published the following Wednesday. Every signal is lined up with
the **Wednesday close after release**, so nothing uses data before it was public.

## Results

All numbers come from `results/summary.json`, produced by `scripts/run_crude.py`.

### 1. Fundamentals snapshot (week ending 25 Sep 2026)

| Indicator | Level | vs. prior 5-year average |
|---|---|---|
| US commercial crude stocks | 427.3 mb | +1.9% |
| Cushing stocks | 24.3 mb | **−7.0%** (5-yr range 22.1–35.5 mb) |
| US production | 13.96 mb/d | record high |
| US exports | 3.57 mb/d | |
| Brent − WTI | **USD 17.80/bbl** | arb after ~USD 3 freight: **USD 14.80/bbl** |

Monthly-average spreads in April and September 2026 (about USD 17–18) are the widest since 2013,
ignoring the one-day April 2020 negative-WTI print, while Cushing sits near the bottom of its 5-year range.
That is a strong export pull on US barrels (`fig1_fundamentals_dashboard.png`).

### 2. The physical arbitrage moves barrels

From 2018 to 2026, US exports (measured as the deviation from their 26-week trend) rise **after** the
Brent–WTI spread widens. The response peaks about **4 weeks later** (correlation 0.31), which matches the
time needed to fix and load an export cargo.

| Measure | Value |
|---|---|
| Peak response lag | 4 weeks |
| Export response | **+61 kb/d per USD 1/bbl of spread** (Newey–West t = 8.8) |
| Days the export arb was open (spread > USD 3 freight) | 72% |

This is the mechanism behind spread mean reversion: a wide spread pulls US crude into the export market
and narrows the gap (`fig2_export_arbitrage.png`).

### 3. What does *not* work, and why that matters

**Weekly inventory surprises do not predict next-week prices.** Here a "surprise" is the reported stock
change minus the 5-year average change for that week.

| Test, 2010–2026 (772 weeks) | Correlation | Newey–West t |
|---|---|---|
| US stock surprise → next-week WTI return | 0.02 | 0.56 |
| Cushing stock surprise → next-week Brent–WTI change | 0.03 | 1.06 |

The market prices the EIA report on release day, and no weekly edge is left after that.

**Stock levels relate to the spread differently in different periods:**

| Period | Correlation: spread vs. Cushing tightness | Mean spread |
|---|---|---|
| 2010–2014 (pipeline bottleneck era) | +0.61 | USD 10.95 |
| 2015–2019 (export ban lifted Dec 2015) | −0.56 | USD 4.11 |
| 2020–2026 | −0.06 | USD 4.45 |

A static "fair value from inventories" model would have been fitted to one regime and then failed in the
next. So the strategy trades the spread's tendency to revert, which the flow analysis explains, rather
than a fragile inventory-based fair value.

### 4. Spread trading strategy: calibrate 2012–17, test 2018–26

**Rule:** each Wednesday, measure how far the spread is from its rolling mean in standard deviations (z).
- Short the spread (sell Brent, buy WTI) when z > entry; go long when z < −entry.
- Close when |z| < exit. Step aside when |z| > 3, since moves that large tend to be dislocations rather
  than noise.

**Calibration:** a 54-variant grid search on 2012–2017 only selected a 13-week lookback, entry 1.5, exit
0.5 and stop 3.0. Size is 100,000 bbl per leg, with costs of USD 0.02/bbl per leg per trade.

| | In-sample 2012–17 | **Out-of-sample 2018–26** |
|---|---|---|
| Net P&L | USD 3.75m | **USD 5.79m** |
| Annual P&L | USD 0.62m | USD 0.66m |
| **Sharpe ratio** | 0.82 | **0.72** |
| Max drawdown | −USD 1.03m | −USD 1.21m |
| Time in market | | 43%, average hold 4.7 weeks |

**Out-of-sample P&L by year (USD m):** 2018 +0.26 · 2019 +0.47 · 2020 +1.29 · 2021 +0.59 · 2022 +0.94 ·
2023 +0.50 · 2024 +0.71 · 2025 +0.59 · 2026 +0.45. **Profitable in all 9 years**
(`fig3_strategy_backtest.png`).

**Robustness:**

| Check | Result |
|---|---|
| Out-of-sample Sharpe vs. cost per bbl (USD 0 / 0.02 / 0.05 / 0.10) | 0.76 / 0.72 / 0.66 / 0.55 |
| Grid variants with positive out-of-sample Sharpe | **87%** (median 0.34) |
| Untuned default rule (26 weeks, entry 1.0), out of sample | Sharpe 0.41, USD 4.46m |

The edge doesn't depend on the chosen parameters, since most rule variants make money out of sample
(`fig4_parameter_robustness.png`). The worst week out of sample (−USD 1.2m, Sep 2026) came when the
strategy shorted the spread at USD 12 and it kept widening to USD 24. That is the main risk: a real
structural dislocation can look like an overshoot.

## Figures

| File | Content |
|---|---|
| `fig1_fundamentals_dashboard.png` | US and Cushing stocks vs. 5-year range; production; exports |
| `fig2_export_arbitrage.png` | Brent–WTI vs. freight (arb open) and the export response by lag |
| `fig3_strategy_backtest.png` | Spread, positions, and cumulative P&L, in- and out-of-sample |
| `fig4_parameter_robustness.png` | In-sample vs. out-of-sample Sharpe for all 54 rule variants |

## Simplifications

- Spot benchmarks stand in for front-month futures; roll yield and the calendar mismatch between dated
  Brent and Cushing WTI are ignored.
- Freight is a flat USD 3/bbl, whereas real USGC–Europe Aframax rates move between about USD 1.5 and 6.
- The inventory "surprise" uses a seasonal benchmark, not analyst consensus (which is not freely
  available). A consensus-based surprise may well have a short-lived intraday effect that weekly data
  cannot see.
- Positions are a fixed size, with no volatility targeting or stop-loss in USD terms.

## Tests (`tests/`, 6 passing)

- Hysteresis in the position rule.
- Hand-calculated P&L and costs.
- **No look-ahead:** changing future prices leaves past positions unchanged.
- The strategy profits on a simulated mean-reverting series and does not on a random walk.
- The seasonal baseline uses only prior years.
- The lag-correlation tool finds a known lag, and Newey–West recovers a known slope.

## Run

```bash
pip install -r requirements.txt
python -m pytest -q tests      # 6 tests, < 1 s
python scripts/run_crude.py    # ~5 s: downloads EIA + FRED data, writes results/ and figures/
```
