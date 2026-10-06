import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from crude import (RuleParams, backtest, lagged_correlation, newey_west_ols, performance,  # noqa: E402
                   positions_from_z, seasonal_baseline)


def weekly_index(n):
    return pd.date_range("2010-01-06", periods=n, freq="W-WED")


def test_positions_hysteresis():
    z = pd.Series([0, 1.2, 0.8, 0.3, 0.1, -1.5, -0.5, -0.2, 3.0], index=weekly_index(9))
    pos = positions_from_z(z, entry=1.0, exit_=0.25, stop=2.5).tolist()
    assert pos == [0, -1, -1, -1, 0, 1, 1, 0, 0]


def test_backtest_pnl_and_costs_by_hand():
    s = pd.Series([5.0, 5.0, 5.0, 8.0, 6.0, 5.0], index=weekly_index(6))
    p = RuleParams(lookback=3, entry=1.0, exit=0.25, lots=10, cost_per_bbl=0.05)
    bt = backtest(s, p)
    z3 = (8 - np.mean([5, 5, 8])) / np.std([5, 5, 8], ddof=1)
    assert bt.loc[s.index[3], "z"] == pytest.approx(z3)
    assert bt.loc[s.index[3], "position"] == -1                              # short after the spike
    assert bt.loc[s.index[3], "gross_pnl"] == pytest.approx(-1 * (6 - 8) * 10_000)
    assert bt.loc[s.index[3], "cost"] == pytest.approx(1 * 10_000 * 0.05 * 2)


def test_no_lookahead_future_values_do_not_change_past_signals():
    rng = np.random.default_rng(0)
    s = pd.Series(np.cumsum(rng.normal(size=300)), index=weekly_index(300))
    a = backtest(s, RuleParams())
    s2 = s.copy()
    s2.iloc[200:] += 50
    b = backtest(s2, RuleParams())
    pd.testing.assert_series_equal(a["position"].iloc[:200], b["position"].iloc[:200])


def test_mean_reversion_profits_on_ou_and_not_on_random_walk():
    rng = np.random.default_rng(1)
    n = 3000
    ou = np.zeros(n)
    for t in range(1, n):
        ou[t] = 0.80 * ou[t - 1] + rng.normal()
    rw = np.cumsum(rng.normal(size=n))
    p = RuleParams(cost_per_bbl=0.0)
    assert performance(backtest(pd.Series(ou, index=weekly_index(n)), p)["pnl"])["sharpe"] > 0.75
    assert abs(performance(backtest(pd.Series(rw, index=weekly_index(n)), p)["pnl"])["sharpe"]) < 0.6


def test_seasonal_baseline_uses_only_prior_years():
    idx = pd.date_range("2010-01-01", "2017-12-31", freq="W-FRI")
    s = pd.Series(idx.year.astype(float), index=idx)            # value = year
    b = seasonal_baseline(s, years=5)
    assert np.isnan(b.loc["2014-06"].avg5).all()                # only 2010-2013 before mid-2014
    assert b.loc["2016-06"].avg5.iloc[0] == pytest.approx(np.mean([2011, 2012, 2013, 2014, 2015]))


def test_lagged_correlation_finds_true_lag_and_newey_west():
    rng = np.random.default_rng(2)
    idx = weekly_index(2000)
    x = pd.Series(rng.normal(size=2000), index=idx)
    y = 0.8 * x.shift(4) + rng.normal(scale=0.5, size=2000)
    lc = lagged_correlation(x, y, range(0, 9))
    assert lc.idxmax() == 4
    d = pd.concat([y, x.shift(4)], axis=1).dropna().to_numpy()
    beta, se = newey_west_ols(d[:, 0], np.column_stack([np.ones(len(d)), d[:, 1]]), lags=4)
    assert beta[1] == pytest.approx(0.8, abs=0.05) and se[1] < 0.05
