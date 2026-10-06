"""Brent-WTI spread: flow analysis and a walk-forward mean-reversion strategy.

Economic mechanism: when Brent trades far above WTI (net of freight), exporting US crude becomes
profitable, export cargoes are fixed, barrels leave the US and the spread narrows again. The
strategy trades that convergence; the flow regression tests that the mechanism is visible in
the physical data.

Trading rule (weekly, Wednesday close; information up to and including that close):
    z = (spread - rolling mean) / rolling std over `lookback` weeks
    z >  entry -> short the spread (sell Brent, buy WTI)      z < -entry -> long the spread
    |z| < exit -> flat;  optional stop: flat if |z| > stop
Position size: `lots` x 1,000 bbl per leg; costs per leg on every change in position.
P&L for week t = position_t x (spread_{t+1} - spread_t) x barrels.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

BBL_PER_LOT = 1_000


@dataclass(frozen=True)
class RuleParams:
    lookback: int = 26
    entry: float = 1.0
    exit: float = 0.25
    stop: float = 99.0
    lots: int = 100                  # 100 lots = 100,000 bbl per leg
    cost_per_bbl: float = 0.02       # USD/bbl per leg per trade (fees + half bid/ask)


def zscore(spread: pd.Series, lookback: int) -> pd.Series:
    m = spread.rolling(lookback).mean()
    s = spread.rolling(lookback).std()
    return (spread - m) / s


def positions_from_z(z: pd.Series, entry: float, exit_: float, stop: float = 99.0) -> pd.Series:
    """+1 long spread, -1 short spread, 0 flat, with hysteresis between entry and exit."""
    pos, out = 0, []
    for v in z.to_numpy():
        if np.isnan(v) or abs(v) > stop:
            pos = 0
        elif pos == 0 and v > entry:
            pos = -1
        elif pos == 0 and v < -entry:
            pos = 1
        elif pos != 0 and abs(v) < exit_:
            pos = 0
        out.append(pos)
    return pd.Series(out, index=z.index, name="position", dtype=float)


def backtest(spread: pd.Series, p: RuleParams) -> pd.DataFrame:
    bt = pd.DataFrame({"spread": spread})
    bt["z"] = zscore(spread, p.lookback)
    bt["position"] = positions_from_z(bt["z"], p.entry, p.exit, p.stop)
    size = p.lots * BBL_PER_LOT
    bt["next_chg"] = spread.shift(-1) - spread
    bt["gross_pnl"] = bt["position"] * bt["next_chg"] * size
    bt["trades"] = bt["position"].diff().abs().fillna(bt["position"].abs())
    bt["cost"] = bt["trades"] * size * p.cost_per_bbl * 2
    bt["pnl"] = bt["gross_pnl"] - bt["cost"]
    return bt.dropna(subset=["next_chg"])


def performance(pnl: pd.Series, periods_per_year: int = 52) -> dict:
    pnl = pnl.dropna()
    cum = pnl.cumsum()
    dd = cum - cum.cummax()
    active = pnl[pnl != 0]
    sd = pnl.std(ddof=1)
    return {"total_usd_m": round(float(pnl.sum()) / 1e6, 2),
            "annual_usd_m": round(float(pnl.mean() * periods_per_year) / 1e6, 2),
            "sharpe": round(float(pnl.mean() / sd * np.sqrt(periods_per_year)), 2) if sd > 0 else float("nan"),
            "max_drawdown_usd_m": round(float(dd.min()) / 1e6, 2),
            "hit_rate_active_weeks": round(float((active > 0).mean()), 3) if len(active) else float("nan"),
            "active_share": round(float(len(active) / len(pnl)), 3), "weeks": int(len(pnl))}


def grid_search(spread: pd.Series, base: RuleParams, grid: dict) -> pd.DataFrame:
    """In-sample Sharpe for every parameter combination."""
    rows = []
    for combo in itertools.product(*grid.values()):
        p = replace(base, **dict(zip(grid, combo)))
        if p.exit >= p.entry:
            continue
        perf = performance(backtest(spread, p)["pnl"])
        rows.append({**dict(zip(grid, combo)), **perf})
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False)


def lagged_correlation(x: pd.Series, y: pd.Series, lags: range) -> pd.Series:
    return pd.Series({k: x.corr(y.shift(-k)) for k in lags}, name="corr")


def newey_west_ols(y: np.ndarray, X: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray]:
    """OLS coefficients and Newey-West (Bartlett kernel) standard errors."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    u = y - X @ beta
    n = len(y)
    Xu = X * u[:, None]
    S = Xu.T @ Xu
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        G = Xu[L:].T @ Xu[:-L]
        S += w * (G + G.T)
    XtX_inv = np.linalg.inv(X.T @ X)
    cov = XtX_inv @ S @ XtX_inv
    return beta, np.sqrt(np.diag(cov)) * np.sqrt(n / (n - X.shape[1]))
