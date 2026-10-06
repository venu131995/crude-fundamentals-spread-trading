"""Inventory surprises and the weekly fundamentals panel.

Inventory "surprise": oil markets trade the *difference* between the reported weekly stock change and
what was expected. Consensus forecasts are not freely available, so the expected change is the
5-year average change for the same calendar week (the standard seasonal benchmark desks quote).
Stocks are also compared with their 5-year seasonal range ("days of cover"-style tightness).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def seasonal_baseline(s: pd.Series, years: int = 5) -> pd.DataFrame:
    """For each week: mean/min/max of the same ISO week over the previous `years` years (no look-ahead)."""
    iso = s.index.isocalendar()
    df = pd.DataFrame({"v": s.to_numpy(), "wk": iso.week.clip(upper=52).astype(int).to_numpy(),
                       "yr": iso.year.astype(int).to_numpy()}, index=s.index)
    # one value per (ISO year, week): 53-week years would otherwise count a year twice
    per_year = df.groupby(["wk", "yr"])["v"].mean()
    rows = []
    for wk, yr in zip(df["wk"], df["yr"]):
        past = per_year.loc[wk]
        past = past[(past.index < yr) & (past.index >= yr - years)]
        rows.append((past.mean(), past.min(), past.max(), past.index.nunique()))
    out = pd.DataFrame(rows, index=df.index, columns=["avg5", "min5", "max5", "n"])
    out.loc[out["n"] < years, ["avg5", "min5", "max5"]] = np.nan
    return out.drop(columns="n")


def build_panel(eia: pd.DataFrame, px: pd.DataFrame) -> pd.DataFrame:
    """Weekly panel aligned to the price available after each EIA release.

    EIA data for the week ending Friday t is published on Wednesday t+5. Prices are sampled at the
    Wednesday close after release (`px_release`) and at the following Wednesday close (`px_next`), so
    any signal built from the report is traded strictly after it was public.
    """
    p = eia.dropna(subset=["us_stocks_kb", "cushing_kb"]).copy()
    for col in ["us_stocks_kb", "cushing_kb"]:
        chg = p[col].diff()
        p[f"{col}_chg"] = chg
        p[f"{col}_surprise"] = chg - seasonal_baseline(chg).avg5
        base = seasonal_baseline(p[col])
        p[f"{col}_vs5y"] = p[col] - base.avg5
        p[f"{col}_vs5y_pct"] = p[col] / base.avg5 - 1
    spread = (px["brent"] - px["wti"]).rename("spread")
    daily = pd.concat([px, spread], axis=1)
    release = p.index + pd.Timedelta(days=5)
    nxt = release + pd.Timedelta(days=7)
    p["px_release"] = daily["spread"].reindex(release, method="ffill").to_numpy()
    p["px_next"] = daily["spread"].reindex(nxt, method="ffill").to_numpy()
    p["wti_release"] = daily["wti"].reindex(release, method="ffill").to_numpy()
    p["wti_next"] = daily["wti"].reindex(nxt, method="ffill").to_numpy()
    p["spread_chg_next"] = p["px_next"] - p["px_release"]
    p["wti_ret_next"] = np.log(p["wti_next"] / p["wti_release"])
    return p
