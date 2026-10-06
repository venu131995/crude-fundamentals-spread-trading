"""US crude fundamentals, the Brent-WTI export arbitrage, and a walk-forward spread strategy."""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crude import (RuleParams, backtest, build_panel, grid_search, lagged_correlation, newey_west_ols,  # noqa: E402
                   performance, seasonal_baseline)
from crude.data import crude_prices, eia_weekly  # noqa: E402

FIG, RES = ROOT / "figures", ROOT / "results"
IS_END, OOS_START = "2017-12-31", "2018-01-01"
FREIGHT_USD_BBL = 3.0          # representative USGC -> NW Europe Aframax freight + losses
GRID = {"lookback": [13, 26, 52], "entry": [0.75, 1.0, 1.5], "exit": [0.0, 0.25, 0.5], "stop": [3.0, 99.0]}
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9})


def main() -> None:
    FIG.mkdir(exist_ok=True); RES.mkdir(exist_ok=True)
    eia, px = eia_weekly(ROOT / "data"), crude_prices(ROOT / "data")
    panel = build_panel(eia["2004":], px)
    spread_d = (px["brent"] - px["wti"]).rename("spread")
    spread_w = spread_d["2012":].resample("W-WED").last().dropna()

    # ------------------------------------------------------------------ 1. fundamentals snapshot
    last = panel.iloc[-1]
    cush_base = seasonal_baseline(panel["cushing_kb"]).iloc[-1]
    us_base = seasonal_baseline(panel["us_stocks_kb"]).iloc[-1]
    snapshot = {
        "week_ending": str(panel.index[-1].date()),
        "us_commercial_stocks_mb": round(last["us_stocks_kb"] / 1e3, 1),
        "us_vs_5y_avg_pct": round(100 * (last["us_stocks_kb"] / us_base.avg5 - 1), 1),
        "cushing_stocks_mb": round(last["cushing_kb"] / 1e3, 1),
        "cushing_vs_5y_avg_pct": round(100 * (last["cushing_kb"] / cush_base.avg5 - 1), 1),
        "cushing_5y_range_mb": [round(cush_base.min5 / 1e3, 1), round(cush_base.max5 / 1e3, 1)],
        "production_mbd": round(last["production_kbd"] / 1e3, 2),
        "exports_mbd": round(last["exports_kbd"] / 1e3, 2),
        "brent_wti_spread": round(float(spread_d.iloc[-1]), 2),
        "export_arb_after_freight": round(float(spread_d.iloc[-1] - FREIGHT_USD_BBL), 2),
    }

    # ------------------------------------------------------------------ 2. does the arb move barrels?
    p18 = panel["2018":].copy()                                       # post export-ban, mature export market
    exp_dev = p18["exports_kbd"] - p18["exports_kbd"].rolling(26).mean()
    lagcorr = lagged_correlation(p18["px_release"], exp_dev, range(0, 13))
    k_star = int(lagcorr.idxmax())
    reg = pd.concat([exp_dev.shift(-k_star).rename("y"), p18["px_release"].rename("x")], axis=1).dropna()
    beta, se = newey_west_ols(reg["y"].to_numpy(), np.column_stack([np.ones(len(reg)), reg["x"].to_numpy()]), lags=8)
    arb_open = (spread_d["2018":] > FREIGHT_USD_BBL).mean()

    # ------------------------------------------------------------------ 3. do inventory surprises predict prices?
    s = panel["2010":].dropna(subset=["us_stocks_kb_surprise", "cushing_kb_surprise", "wti_ret_next", "spread_chg_next"])
    surprise_tests = {}
    for x, y in [("us_stocks_kb_surprise", "wti_ret_next"), ("cushing_kb_surprise", "spread_chg_next")]:
        b, e = newey_west_ols(s[y].to_numpy(), np.column_stack([np.ones(len(s)), s[x].to_numpy() / 1e3]), lags=4)
        surprise_tests[f"{x} -> {y}"] = {"slope_per_mb": round(float(b[1]), 5), "t_stat": round(float(b[1] / e[1]), 2),
                                        "corr": round(float(s[x].corr(s[y])), 3), "weeks": int(len(s))}
    regimes = {}
    for a, b_ in [("2010", "2014"), ("2015", "2019"), ("2020", "2026")]:
        q = panel.loc[a:b_].dropna(subset=["cushing_kb_vs5y", "px_release"])
        regimes[f"{a}-{b_}"] = {"corr_spread_vs_cushing_tightness": round(float(q["px_release"].corr(q["cushing_kb_vs5y"])), 2),
                                "mean_spread": round(float(q["px_release"].mean()), 2)}

    # ------------------------------------------------------------------ 4. strategy: calibrate IS, test OOS
    is_spread, base = spread_w[:IS_END], RuleParams()
    grid = grid_search(is_spread, base, GRID)
    grid.to_csv(RES / "grid_in_sample.csv", index=False)
    best = replace(base, **{k: grid.iloc[0][k] for k in GRID})
    best = replace(best, lookback=int(best.lookback))
    full = backtest(spread_w, best)
    oos = full[OOS_START:]
    default_full = backtest(spread_w, base)
    perf = {"in_sample_2012_2017": performance(full[:IS_END]["pnl"]), "out_of_sample_2018_2026": performance(oos["pnl"]),
            "oos_default_params": performance(default_full[OOS_START:]["pnl"]),
            "oos_gross_no_costs": performance(oos["gross_pnl"])}
    by_year = oos["pnl"].groupby(oos.index.year).sum().div(1e6).round(2).to_dict()
    cost_sens = {c: performance(backtest(spread_w, replace(best, cost_per_bbl=c))[OOS_START:]["pnl"])["sharpe"]
                 for c in [0.0, 0.02, 0.05, 0.10]}
    robust = grid.copy()
    robust_share = float((robust["sharpe"] > 0).mean())
    oos_grid = []
    for _, r in grid.iterrows():
        pr = replace(base, lookback=int(r.lookback), entry=r.entry, exit=r.exit, stop=r.stop)
        oos_grid.append(performance(backtest(spread_w, pr)[OOS_START:]["pnl"])["sharpe"])
    full.to_csv(RES / "strategy_weekly.csv", float_format="%.4f")

    out = {
        "data": {"eia": "WCESTUS1, W_EPC0_SAX_YCUOK_MBBL, WCRFPUS2, WCREXUS2 (weekly)", "prices": "FRED DCOILBRENTEU, DCOILWTICO",
                 "eia_last": str(eia.index[-1].date()), "price_last": str(px.index[-1].date())},
        "snapshot": snapshot,
        "export_flow_response_2018_2026": {"lag_weeks_peak": k_star, "corr_at_peak": round(float(lagcorr.max()), 3),
                                           "kbd_per_usd_spread": round(float(beta[1]), 1), "newey_west_t": round(float(beta[1] / se[1]), 2),
                                           "share_days_arb_open": round(float(arb_open), 3),
                                           "lag_profile": {int(k): round(float(v), 3) for k, v in lagcorr.items()}},
        "inventory_surprise_tests_2010_2026": surprise_tests,
        "fundamental_regimes": regimes,
        "strategy": {"selected_params_in_sample": {k: (float(v) if k != "lookback" else int(v)) for k, v in asdict(best).items()},
                     "performance": perf, "oos_pnl_by_year_usd_m": {int(k): v for k, v in by_year.items()},
                     "oos_sharpe_vs_cost_per_bbl": cost_sens,
                     "grid_share_positive_sharpe_in_sample": round(robust_share, 3),
                     "grid_share_positive_sharpe_out_of_sample": round(float(np.mean(np.array(oos_grid) > 0)), 3),
                     "grid_median_oos_sharpe": round(float(np.median(oos_grid)), 2)},
    }
    (RES / "summary.json").write_text(json.dumps(out, indent=2, default=str))

    # ------------------------------------------------------------------ figures
    fig, axs = plt.subplots(2, 2, figsize=(9, 5.6))
    for ax, col, lab in [(axs[0, 0], "us_stocks_kb", "US commercial crude stocks"), (axs[0, 1], "cushing_kb", "Cushing stocks")]:
        q = panel["2019":]
        base = seasonal_baseline(panel[col]).loc["2019":]
        ax.fill_between(q.index, base.min5 / 1e3, base.max5 / 1e3, color="0.85", label="Prior 5-yr range")
        ax.plot(q.index, base.avg5 / 1e3, color="0.5", lw=0.8, ls="--", label="5-yr average")
        ax.plot(q.index, q[col] / 1e3, color="tab:blue", lw=1.1, label="Actual")
        ax.set_title(lab); ax.set_ylabel("million bbl"); ax.legend(fontsize=6.5, loc="upper left")
    q = panel["2016":]
    axs[1, 0].plot(q.index, q["production_kbd"] / 1e3, lw=1); axs[1, 0].set_title("US crude production"); axs[1, 0].set_ylabel("mb/d")
    axs[1, 1].plot(q.index, q["exports_kbd"].rolling(4).mean() / 1e3, lw=1, color="tab:green")
    axs[1, 1].set_title("US crude exports (4-week avg)"); axs[1, 1].set_ylabel("mb/d")
    fig.suptitle(f"US crude fundamentals (EIA weekly, to {panel.index[-1].date()})", y=1.0)
    fig.tight_layout(); fig.savefig(FIG / "fig1_fundamentals_dashboard.png", bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.3), gridspec_kw={"width_ratios": [1.6, 1]})
    sd = spread_d["2016":]
    ax[0].plot(sd.index, sd, lw=0.7, color="0.3", label="Brent - WTI")
    ax[0].axhline(FREIGHT_USD_BBL, color="tab:red", ls="--", lw=1, label=f"Freight USGC-Europe ~${FREIGHT_USD_BBL:.0f}/bbl")
    ax[0].fill_between(sd.index, FREIGHT_USD_BBL, sd, where=sd > FREIGHT_USD_BBL, color="tab:green", alpha=0.3, label="Export arb open")
    ax[0].set_ylabel("USD/bbl"); ax[0].legend(fontsize=7, loc="upper left"); ax[0].set_title("Brent-WTI spread vs freight")
    ax[1].bar(lagcorr.index, lagcorr.values, color=["tab:green" if k == k_star else "tab:gray" for k in lagcorr.index])
    ax[1].set_xlabel("Weeks after the spread is observed"); ax[1].set_ylabel("Correlation with export deviation")
    ax[1].set_title(f"Exports respond after ~{k_star} weeks")
    fig.tight_layout(); fig.savefig(FIG / "fig2_export_arbitrage.png"); plt.close(fig)

    fig, ax = plt.subplots(2, 1, figsize=(8, 5.4), sharex=True, gridspec_kw={"height_ratios": [1.2, 1]})
    ax[0].plot(full.index, full["spread"], color="0.3", lw=0.8, label="Brent - WTI (weekly)")
    ax[0].plot(full.index, full["spread"].rolling(best.lookback).mean(), color="tab:blue", lw=1, label=f"{best.lookback}-week mean")
    for sgn, col, lab in [(1, "tab:green", "Long spread"), (-1, "tab:red", "Short spread")]:
        m = full["position"] == sgn
        ax[0].scatter(full.index[m], full["spread"][m], s=5, color=col, label=lab, zorder=3)
    ax[0].axvline(pd.Timestamp(OOS_START), color="k", ls=":", lw=1); ax[0].text(pd.Timestamp(OOS_START), ax[0].get_ylim()[1] * 0.92, "  out-of-sample →", fontsize=8)
    ax[0].set_ylabel("USD/bbl"); ax[0].legend(fontsize=7, ncol=2, loc="upper right"); ax[0].set_title("Mean-reversion strategy on the Brent-WTI spread")
    ax[1].plot(full.index, full["pnl"].cumsum() / 1e6, color="tab:blue", lw=1.3, label="Selected parameters (net of costs)")
    ax[1].plot(default_full.index, default_full["pnl"].cumsum() / 1e6, color="tab:orange", lw=1, ls="--", label="Default parameters")
    ax[1].axvline(pd.Timestamp(OOS_START), color="k", ls=":", lw=1)
    ax[1].axhline(0, color="k", lw=0.5); ax[1].set_ylabel("Cumulative P&L (USD m)\n100k bbl per leg"); ax[1].legend(fontsize=7, loc="upper left")
    fig.tight_layout(); fig.savefig(FIG / "fig3_strategy_backtest.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.scatter(grid["sharpe"], oos_grid, s=14, alpha=0.7)
    lim = [min(grid["sharpe"].min(), min(oos_grid)) - 0.1, max(grid["sharpe"].max(), max(oos_grid)) + 0.1]
    ax.plot(lim, lim, "k--", lw=0.7); ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
    ax.scatter([grid.iloc[0]["sharpe"]], [perf["out_of_sample_2018_2026"]["sharpe"]], s=60, color="tab:red", label="Selected")
    ax.set_xlabel("In-sample Sharpe (2012-2017)"); ax.set_ylabel("Out-of-sample Sharpe (2018-2026)"); ax.legend()
    ax.set_title(f"Parameter robustness: {len(grid)} rule variants")
    fig.tight_layout(); fig.savefig(FIG / "fig4_parameter_robustness.png"); plt.close(fig)

    print(json.dumps({k: out[k] for k in ["snapshot", "export_flow_response_2018_2026", "inventory_surprise_tests_2010_2026",
                                          "fundamental_regimes", "strategy"]}, indent=1, default=str))


if __name__ == "__main__":
    main()
