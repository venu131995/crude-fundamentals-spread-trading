"""EIA weekly petroleum fundamentals and FRED daily crude prices."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd

EIA_URL = "https://www.eia.gov/dnav/pet/hist_xls/{series}w.xls"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
EIA = {
    "us_stocks_kb": "WCESTUS1",                    # US commercial crude stocks ex-SPR, thousand bbl
    "cushing_kb": "W_EPC0_SAX_YCUOK_MBBL",         # Cushing, OK stocks (WTI delivery point), thousand bbl
    "production_kbd": "WCRFPUS2",                  # US field production, thousand bbl/day
    "exports_kbd": "WCREXUS2",                     # US crude exports, thousand bbl/day
}
FRED = {"brent": "DCOILBRENTEU", "wti": "DCOILWTICO"}


def _curl(url: str, path: Path) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["curl", "-sL", "--max-time", "90", "-o", str(path), url], check=True)


def eia_weekly(data_dir: Path) -> pd.DataFrame:
    cols = {}
    for name, code in EIA.items():
        f = data_dir / f"eia_{code}.xls"
        _curl(EIA_URL.format(series=code), f)
        x = pd.read_excel(f, sheet_name="Data 1", skiprows=2)
        cols[name] = x.set_index(x.columns[0])[x.columns[1]].astype(float)
    df = pd.DataFrame(cols)
    df.index = pd.to_datetime(df.index)
    df.index.name = "week"                       # EIA week-ending Friday; data published the following Wednesday
    return df.sort_index()


def crude_prices(data_dir: Path) -> pd.DataFrame:
    cols = {}
    for name, code in FRED.items():
        f = data_dir / f"{name}_daily.csv"
        _curl(FRED_URL.format(series=code), f)
        d = pd.read_csv(f, parse_dates=["observation_date"], na_values=["."])
        cols[name] = d.set_index("observation_date")[code].astype(float)
    px = pd.DataFrame(cols).dropna()
    px.index.name = "date"
    return px
