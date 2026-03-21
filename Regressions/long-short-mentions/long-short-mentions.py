"""
Long-Short Portfolio (2019-2023 + Jan 2024, Weekly Panel)

Long : weekly_mentions > weekly median among mentioned stocks (Dummy = 1)
Short: zero mentions that week                                (Dummy = 0)

Reg 1 (t)  : abnormal_ret_i,t   = a + b1*Dummy_i,t + b2*lag_abnormal_ret_i,t-1 + e
Reg 2 (t+1): abnormal_ret_i,t+1 = a + b1*Dummy_i,t + b2*lag_abnormal_ret_i,t-1 + e

Benchmark: IWC via yfinance. HC3 robust SEs.
"""

import pandas as pd
import yfinance as yf
import statsmodels.formula.api as smf
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent.parent   # MGFD40-Final-Project/
OUT_DIR = Path(__file__).parent
YEARS   = list(range(2019, 2024)) + [2024]   # 2019-2023 full + Jan 2024 only

# ── 1. IWC benchmark (W-FRI) ──────────────────────────────────────────────────
iwc_raw = yf.download("IWC", start="2019-01-01", end="2025-01-01",
                       auto_adjust=True, progress=False)
iwc_weekly = (
    iwc_raw["Close"].squeeze()
    .pct_change()
    .resample("W-FRI")
    .apply(lambda x: (1 + x).prod() - 1)
    .reset_index()
)
iwc_weekly.columns = ["date", "benchmark_ret"]
iwc_weekly["date"] = pd.to_datetime(iwc_weekly["date"])
print(f"IWC: {iwc_weekly['date'].min().date()} to {iwc_weekly['date'].max().date()}")

# ── 2. CRSP weekly returns (W-FRI compounded) ─────────────────────────────────
print("Loading CRSP...")
crsp = pd.read_csv(ROOT / "original_data_crsp" / "crsp_daily_20260107_044825.csv",
                   usecols=["date", "ticker", "ret"], parse_dates=["date"])
crsp = crsp[(crsp["date"].dt.year.isin(YEARS)) &
            ~((crsp["date"].dt.year == 2024) & (crsp["date"].dt.month > 1))].copy()
crsp["ret"] = pd.to_numeric(crsp["ret"], errors="coerce")
crsp = crsp[crsp["ret"] > -1].dropna(subset=["ret"])   # drop missing/delisting codes

crsp_weekly = (
    crsp.groupby(["ticker", pd.Grouper(key="date", freq="W-FRI")])["ret"]
    .apply(lambda x: (1 + x).prod() - 1)
    .reset_index()
    .rename(columns={"ret": "weekly_ret"})
)
print(f"CRSP weekly: {len(crsp_weekly):,} stock-weeks")

# ── 3. Mentions & no-mentions (all years) ─────────────────────────────────────
mentions_frames, no_mention_frames = [], []
for year in YEARS:
    m = pd.read_csv(ROOT / f"{year}_reddit_mentions" / f"reddit_mentions_{year}.csv",
                    parse_dates=["date"])
    # Jan 2024 only
    if year == 2024:
        m = m[m["date"].dt.month == 1]
    m["date"] = m["date"].dt.to_period("W-FRI").dt.end_time.dt.normalize()
    m = m.groupby(["date", "ticker"])["mentions"].sum().reset_index()
    mentions_frames.append(m)

    n = pd.read_csv(ROOT / f"{year}_reddit_mentions" / f"no_mentions_{year}.csv",
                    parse_dates=["date"])
    no_mention_frames.append(n)

mentions   = pd.concat(mentions_frames,   ignore_index=True)
no_mention = pd.concat(no_mention_frames, ignore_index=True)

# ── 4. Long leg: mentions > weekly median ─────────────────────────────────────
weekly_median = mentions.groupby("date")["mentions"].median().rename("med")
mentions = mentions.merge(weekly_median, on="date")
long_df = mentions[mentions["mentions"] > mentions["med"]][["date", "ticker"]].copy()
long_df["Dummy"] = 1
print(f"Long leg:  {len(long_df):,} stock-weeks | {long_df['date'].nunique()} weeks")

# ── 5. Short leg: no mentions ─────────────────────────────────────────────────
short_df = no_mention[["date", "ticker"]].copy()
short_df["Dummy"] = 0
print(f"Short leg: {len(short_df):,} stock-weeks")

# ── 6. Combine -> merge returns + benchmark ───────────────────────────────────
panel = pd.concat([long_df, short_df], ignore_index=True)
panel = panel.merge(crsp_weekly, on=["date", "ticker"], how="inner")
panel = panel.merge(iwc_weekly,  on="date",             how="inner")
panel["abnormal_ret"] = panel["weekly_ret"] - panel["benchmark_ret"]

panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
panel["lag_abnormal_ret"]  = panel.groupby("ticker")["abnormal_ret"].shift(1)
panel["lead_abnormal_ret"] = panel.groupby("ticker")["abnormal_ret"].shift(-1)

print(f"\nPanel: {len(panel):,} obs | {panel['ticker'].nunique()} tickers | {panel['date'].nunique()} weeks")

# ── 7. Portfolio performance summary ─────────────────────────────────────────
long_ret  = panel[panel["Dummy"] == 1].groupby("date")["weekly_ret"].mean()
short_ret = panel[panel["Dummy"] == 0].groupby("date")["weekly_ret"].mean()
ls_ret    = (long_ret - short_ret).dropna()

print("\n" + "=" * 55)
print("PORTFOLIO PERFORMANCE (2019 - Jan 2024)")
print("=" * 55)
for label, s in [("Long (mentions > median)", long_ret),
                 ("Short (no mention)",        short_ret),
                 ("L/S Spread",                ls_ret)]:
    sr = s.mean() / s.std() * (52 ** 0.5)
    print(f"  {label:<30} mean={s.mean():.4f}  Sharpe={sr:.3f}")

# ── 8. Regressions ────────────────────────────────────────────────────────────
reg1 = panel.dropna(subset=["abnormal_ret",      "Dummy", "lag_abnormal_ret"])
reg2 = panel.dropna(subset=["lead_abnormal_ret", "Dummy", "lag_abnormal_ret"])

print("\n" + "=" * 55)
print(f"REG 1 (t)   N={len(reg1):,}")
print("abnormal_ret ~ Dummy + lag_abnormal_ret")
print("=" * 55)
m1 = smf.ols("abnormal_ret ~ Dummy + lag_abnormal_ret", data=reg1).fit(cov_type="HC3")
print(m1.summary())

print("\n" + "=" * 55)
print(f"REG 2 (t+1) N={len(reg2):,}")
print("lead_abnormal_ret ~ Dummy + lag_abnormal_ret")
print("=" * 55)
m2 = smf.ols("lead_abnormal_ret ~ Dummy + lag_abnormal_ret", data=reg2).fit(cov_type="HC3")
print(m2.summary())

# ── 9. Save ───────────────────────────────────────────────────────────────────
panel.to_csv(OUT_DIR / "long_short_mentions.csv", index=False)
print(f"\nSaved: {OUT_DIR / 'long_short_mentions.csv'}")
print(f"Columns: {list(panel.columns)}")
