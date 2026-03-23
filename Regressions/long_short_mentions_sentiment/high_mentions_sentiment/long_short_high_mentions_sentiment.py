"""
Long-Short Portfolio — Mentions + Bullish Sentiment (2019-Jan 2024)

Long : top quintile mentions (>= 80th pct) AND top_label == 'positive' that week (Dummy = 1)
Short: zero mentions                                                               (Dummy = 0)

Sentiment score = avg sentiment_value (positive - negative) for stock i in week t

Reg 1 (t)  : abnormal_ret_i,t   = a + b1*SentimentScore_i,t + b2*lag_abnormal_ret_i,t-1 + e
Reg 2 (t+1): abnormal_ret_i,t+1 = a + b1*SentimentScore_i,t + b2*lag_abnormal_ret_i,t-1 + e

Benchmark: IWC via yfinance. HC3 robust SEs.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.formula.api as smf
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent.parent.parent
OUT_DIR = Path(__file__).parent
YEARS   = list(range(2019, 2024)) + [2024]   # 2019-2023 full + Jan 2024

# ── 1. IWC benchmark ──────────────────────────────────────────────────────────
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

# ── 2. CRSP weekly returns ────────────────────────────────────────────────────
print("Loading CRSP...")
crsp = pd.read_parquet(ROOT / "original_data_crsp" / "crsp_daily_20260107_044825.parquet",
                       columns=["date", "ticker", "ret"])
crsp = crsp[(crsp["date"].dt.year.isin(YEARS)) &
            ~((crsp["date"].dt.year == 2024) & (crsp["date"].dt.month > 1))].copy()
crsp["ret"] = pd.to_numeric(crsp["ret"], errors="coerce")
crsp = crsp[crsp["ret"] > -1].dropna(subset=["ret"])
crsp["log1r"] = np.log1p(crsp["ret"])
crsp_weekly = (
    crsp.groupby(["ticker", pd.Grouper(key="date", freq="W-FRI")])["log1r"]
    .sum()
    .apply(np.expm1)
    .reset_index()
    .rename(columns={"log1r": "weekly_ret"})
)

# ── 3. FinBERT posts -> weekly sentiment per stock ────────────────────────────
posts = pd.read_csv(ROOT / "Finbert_Sentiment" / "reddit_finbert_sentiment_posts.csv",
                    usecols=["date", "ticker", "top_label", "sentiment_value"],
                    parse_dates=["date"])
posts = posts[(posts["date"].dt.year.isin(YEARS)) &
              ~((posts["date"].dt.year == 2024) & (posts["date"].dt.month > 1))]
posts["week"] = posts["date"].dt.to_period("W-FRI").dt.end_time.dt.normalize()

weekly_senti = (
    posts.groupby(["week", "ticker"])
    .agg(
        sentiment_score=("sentiment_value", "mean"),
        bullish=("top_label", lambda x: (x == "positive").any())
    )
    .reset_index()
    .rename(columns={"week": "date"})
)

# ── 4. Mentions -> weekly totals ──────────────────────────────────────────────
mentions_frames, no_mention_frames = [], []
for year in YEARS:
    m = pd.read_parquet(ROOT / f"{year}_reddit_mentions" / f"reddit_mentions_{year}.parquet")
    m["date"] = pd.to_datetime(m["date"])
    if year == 2024:
        m = m[m["date"].dt.month == 1]
    m["date"] = m["date"].dt.to_period("W-FRI").dt.end_time.dt.normalize()
    mentions_frames.append(m.groupby(["date", "ticker"])["mentions"].sum().reset_index())

    n = pd.read_csv(ROOT / f"{year}_reddit_mentions" / f"no_mentions_{year}.csv",
                    parse_dates=["date"])
    no_mention_frames.append(n)

mentions   = pd.concat(mentions_frames,   ignore_index=True)
no_mention = pd.concat(no_mention_frames, ignore_index=True)

# ── 5. Long leg: top quintile mentions AND bullish ────────────────────────────
weekly_q80   = mentions.groupby("date")["mentions"].quantile(0.8).rename("q80")
mentions     = mentions.merge(weekly_q80, on="date")
high_mention = mentions[mentions["mentions"] >= mentions["q80"]][["date", "ticker"]]

long_df = high_mention.merge(weekly_senti, on=["date", "ticker"], how="inner")
long_df = long_df[long_df["bullish"] == True][["date", "ticker", "sentiment_score"]].copy()
long_df["Dummy"] = 1

# ── 6. Short leg: no mentions ─────────────────────────────────────────────────
short_df = no_mention[["date", "ticker"]].copy()
short_df["sentiment_score"] = 0.0
short_df["Dummy"] = 0

# ── 7. Panel: merge returns + benchmark ───────────────────────────────────────
panel = pd.concat([long_df, short_df], ignore_index=True)
panel = panel.merge(crsp_weekly, on=["date", "ticker"], how="inner")
panel = panel.merge(iwc_weekly,  on="date",             how="inner")
panel["abnormal_ret"] = panel["weekly_ret"] - panel["benchmark_ret"]

panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
panel["lag_abnormal_ret"]  = panel.groupby("ticker")["abnormal_ret"].shift(1)
panel["lead_abnormal_ret"] = panel.groupby("ticker")["abnormal_ret"].shift(-1)

print(f"Panel: {len(panel):,} observations | {panel['ticker'].nunique()} tickers | {panel['date'].nunique()} weeks")
print(f"Long leg: {(panel['Dummy']==1).sum()} observations | Short leg: {(panel['Dummy']==0).sum()} observations")

# ── 8. Portfolio performance ──────────────────────────────────────────────────
long_ret  = panel[panel["Dummy"] == 1].groupby("date")["weekly_ret"].mean()
short_ret = panel[panel["Dummy"] == 0].groupby("date")["weekly_ret"].mean()
ls_ret    = (long_ret - short_ret).dropna()

print("\n" + "=" * 55)
print("PORTFOLIO PERFORMANCE (2019 - Jan 2024)")
print("=" * 55)
for label, s in [("Long (high mention + bullish)", long_ret),
                 ("Short (no mention)",             short_ret),
                 ("L/S Spread",                     ls_ret)]:
    sr = s.mean() / s.std() * (52 ** 0.5)
    print(f"  {label:<35} mean={s.mean():.4f}  Sharpe={sr:.3f}")

# ── 9. Regressions ────────────────────────────────────────────────────────────
reg1 = panel.dropna(subset=["abnormal_ret",      "sentiment_score", "lag_abnormal_ret"])
reg2 = panel.dropna(subset=["lead_abnormal_ret", "sentiment_score", "lag_abnormal_ret"])

print("\n" + "=" * 55)
print(f"REG 1 (t)   N={len(reg1):,}")
print("abnormal_ret ~ sentiment_score + lag_abnormal_ret")
print("=" * 55)
m1 = smf.ols("abnormal_ret ~ sentiment_score + lag_abnormal_ret", data=reg1).fit(cov_type="HC3")
print(m1.summary())

print("\n" + "=" * 55)
print(f"REG 2 (t+1) N={len(reg2):,}")
print("lead_abnormal_ret ~ sentiment_score + lag_abnormal_ret")
print("=" * 55)
m2 = smf.ols("lead_abnormal_ret ~ sentiment_score + lag_abnormal_ret", data=reg2).fit(cov_type="HC3")
print(m2.summary())

# ── 10. Save ──────────────────────────────────────────────────────────────────
panel.to_csv(OUT_DIR / "long_short_high_mentions_sentiment.csv", index=False)
print(f"\nSaved: {OUT_DIR / 'long_short_high_mentions_sentiment.csv'}")
print(f"Columns: {list(panel.columns)}")
