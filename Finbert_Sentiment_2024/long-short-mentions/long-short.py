"""
Long-Short Portfolio + Event Study Regressions (2024)

Portfolio (fixed):
  Long  = top 25 most-mentioned penny stocks (2024 Reddit mentions/top25_most_mentioned_2024.csv)
  Short = penny stocks with no Reddit mentions (2024 Reddit mentions/no_mentions_2024.csv)
  Equal-weighted within each leg; rebalanced every Friday close.

Benchmark: IWC (iShares Micro-Cap ETF) via yfinance

Regression 1 — Contemporaneous (Formula 1):
  ls_abnormal_ret_t = α + β₁·Dummy_t + β₂·lag_ls_abnormal_ret_t + ε_t
  where Dummy_t = 1 if top25 long basket had any Reddit mentions in week t

Regression 2 — Sentiment (Formula 2):
  ls_abnormal_ret_t = α + β₁·Sentiment_t + ε_t
  where Sentiment_t = equal-weighted avg FinBERT sentiment of top25 stocks in week t

Regressions 3 & 4 repeat Formulas 1 & 2 with lead_ls_abnormal_ret (t+1) as the LHS.
"""

import pandas as pd
import yfinance as yf
import statsmodels.formula.api as smf
from pathlib import Path

ROOT         = Path(__file__).parent.parent.parent   # MGFD40-Final-Project/
SENTI_DIR    = Path(__file__).parent.parent          # Finbert_Sentiment_2024/
MENTIONS_DIR = ROOT / "2024 Reddit mentions"
OUT_DIR      = Path(__file__).parent

# ############## 1. LOAD LONG / SHORT TICKERS (fixed) ##############
long_tickers  = set(pd.read_csv(MENTIONS_DIR / "top25_most_mentioned_2024.csv")["ticker"])
short_tickers = set(pd.read_csv(MENTIONS_DIR / "no_mentions_2024.csv")["ticker"])

print(f"Long  (top25 mentioned): {len(long_tickers)} tickers")
print(f"Short (no mentions):     {len(short_tickers)} tickers")

# ############## 2. BENCHMARK: IWC weekly returns ##############
# end="2025-01-01" because yfinance end date is exclusive
iwc_raw = yf.download("IWC", start="2024-01-01", end="2025-01-01",
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

print(f"\nIWC benchmark: {iwc_weekly['date'].min().date()} to {iwc_weekly['date'].max().date()}")

# ############## 3. LOAD STOCK RETURNS ##############
df = pd.read_csv(ROOT / "returns" / "week_month_ret_with_mentions.csv",
                 parse_dates=["date"])
df = df[["date", "ticker", "permno", "weekly_ret", "weekly_mentions"]].copy()

long_df  = df[df["ticker"].isin(long_tickers)].copy()
short_df = df[df["ticker"].isin(short_tickers)].copy()

print(f"Long  tickers in CRSP: {long_df['ticker'].nunique()} / {len(long_tickers)}")
print(f"Short tickers in CRSP: {short_df['ticker'].nunique()} / {len(short_tickers)}")

# ############## 4. BUILD PORTFOLIO TIME SERIES ##############
long_weekly  = long_df.groupby("date")["weekly_ret"].mean().rename("long_ret")
short_weekly = short_df.groupby("date")["weekly_ret"].mean().rename("short_ret")

# Dummy_t: 1 if any top25 ticker had mentions that week
dummy_weekly = (
    long_df.groupby("date")["weekly_mentions"]
    .sum()
    .gt(0)
    .astype(int)
    .rename("Dummy")
)

# Diagnostics
long_count  = long_df.groupby("date")["ticker"].count().rename("long_n_stocks")
short_count = short_df.groupby("date")["ticker"].count().rename("short_n_stocks")

portfolio = pd.concat([long_weekly, short_weekly, dummy_weekly,
                       long_count, short_count], axis=1)
portfolio = portfolio.dropna(subset=["long_ret", "short_ret"])
portfolio["ls_ret"] = portfolio["long_ret"] - portfolio["short_ret"]

# Merge benchmark → compute abnormal returns
portfolio = portfolio.reset_index().merge(iwc_weekly, on="date", how="inner")
portfolio["ls_abnormal_ret"]    = portfolio["ls_ret"]    - portfolio["benchmark_ret"]
portfolio["long_abnormal_ret"]  = portfolio["long_ret"]  - portfolio["benchmark_ret"]
portfolio["short_abnormal_ret"] = portfolio["short_ret"] - portfolio["benchmark_ret"]

# Lagged and lead portfolio abnormal returns
portfolio = portfolio.sort_values("date").reset_index(drop=True)
portfolio["lag_ls_abnormal_ret"]  = portfolio["ls_abnormal_ret"].shift(1)   # t-1 → control
portfolio["lead_ls_abnormal_ret"] = portfolio["ls_abnormal_ret"].shift(-1)  # t+1 → predictive LHS

# ############## 5. WEEKLY SENTIMENT (time-varying, top25 tickers each week) ##############
posts = pd.read_csv(SENTI_DIR / "reddit_finbert_sentiment_posts.csv",
                    parse_dates=["date"],
                    usecols=["date", "ticker", "sentiment_value"])
posts = posts[(posts["ticker"].isin(long_tickers)) & (posts["date"].dt.year == 2024)].copy()

# Aggregate to W-FRI week — same grouper as CRSP
weekly_senti = (
    posts.groupby(pd.Grouper(key="date", freq="W-FRI"))["sentiment_value"]
    .mean()
    .reset_index()
    .rename(columns={"sentiment_value": "weekly_sentiment"})
)

portfolio = portfolio.merge(weekly_senti, on="date", how="left")

# ############## 6. DIAGNOSTICS ##############
print(f"\nPortfolio weeks total:         {len(portfolio)}")
print(f"Avg long  stocks/week:         {portfolio['long_n_stocks'].mean():.1f}")
print(f"Avg short stocks/week:         {portfolio['short_n_stocks'].mean():.1f}")
print(f"Weeks with mentions (Dummy=1): {portfolio['Dummy'].sum()}")
print(f"Weeks with sentiment data:     {portfolio['weekly_sentiment'].notna().sum()}")

# ############## 7. PORTFOLIO PERFORMANCE SUMMARY ##############
print("\n" + "=" * 60)
print("WEEKLY PORTFOLIO PERFORMANCE — TOP25 L/S (2024)")
print("=" * 60)
print(portfolio[["long_ret", "short_ret", "ls_ret", "ls_abnormal_ret"]].describe().to_string())

for col, label in [("long_ret", "Long (top25)"), ("short_ret", "Short (no mentions)"),
                   ("ls_ret", "L/S Spread"), ("ls_abnormal_ret", "L/S Abnormal")]:
    sr = portfolio[col].mean() / portfolio[col].std() * (52 ** 0.5)
    print(f"Annualised Sharpe ({label}): {sr:.3f}")

# ############## 8. REGRESSION 1 — CONTEMPORANEOUS FORMULA 1 ##############
# ls_abnormal_ret_t = α + β₁·Dummy_t + β₂·lag_ls_abnormal_ret_t + ε_t
reg1 = portfolio.dropna(subset=["ls_abnormal_ret", "Dummy", "lag_ls_abnormal_ret"])

print("\n" + "=" * 60)
print("REGRESSION 1 — CONTEMPORANEOUS (Formula 1)")
print("ls_abnormal_ret ~ Dummy + lag_ls_abnormal_ret")
print(f"N = {len(reg1)} weeks")
print("=" * 60)
m1 = smf.ols("ls_abnormal_ret ~ Dummy + lag_ls_abnormal_ret", data=reg1).fit(cov_type="HC3")
print(m1.summary())

# ############## 9. REGRESSION 2 — CONTEMPORANEOUS FORMULA 2 ##############
# ls_abnormal_ret_t = α + β₁·Sentiment_t + ε_t
reg2 = portfolio.dropna(subset=["ls_abnormal_ret", "weekly_sentiment"])

print("\n" + "=" * 60)
print("REGRESSION 2 — CONTEMPORANEOUS SENTIMENT (Formula 2)")
print("ls_abnormal_ret ~ weekly_sentiment")
print(f"N = {len(reg2)} weeks")
print("=" * 60)
m2 = smf.ols("ls_abnormal_ret ~ weekly_sentiment", data=reg2).fit(cov_type="HC3")
print(m2.summary())

# ############## 10. REGRESSION 3 — PREDICTIVE FORMULA 1 ##############
# ls_abnormal_ret_{t+1} = α + β₁·Dummy_t + β₂·lag_ls_abnormal_ret_t + ε_t
reg3 = portfolio.dropna(subset=["lead_ls_abnormal_ret", "Dummy", "lag_ls_abnormal_ret"])

print("\n" + "=" * 60)
print("REGRESSION 3 — PREDICTIVE (Formula 1, t+1)")
print("lead_ls_abnormal_ret ~ Dummy + lag_ls_abnormal_ret")
print(f"N = {len(reg3)} weeks")
print("=" * 60)
m3 = smf.ols("lead_ls_abnormal_ret ~ Dummy + lag_ls_abnormal_ret", data=reg3).fit(cov_type="HC3")
print(m3.summary())

# ############## 11. REGRESSION 4 — PREDICTIVE FORMULA 2 ##############
# ls_abnormal_ret_{t+1} = α + β₁·Sentiment_t + ε_{t+1}
reg4 = portfolio.dropna(subset=["lead_ls_abnormal_ret", "weekly_sentiment"])

print("\n" + "=" * 60)
print("REGRESSION 4 — PREDICTIVE SENTIMENT (Formula 2, t+1)")
print("lead_ls_abnormal_ret ~ weekly_sentiment")
print(f"N = {len(reg4)} weeks")
print("=" * 60)
m4 = smf.ols("lead_ls_abnormal_ret ~ weekly_sentiment", data=reg4).fit(cov_type="HC3")
print(m4.summary())

# ############## 12. OUTPUT ##############
portfolio.to_csv(OUT_DIR / "long_short_portfolio.csv", index=False)
print(f"\nSaved: {OUT_DIR / 'long_short_portfolio.csv'}")
print(f"Columns: {list(portfolio.columns)}")
