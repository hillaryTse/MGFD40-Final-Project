"""
For each year 2019-2023, compute weekly no-mention tickers.

Logic:
  Universe  = CRSP tickers active in that week (have at least one trading day Mon-Fri)
  Mentioned = tickers with at least one Reddit post Mon-Fri that week
  No-mention = Universe - Mentioned

Output: date (W-FRI Friday label), ticker
Saved to: All_Year_Reddit_Data/{year}_reddit_mentions/reddit_mentions_{year}.csv
"""

from pathlib import Path
import pandas as pd

ROOT      = Path(__file__).resolve().parent.parent
DATA_DIR  = Path(__file__).parent
REDDIT_PATH = DATA_DIR / "Reddit_Data_All_Years" / "reddit_data_2013_2024.csv"
CRSP_PATH   = ROOT / "original_data_crsp" / "crsp_daily_20260107_044825.csv"

YEARS = range(2019, 2024)

# ── 1. Load CRSP (date, ticker), filter to 2019-2023, Mon-Fri trading days ──
print("Loading CRSP...")
crsp = pd.read_csv(CRSP_PATH, usecols=["date", "ticker"], parse_dates=["date"])
crsp = crsp[crsp["date"].dt.year.isin(YEARS)]
crsp = crsp[crsp["date"].dt.dayofweek < 5]          # Mon=0 … Fri=4
crsp = crsp.dropna(subset=["ticker"]).drop_duplicates()

# Assign W-FRI week label
crsp["week"] = crsp["date"] + pd.offsets.Week(weekday=4) - pd.offsets.Week(weekday=4)
crsp["week"] = crsp["date"].dt.to_period("W-FRI").dt.end_time.dt.normalize()
print(f"  CRSP stock-days: {len(crsp):,}  |  unique tickers: {crsp['ticker'].nunique()}")

# ── 2. Load Reddit posts, filter to 2019-2023, Mon-Fri only ──
print("Loading Reddit posts...")
reddit = pd.read_csv(REDDIT_PATH, usecols=["date", "ticker"], parse_dates=["date"])
reddit = reddit[reddit["date"].dt.year.isin(YEARS)]
reddit = reddit[reddit["date"].dt.dayofweek < 5]     # Mon-Fri only
reddit = reddit.dropna(subset=["ticker"]).drop_duplicates()

# Assign W-FRI week label (same logic as CRSP)
reddit["week"] = reddit["date"].dt.to_period("W-FRI").dt.end_time.dt.normalize()
print(f"  Reddit posts (Mon-Fri, 2019-2023): {len(reddit):,}")

# ── 3. Per-week sets ──
# CRSP: set of active tickers per week
crsp_weekly  = crsp.groupby("week")["ticker"].apply(set)

# Reddit: set of mentioned tickers per week
reddit_weekly = reddit.groupby("week")["ticker"].apply(set)

# ── 4. For each year, compute no-mention tickers and save ──
for year in YEARS:
    weeks_in_year = [w for w in crsp_weekly.index if w.year == year]

    rows = []
    for week in weeks_in_year:
        active   = crsp_weekly.get(week, set())
        mentioned = reddit_weekly.get(week, set())
        no_mention = active - mentioned
        for ticker in sorted(no_mention):
            rows.append({"date": week.date(), "ticker": ticker})

    out_df = pd.DataFrame(rows, columns=["date", "ticker"])

    out_dir  = ROOT / f"{year}_reddit_mentions"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"no_mentions_{year}.csv"
    out_df.to_csv(out_path, index=False)
    print(f"  {year}: {len(weeks_in_year)} weeks | {len(out_df):,} no-mention stock-weeks -> {out_path}")

print("\nDone.")
