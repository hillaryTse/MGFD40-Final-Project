# filter for frequency of ticker mentions based on crsp_daily_2024 and reddit_fetch_2007_2024
# date,ticker,mentions --> .csv

"""
Reddit Mention Frequency Filter
---------------------------------
Loads reddit_fetch_2007_2024.parquet and crsp_daily_2024.parquet.
Only counts mentions for tickers that appear in the CRSP data (valid tickers).

Outputs:
  - reddit_mention_counts.parquet  : daily mention counts per valid ticker
  - reddit_mention_counts.csv      : same, as CSV
"""

import pandas as pd
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir   = os.path.join(script_dir, "..")
data_dir   = os.path.join(root_dir, "Data")

REDDIT_PATH = os.path.join(data_dir, "reddit_fetch_2007_2024.parquet")
CRSP_PATH   = os.path.join(data_dir, "crsp_daily_2024.parquet")
OUT_PARQUET = os.path.join(script_dir, "reddit_mentions_2024.parquet")
OUT_CSV     = os.path.join(script_dir, "reddit_mentions_2024.csv")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading CRSP data...")
crsp = pd.read_parquet(CRSP_PATH, columns=["ticker"])
valid_tickers = set(crsp["ticker"].dropna().unique())
print(f"  {len(valid_tickers):,} unique valid tickers in CRSP")

print("Loading Reddit data...")
reddit = pd.read_parquet(REDDIT_PATH, columns=["date", "ticker"])
reddit = reddit[reddit["date"].astype(str).str.startswith("2024")]
print(f"  {len(reddit):,} total mention rows before filtering")

# ---------------------------------------------------------------------------
# Filter to valid tickers only
# ---------------------------------------------------------------------------
reddit_valid = reddit[reddit["ticker"].isin(valid_tickers)].copy()
print(f"  {len(reddit_valid):,} mention rows after filtering to valid tickers")
print(f"  {reddit['ticker'].nunique() - reddit_valid['ticker'].nunique():,} tickers dropped (not in CRSP)")

# ---------------------------------------------------------------------------
# Count mentions per ticker per day
# ---------------------------------------------------------------------------
counts = (
    reddit_valid
    .groupby(["date", "ticker"])
    .size()
    .reset_index(name="mentions")
    .sort_values(["date", "ticker"])
    .reset_index(drop=True)
)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
counts.to_parquet(OUT_PARQUET, index=False)
counts.to_csv(OUT_CSV, index=False)
print(f"\nSaved {len(counts):,} rows to:")
print(f"  {OUT_PARQUET}")
print(f"  {OUT_CSV}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
total_mentions  = counts["mentions"].sum()
unique_tickers  = counts["ticker"].nunique()
date_min        = counts["date"].min()
date_max        = counts["date"].max()

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Total mentions (valid tickers): {total_mentions:,}")
print(f"Unique tickers:                 {unique_tickers:,}")
print(f"Date range:                     {date_min} to {date_max}")

print("\nTop 10 most mentioned tickers (all time):")
top10 = counts.groupby("ticker")["mentions"].sum().sort_values(ascending=False).head(10)
for rank, (ticker, n) in enumerate(top10.items(), 1):
    print(f"  {rank:2}. {ticker:<8} {n:,}")
