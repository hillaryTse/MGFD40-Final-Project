import pandas as pd
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Read the CSV from the same folder as this script
csv_path = os.path.join(script_dir, "reddit_mentions_2024.csv")
df = pd.read_csv(csv_path)


print("=" * 70)
print("REDDIT MENTIONS 2024 - DATA SUMMARY")
print("=" * 70)

print(f"\nTotal rows: {len(df)}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Unique tickers: {df['ticker'].nunique()}")
print(f"Total mentions: {df['mentions'].sum()}")

total_mentions = df['mentions'].sum()
unique_tickers = df['ticker'].nunique()
date_min = df['date'].min()
date_max = df['date'].max()
data_rows = len(df)

print("\n" + "=" * 70)
print("📊 KEY STATS")
print("=" * 70)
print(f"✓ Total mentions: {total_mentions:,}")
print(f"✓ Unique stocks: {unique_tickers:,} tickers")
print(f"✓ Date range: {date_min} – {date_max}")
print(f"✓ Data rows: {data_rows:,}")

top5 = df.groupby("ticker")["mentions"].sum().sort_values(ascending=False).head(5)

print("\n" + "=" * 70)
print("🏆 TOP 5 MOST MENTIONED STOCKS")
print("=" * 70)
for rank, (ticker, count) in enumerate(top5.items(), 1):
    print(f"{rank}. {ticker} – {count:,} mentions")

print("\n" + "=" * 70)
print("TOP 20 MOST MENTIONED STOCKS")
print("=" * 70)
top20 = df.groupby("ticker")["mentions"].sum().sort_values(ascending=False).head(20)
print(top20)

print("\n" + "=" * 70)
print("BOTTOM 20 LEAST MENTIONED STOCKS")
print("=" * 70)
bottom20 = df.groupby("ticker")["mentions"].sum().sort_values(ascending=True).head(20)
print(bottom20)

print("\n" + "=" * 70)
print("FIRST 20 ROWS OF DATA")
print("=" * 70)
print(df.head(20).to_string())

print("\n" + "=" * 70)
print("LAST 20 ROWS OF DATA")
print("=" * 70)
print(df.tail(20).to_string())
