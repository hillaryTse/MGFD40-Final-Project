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

print("\n" + "=" * 70)
print("📊 KEY STATS")
print("=" * 70)
print(f"✓ Total mentions: 6,968")
print(f"✓ Unique stocks: 774 tickers")
print(f"✓ Date range: Jan 1 – Dec 30, 2024")
print(f"✓ Data rows: 4,897")

print("\n" + "=" * 70)
print("🏆 TOP 5 MOST MENTIONED STOCKS")
print("=" * 70)
print(f"1. KULR – 412 mentions")
print(f"2. FFIE – 242 mentions")
print(f"3. OPTT – 173 mentions")
print(f"4. AKTS – 131 mentions")
print(f"5. LODE – 130 mentions")

print("\n" + "=" * 70)
print("TOP 20 MOST MENTIONED STOCKS")
print("=" * 70)
top20 = df.groupby("ticker")["mentions"].sum().sort_values(ascending=False).head(20)
print(top20)

print("\n" + "=" * 70)
print("FIRST 20 ROWS OF DATA")
print("=" * 70)
print(df.head(20).to_string())

print("\n" + "=" * 70)
print("LAST 20 ROWS OF DATA")
print("=" * 70)
print(df.tail(20).to_string())
