"""
<<<<<<< HEAD
Reads reddit_data_2007_2024.parquet, keeps only rows where the ticker
is mentioned with a dollar sign ($TICKER) in the subject or content,
then saves cleaned data as both .parquet and .csv.
=======
Reads reddit_fetch_2007_2024.csv, flattens newlines in content,
and saves as reddit_fetch_2007_2024.parquet next to the CSV.
>>>>>>> aa318d1 (refetch reddit mentions 2007-2024)
"""

import pandas as pd
import os

script_dir   = os.path.dirname(os.path.abspath(__file__))
<<<<<<< HEAD
PARQUET_PATH = os.path.join(script_dir, "reddit_data_2007_2024.parquet")
CSV_PATH     = os.path.join(script_dir, "reddit_data_2007_2024.csv")

print(f"Reading {PARQUET_PATH} ...")
df = pd.read_parquet(PARQUET_PATH)
print(f"  {len(df):,} rows loaded")

# Keep only rows where $TICKER appears in subject or content (vectorized per ticker)
import re

subject = df['subject'].fillna('')
content = df['content'].fillna('')
mask = pd.Series(False, index=df.index)

for ticker, idx in df.groupby('ticker').groups.items():
    pat = r'\$' + re.escape(str(ticker))
    mask[idx] = (
        subject[idx].str.contains(pat, case=False, regex=True, na=False) |
        content[idx].str.contains(pat, case=False, regex=True, na=False)
    )
df_clean = df[mask].reset_index(drop=True)
print(f"  {len(df_clean):,} rows kept after filtering to $TICKER mentions ({len(df) - len(df_clean):,} dropped)")

df_clean.to_parquet(PARQUET_PATH, index=False)
print(f"  Saved {len(df_clean):,} rows to {PARQUET_PATH}")

df_clean.to_csv(CSV_PATH, index=False)
print(f"  Saved {len(df_clean):,} rows to {CSV_PATH}")
=======
CSV_PATH     = os.path.join(script_dir, "reddit_fetch_2007_2024.csv")
PARQUET_PATH = os.path.join(script_dir, "reddit_fetch_2007_2024.parquet")

print(f"Reading {CSV_PATH} ...")
df = pd.read_csv(CSV_PATH, engine="python")
print(f"  {len(df):,} rows loaded")

if "content" in df.columns:
    df["content"] = (
        df["content"]
        .astype(str)
        .str.replace("\r\n", " ", regex=False)
        .str.replace("\r",   " ", regex=False)
        .str.replace("\n",   " ", regex=False)
    )

df.to_parquet(PARQUET_PATH, index=False)
print(f"  Saved {len(df):,} rows to {PARQUET_PATH}")
>>>>>>> aa318d1 (refetch reddit mentions 2007-2024)
