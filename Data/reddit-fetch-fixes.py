"""
Reads reddit_fetch_2007_2024.csv, flattens newlines in content,
and saves as reddit_fetch_2007_2024.parquet next to the CSV.
"""

import pandas as pd
import os

script_dir   = os.path.dirname(os.path.abspath(__file__))
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
