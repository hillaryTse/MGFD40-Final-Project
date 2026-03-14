"""
Reads reddit_data_2007_2024.parquet and saves as reddit_data_2007_2024.csv.
"""

import pandas as pd
import os

script_dir   = os.path.dirname(os.path.abspath(__file__))
PARQUET_PATH = os.path.join(script_dir, "reddit_data_2007_2024.parquet")
CSV_PATH     = os.path.join(script_dir, "reddit_data_2007_2024.csv")

print(f"Reading {PARQUET_PATH} ...")
df = pd.read_parquet(PARQUET_PATH)
print(f"  {len(df):,} rows loaded")

df.to_csv(CSV_PATH, index=False)
print(f"  Saved {len(df):,} rows to {CSV_PATH}")
