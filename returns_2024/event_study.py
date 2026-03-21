"""
Event Study: Do Reddit mentions affect penny stock returns?

Uses week_month_ret_with_mentions.csv (output of returns.py)
Compares returns in:
  - Mentioned vs. not-mentioned weeks (contemporaneous)
  - Week after a mention (lagged / predictive)
"""

import pandas as pd
from pathlib import Path

OUT_DIR = Path(__file__).parent

df = pd.read_csv(OUT_DIR / 'week_month_ret_with_mentions.csv', parse_dates=['date'])
df['month'] = df['month'].astype(str)
df = df.sort_values(['permno', 'date']).reset_index(drop=True)

# ############## CONTEMPORANEOUS: mentioned vs not-mentioned weeks ##############
df['mentioned'] = (df['weekly_mentions'] > 0).astype(int)

contemp = df.groupby('mentioned')['weekly_ret'].agg(['mean', 'median', 'std', 'count'])
contemp.index = contemp.index.map({0: 'Not mentioned', 1: 'Mentioned'})
contemp.columns = ['mean_ret', 'median_ret', 'std_ret', 'n_weeks']

# ############## LAGGED: return in week *after* a mention ##############
# For each stock, shift weekly_mentions forward by 1 week
df['mentions_lag1'] = df.groupby('permno')['weekly_mentions'].shift(1)
df['mentioned_lag1'] = (df['mentions_lag1'] > 0).astype(int)

# Drop rows where lag is NaN (first week of each stock)
df_lag = df.dropna(subset=['mentions_lag1'])

lagged = df_lag.groupby('mentioned_lag1')['weekly_ret'].agg(['mean', 'median', 'std', 'count'])
lagged.index = lagged.index.map({0: 'No mention prior week', 1: 'Mention prior week'})
lagged.columns = ['mean_ret', 'median_ret', 'std_ret', 'n_weeks']

# ############## OUTPUT ##############
print("=" * 55)
print("CONTEMPORANEOUS: return in mentioned vs non-mentioned weeks")
print("=" * 55)
print(contemp.to_string())
print()
print("=" * 55)
print("LAGGED: return in week *after* a mention")
print("=" * 55)
print(lagged.to_string())
print()

# Save both tables
contemp.to_csv(OUT_DIR / 'event_contemp.csv')
lagged.to_csv(OUT_DIR / 'event_lagged.csv')
print("Saved: event_contemp.csv, event_lagged.csv")
