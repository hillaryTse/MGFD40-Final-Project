"""
Calculate weekly and monthly returns for 2024
Merge reddit_mentions_2024.parquet and crsp_daily_2024.parquet -- FREQUENCY and CRSP data

Returns are computed from CRSP 2024 daily data (all trading days) before merging with Reddit mentions.
Reddit mentions are aggregated to weekly and monthly frequency separately, then merged.
"""

import pandas as pd
from pathlib import Path

OUT_DIR = Path(__file__).parent

dfCRSP = pd.read_parquet(Path(__file__).parent.parent / 'Data' / 'crsp_daily_2024.parquet')
dfMentions = pd.read_parquet(Path(__file__).parent.parent / '2024 Reddit mentions' / 'reddit_mentions_2024.parquet')

dfMentions['date'] = pd.to_datetime(dfMentions['date'])
dfCRSP['date'] = pd.to_datetime(dfCRSP['date'])

# ############## WEEKLY RETURNS (from CRSP only — all trading days) ##############
df_eow_price = (
    dfCRSP
    .groupby(['permno', 'ticker', pd.Grouper(key='date', freq='W-FRI')])['prc']
    .last()
    .reset_index()
)

df_weekly_ret = (
    dfCRSP
    .groupby(['permno', 'ticker', pd.Grouper(key='date', freq='W-FRI')])['ret']
    .apply(lambda x: (1 + x).prod() - 1)
    .reset_index()
    .rename(columns={'ret': 'weekly_ret'})
)

df_weekly = df_eow_price.merge(df_weekly_ret, on=['permno', 'ticker', 'date'])
df_weekly['month'] = df_weekly['date'].dt.to_period('M')

# ############## MONTHLY RETURNS (from CRSP only — all trading days) ##############
df_monthly_ret = (
    dfCRSP
    .groupby(['permno', 'ticker', pd.Grouper(key='date', freq='ME')])['ret']
    .apply(lambda x: (1 + x).prod() - 1)
    .reset_index()
    .rename(columns={'ret': 'monthly_ret', 'date': 'month'})
)
df_monthly_ret['month'] = df_monthly_ret['month'].dt.to_period('M')

df_week_month_ret = df_weekly.merge(df_monthly_ret, on=['permno', 'ticker', 'month'])
df_week_month_ret = df_week_month_ret[['date', 'month', 'permno', 'ticker', 'prc', 'weekly_ret', 'monthly_ret']]
df_week_month_ret = df_week_month_ret.sort_values(by='date', ascending=True)

# ############## AGGREGATE REDDIT MENTIONS ##############
# Use same W-FRI grouper as CRSP so week-end labels align exactly
# (pd.offsets.Week would misassign Friday mentions to the following week)
weekly_mentions = (
    dfMentions
    .groupby(['ticker', pd.Grouper(key='date', freq='W-FRI')])['mentions']
    .sum()
    .reset_index()
    .rename(columns={'mentions': 'weekly_mentions'})
)

dfMentions['month'] = dfMentions['date'].dt.to_period('M')

monthly_mentions = (
    dfMentions
    .groupby(['ticker', 'month'])['mentions']
    .sum()
    .reset_index()
    .rename(columns={'mentions': 'monthly_mentions'})
)

# ############## MERGE RETURNS WITH MENTIONS ##############
df_final = df_week_month_ret.merge(weekly_mentions, on=['ticker', 'date'], how='left')
df_final = df_final.merge(monthly_mentions, on=['ticker', 'month'], how='left')
df_final['weekly_mentions'] = df_final['weekly_mentions'].fillna(0).astype(int)
df_final['monthly_mentions'] = df_final['monthly_mentions'].fillna(0).astype(int)
df_final = df_final.sort_values(by='date', ascending=True).reset_index(drop=True)

# ############## OUTPUT ##############
df_week_month_ret.to_csv(OUT_DIR / 'week_month_ret.csv', index=False)
df_final.to_csv(OUT_DIR / 'week_month_ret_with_mentions.csv', index=False)

# ############## SUMMARY STATS ##############
print("=" * 50)
print("SUMMARY STATS")
print("=" * 50)
print(f"Unique stocks:         {df_final['permno'].nunique()}")
print(f"Unique tickers:        {df_final['ticker'].nunique()}")
print(f"Date range:            {df_final['date'].min().date()} to {df_final['date'].max().date()}")
print(f"Total stock-weeks:     {len(df_final)}")
print(f"Weeks with mentions:   {(df_final['weekly_mentions'] > 0).sum()} ({(df_final['weekly_mentions'] > 0).mean():.1%})")
print()
print("Weekly return stats:")
print(df_final['weekly_ret'].describe().to_string())
print()
print("Monthly return stats:")
print(df_final[['month', 'permno', 'monthly_ret']].drop_duplicates(subset=['month', 'permno'])['monthly_ret'].describe().to_string())
print()
print("Weekly mentions stats (weeks with any mention):")
print(df_final.loc[df_final['weekly_mentions'] > 0, 'weekly_mentions'].describe().to_string())
print()
print(df_final.head(10))
