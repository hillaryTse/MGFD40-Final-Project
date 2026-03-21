"""
Daily and weekly returns for top 25 most-mentioned penny stocks,
merged with unigram sentiment scores.

Sentiment is aggregate per ticker (not time-varying), so it joins as a
stock-level attribute on every row.
"""

import pandas as pd
from pathlib import Path

OUT_DIR = Path(__file__).parent
DATA_DIR = Path(__file__).parent.parent

# ############## LOAD DATA ##############
dfCRSP = pd.read_parquet(DATA_DIR / 'Data' / 'crsp_daily_2024.parquet')
dfCRSP['date'] = pd.to_datetime(dfCRSP['date'])

top25 = pd.read_csv(DATA_DIR / 'Shortlisted Stocks' / 'top25_stocks.csv')
top25_tickers = set(top25['ticker'])

dfSenti = pd.read_csv(DATA_DIR / 'Shortlisted Stocks' / 'output-unigrams' / 'sentiment_summary_unigram_top25.csv')

# ############## FILTER CRSP TO TOP 25 ##############
df = dfCRSP[dfCRSP['ticker'].isin(top25_tickers)].copy()
df = df.sort_values(['permno', 'ticker', 'date']).reset_index(drop=True)

# ############## DAILY RETURNS ##############
# CRSP already provides daily ret
df_daily = df[['date', 'permno', 'ticker', 'prc', 'ret', 'mktcap']].copy()

# Join sentiment (stock-level) onto daily
df_daily = df_daily.merge(
    dfSenti[['ticker', 'mentions', 'avg_sentiment_score', 'corpus_sentiment_score',
             'bullish_pct', 'bearish_pct', 'neutral_pct', 'overall_sentiment']],
    on='ticker', how='left'
)

# ############## WEEKLY RETURNS ##############
df_eow_price = (
    df
    .groupby(['permno', 'ticker', pd.Grouper(key='date', freq='W-FRI')])['prc']
    .last()
    .reset_index()
)

df_weekly_ret = (
    df
    .groupby(['permno', 'ticker', pd.Grouper(key='date', freq='W-FRI')])['ret']
    .apply(lambda x: (1 + x).prod() - 1)
    .reset_index()
    .rename(columns={'ret': 'weekly_ret'})
)

df_weekly = df_eow_price.merge(df_weekly_ret, on=['permno', 'ticker', 'date'])

# Join sentiment (stock-level) onto weekly
df_weekly = df_weekly.merge(
    dfSenti[['ticker', 'mentions', 'avg_sentiment_score', 'corpus_sentiment_score',
             'bullish_pct', 'bearish_pct', 'neutral_pct', 'overall_sentiment']],
    on='ticker', how='left'
)

df_weekly = df_weekly.sort_values(['ticker', 'date']).reset_index(drop=True)

# ############## OUTPUT ##############
df_daily.to_csv(OUT_DIR / 'top25_daily_ret_senti.csv', index=False)
df_weekly.to_csv(OUT_DIR / 'top25_weekly_ret_senti.csv', index=False)

# ############## SUMMARY STATS ##############
print("=" * 55)
print("TOP 25 STOCKS — DAILY & WEEKLY RETURNS + SENTIMENT")
print("=" * 55)
print(f"Tickers in top25 found in CRSP: {df['ticker'].nunique()} / {len(top25_tickers)}")
print(f"Tickers missing from CRSP:      {top25_tickers - set(df['ticker'])}")
print(f"Date range:                     {df_daily['date'].min().date()} to {df_daily['date'].max().date()}")
print(f"Total daily obs:                {len(df_daily)}")
print(f"Total weekly obs:               {len(df_weekly)}")
print()
print("Daily return stats:")
print(df_daily['ret'].describe().to_string())
print()
print("Weekly return stats:")
print(df_weekly['weekly_ret'].describe().to_string())
print()
print("Sentiment scores (avg_sentiment_score) by ticker:")
print(dfSenti[['ticker', 'mentions', 'avg_sentiment_score', 'overall_sentiment']]
      .sort_values('mentions', ascending=False)
      .to_string(index=False))
print()
print("Sample (daily):")
print(df_daily.head(5).to_string(index=False))
print()
print("Sample (weekly):")
print(df_weekly.head(5).to_string(index=False))
