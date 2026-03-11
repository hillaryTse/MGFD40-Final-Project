import pandas as pd
import os

os.chdir(r'c:\Users\hillq\Projects\MGFD40 Project\MGFD40')

dfLink =  pd.read_parquet('crsp_compu_link_table_20260107_045442.parquet')
dfCRSP = pd.read_parquet('crsp_daily_20260107_044825.parquet')
dfCompustat = pd.read_parquet('compustat_quarterly_20260202_014720.parquet')

# Filter: year 2024, price under $2
# Note: prc can be negative (bid/ask average) — use abs value for filtering
df2024 = dfCRSP[
    (dfCRSP['date'].dt.year == 2024) &
    (dfCRSP['prc'].abs() < 2)
].copy()

# Compute market cap in thousands: |prc| * shrout  (shrout is already in thousands of shares)
df2024['mktcap'] = df2024['prc'].abs() * df2024['shrout']

# Select and order columns
result = df2024[['date', 'permno', 'ticker', 'ret', 'prc', 'mktcap']]

print(result.shape)
print(result.head())

result.to_parquet('crsp_daily_2024.parquet', index=False)