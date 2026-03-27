"""
Simple Long-Short Strategy: Sentiment (Contrarian)

Strategy:
- Week t: Identify stocks with bullish sentiment (positive) vs bearish sentiment (negative)
- Friday t+1 4pm: Form portfolio - LONG (bearish), SHORT (bullish) [CONTRARIAN]
- Compute weekly return Mon-Fri of week t+i (i=1 to 26)
- LS_ret(t+i) = avg(long returns) - avg(short returns)
- Period: 2019-2023 (2024 used for 2023 t+i calculations)
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = Path(__file__).resolve().parent
SENTIMENT_FP = ROOT / "Finbert_Sentiment" / "finbert_sentiment_2019_2024.csv"
CRSP_WEEKLY_FP = ROOT / "original_data_crsp" / "crsp_weekly_monday_friday_4pm_2019_2024.csv"


def load_weekly_sentiment() -> pd.DataFrame:
	"""Load sentiment data, aggregate to weekly and classify bullish/bearish"""
	sentiment = pd.read_csv(SENTIMENT_FP)
	
	# Convert date to datetime
	sentiment["date"] = pd.to_datetime(sentiment["date"])
	
	# Get week ending date (Friday of that week)
	sentiment["week_end"] = sentiment["date"] + pd.to_timedelta((4 - sentiment["date"].dt.dayofweek) % 7, unit='D')
	
	# Ensure Friday only
	sentiment = sentiment[sentiment["week_end"].dt.dayofweek == 4].copy()
	
	# Group by ticker-week and get aggregate sentiment
	# Use sentiment_value for aggregation (can be negative or positive)
	weekly_sent = sentiment.groupby(["week_end", "ticker"]).agg({
		"sentiment_value": "mean",
		"positive": "mean",
		"negative": "mean",
		"neutral": "mean"
	}).reset_index()
	
	weekly_sent = weekly_sent.rename(columns={"week_end": "date"})
	weekly_sent = weekly_sent.drop_duplicates(subset=["date", "ticker"])
	
	return weekly_sent


def load_crsp_weekly() -> pd.DataFrame:
	"""Load CRSP Mon-Fri weekly returns 2019-2024"""
	weekly = pd.read_csv(CRSP_WEEKLY_FP)
	if "week_end" in weekly.columns:
		weekly["week_end"] = pd.to_datetime(weekly["week_end"])
		weekly = weekly.rename(columns={"week_end": "date"})
	elif "date" in weekly.columns:
		weekly["date"] = pd.to_datetime(weekly["date"])
	else:
		raise ValueError(f"No date column found in {CRSP_WEEKLY_FP}")

	weekly = weekly[weekly["date"].dt.year.between(2019, 2024)].copy()
	# Ensure Friday only (Monday-Friday week ending)
	weekly = weekly[weekly["date"].dt.dayofweek == 4].copy()
	
	if "weekly_ret" not in weekly.columns:
		raise ValueError(f"No 'weekly_ret' column found in {CRSP_WEEKLY_FP}")
	
	return weekly[["date", "ticker", "weekly_ret"]].copy()


def main() -> None:
	sentiment = load_weekly_sentiment()
	crsp = load_crsp_weekly()

	# Get all unique dates in CRSP data
	all_dates = sorted(crsp["date"].unique())
	
	results = []
	max_forward = 26  # 6 months forward
	
	for i, date_t in enumerate(all_dates[:-1]):
		# Get stocks and their sentiment at date t
		sent_t = sentiment[sentiment["date"] == date_t]
		crsp_t = crsp[crsp["date"] == date_t]
		
		# Only consider stocks that have both sentiment and price data
		stocks_with_both = set(sent_t["ticker"].unique()) & set(crsp_t["ticker"].unique())
		sent_t = sent_t[sent_t["ticker"].isin(stocks_with_both)].copy()
		
		if len(sent_t) == 0:
			continue
		
		# Classify as bullish (positive sentiment_value) or bearish (negative sentiment_value)
		# Use median to split; or use sign of sentiment_value
		bullish_t = set(sent_t[sent_t["sentiment_value"] > 0]["ticker"].unique())
		bearish_t = set(sent_t[sent_t["sentiment_value"] < 0]["ticker"].unique())
		
		if len(bullish_t) == 0 or len(bearish_t) == 0:
			continue

		# Look forward up to max_forward weeks
		for offset in range(1, min(max_forward + 1, len(all_dates) - i)):
			date_forward = all_dates[i + offset]
			
			# Get returns at forward date
			forward_rets = crsp[crsp["date"] == date_forward]
			
			# CONTRARIAN: LONG bearish, SHORT bullish
			long_rets = forward_rets[forward_rets["ticker"].isin(bearish_t)]["weekly_ret"]
			short_rets = forward_rets[forward_rets["ticker"].isin(bullish_t)]["weekly_ret"]
			
			if len(long_rets) > 0 and len(short_rets) > 0:
				ls_ret = long_rets.mean() - short_rets.mean()
				
				results.append({
					"date_formation": date_t,
					"date_return": date_forward,
					"forward_weeks": offset,
					"n_long_bearish": len(bearish_t),
					"n_short_bullish": len(bullish_t),
					"long_ret": long_rets.mean(),
					"short_ret": short_rets.mean(),
					"ls_ret": ls_ret,
				})
	
	ls_panel = pd.DataFrame(results)
	
	# Filter to 2019-2023 formation dates (so returns through 2024)
	ls_panel = ls_panel[ls_panel["date_formation"].dt.year.between(2019, 2023)].copy()
	
	print("=" * 70)
	print(f"Long-Short Strategy: Sentiment (Contrarian) (Forward Returns t+1 to t+26)")
	print(f"LONG: Bearish sentiment | SHORT: Bullish sentiment")
	print(f"Total observations: {len(ls_panel)}")
	print(f"Date range (formation): {ls_panel['date_formation'].min().date()} to {ls_panel['date_formation'].max().date()}")
	print(f"Forward weeks range: 1 to {ls_panel['forward_weeks'].max()}")
	print("=" * 70)
	
	# Summary by forward weeks
	print("\nLS Return by Forward Weeks:")
	summary = ls_panel.groupby("forward_weeks").agg({
		"ls_ret": ["mean", "std", "count"]
	}).round(4)
	print(summary)
	
	print("\n" + "=" * 70)
	print(f"Overall Mean LS return: {ls_panel['ls_ret'].mean():.4f}")
	print(f"Overall Std LS return: {ls_panel['ls_ret'].std():.4f}")
	if ls_panel['ls_ret'].std() > 0:
		print(f"Sharpe (annualized): {(ls_panel['ls_ret'].mean() / ls_panel['ls_ret'].std()) * np.sqrt(52):.4f}")
	print("=" * 70)
	
	ls_panel.to_csv(OUT_DIR / "LS_sentiment_contrarian_forward_2019_2023.csv", index=False)
	print(f"\nSaved: {OUT_DIR / 'LS_sentiment_contrarian_forward_2019_2023.csv'}")


if __name__ == "__main__":
	main()
