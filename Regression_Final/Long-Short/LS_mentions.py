"""
Simple Long-Short Strategy: Mentions

Strategy:
- Week t: Identify stocks with mentions vs no mentions
- Friday t+1 4pm: Form portfolio - LONG (mentioned), SHORT (no mentions)
- Compute weekly return Mon-Fri of week t+i (i=1 to 26)
- LS_ret(t+i) = avg(long returns) - avg(short returns)
- Period: 2019-2023 (2024 used for 2023 t+i calculations)
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = Path(__file__).resolve().parent
CRSP_WEEKLY_FP = ROOT / "original_data_crsp" / "crsp_weekly_monday_friday_4pm_2019_2024.csv"


def load_weekly_mentions() -> pd.DataFrame:
	"""Load all weekly mentions (Mon-Fri only) 2019-2024"""
	year_to_folder = {
		2019: "2019_reddit_mentions",
		2020: "2020_reddit_mentions",
		2021: "2021_reddit_mentions",
		2022: "2022_reddit_mentions",
		2023: "2023_reddit_mentions",
		2024: "2024_Reddit_mentions",
	}

	parts = []
	for year, folder in year_to_folder.items():
		fp = ROOT / folder / f"weekly_mentions_{year}.csv"
		if not fp.exists():
			raise FileNotFoundError(f"Missing weekly mentions file: {fp}")

		df = pd.read_csv(fp)
		if "week_end" in df.columns:
			df["date"] = pd.to_datetime(df["week_end"])
		elif "date" in df.columns:
			df["date"] = pd.to_datetime(df["date"])
		else:
			raise ValueError(f"No weekly date column found in {fp}")

		keep = df[["date", "ticker"]].copy()
		parts.append(keep)

	mentions = pd.concat(parts, ignore_index=True)
	# Monday-Friday week ending date only
	mentions = mentions[mentions["date"].dt.dayofweek == 4].copy()  # Friday = 4
	mentions = mentions.drop_duplicates(subset=["date", "ticker"])
	return mentions


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
	mentions = load_weekly_mentions()
	crsp = load_crsp_weekly()

	# Get all unique dates in CRSP data
	all_dates = sorted(crsp["date"].unique())
	
	# Create mention indicator
	mention_set = set(zip(mentions["date"], mentions["ticker"]))
	
	results = []
	max_forward = 26  # 6 months forward
	
	for i, date_t in enumerate(all_dates[:-1]):
		# Get stocks mentioned at t
		mentioned_t = set(mentions[mentions["date"] == date_t]["ticker"].unique())
		all_stocks_t = set(crsp[crsp["date"] == date_t]["ticker"].unique())
		no_mention_t = all_stocks_t - mentioned_t
		
		if len(mentioned_t) == 0 or len(no_mention_t) == 0:
			continue

		# Look forward up to max_forward weeks
		for offset in range(1, min(max_forward + 1, len(all_dates) - i)):
			date_forward = all_dates[i + offset]
			
			# Get returns at forward date
			forward_rets = crsp[crsp["date"] == date_forward]
			
			long_rets = forward_rets[forward_rets["ticker"].isin(mentioned_t)]["weekly_ret"]
			short_rets = forward_rets[forward_rets["ticker"].isin(no_mention_t)]["weekly_ret"]
			
			if len(long_rets) > 0 and len(short_rets) > 0:
				ls_ret = long_rets.mean() - short_rets.mean()
				
				results.append({
					"date_formation": date_t,
					"date_return": date_forward,
					"forward_weeks": offset,
					"n_long": len(mentioned_t),
					"n_short": len(no_mention_t),
					"long_ret": long_rets.mean(),
					"short_ret": short_rets.mean(),
					"ls_ret": ls_ret,
				})
	
	ls_panel = pd.DataFrame(results)
	
	# Filter to 2019-2023 formation dates (so returns through 2024)
	ls_panel = ls_panel[ls_panel["date_formation"].dt.year.between(2019, 2023)].copy()
	
	print("=" * 70)
	print(f"Long-Short Strategy: Mentions (Forward Returns t+1 to t+26)")
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
	
	ls_panel.to_csv(OUT_DIR / "LS_mentions_forward_2019_2023.csv", index=False)
	print(f"\nSaved: {OUT_DIR / 'LS_mentions_forward_2019_2023.csv'}")


if __name__ == "__main__":
	main()
