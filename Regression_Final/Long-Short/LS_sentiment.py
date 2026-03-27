"""
Simple Long-Short Strategy: Sentiment

Direction determined by t+1 beta from Sentiment_Regression.py results:
  beta > 0  ->  MOMENTUM:   LONG bullish, SHORT bearish
  beta < 0  ->  CONTRARIAN: LONG bearish,  SHORT bullish

Reads beta from: Regression_Final/sentiment_regression_results_2019_2024.csv
Current result : beta_Sentiment (t+1) = -0.081  ->  CONTRARIAN

- Week t: Identify stocks with bullish (sentiment_value > 0) vs bearish (< 0) sentiment
- Friday t+1 4pm: Form portfolio per direction above
- Compute weekly return Mon-Fri of week t+i (i=1 to 26)
- Period: 2019-2023 (2024 used for 2023 t+i calculations)
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = Path(__file__).resolve().parent
SENTIMENT_FP = ROOT / "Finbert_Sentiment" / "finbert_sentiment_2019_2024.csv"
CRSP_WEEKLY_FP = ROOT / "original_data_crsp" / "crsp_weekly_monday_friday_4pm_2019_2024.csv"
REGRESSION_RESULTS_FP = ROOT / "Regression_Final" / "sentiment_regression_results_2019_2024.csv"


def get_strategy_direction() -> tuple[str, float, float]:
    """Read t+1 beta from regression results and return strategy direction."""
    reg = pd.read_csv(REGRESSION_RESULTS_FP)
    row = reg[reg["model"] == "reg2_t_plus_1"].iloc[0]
    beta = float(row["beta_Sentiment"])
    pvalue = float(row["pvalue_Sentiment"])
    direction = "momentum" if beta > 0 else "contrarian"
    print(f"t+1 beta = {beta:.4f}  (p={pvalue:.4f})  ->  {direction.upper()} strategy")
    if pvalue > 0.05:
        print("  WARNING: t+1 beta not significant at 5% — strategy direction not statistically justified")
    return direction, beta, pvalue


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
	direction, beta, pvalue = get_strategy_direction()
	momentum = direction == "momentum"

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

		bullish_t = set(sent_t[sent_t["sentiment_value"] > 0]["ticker"].unique())
		bearish_t = set(sent_t[sent_t["sentiment_value"] < 0]["ticker"].unique())

		if len(bullish_t) == 0 or len(bearish_t) == 0:
			continue

		# Direction driven by t+1 beta sign:
		# momentum   (beta > 0): long bullish, short bearish
		# contrarian (beta < 0): long bearish,  short bullish
		if momentum:
			long_set, short_set = bullish_t, bearish_t
		else:
			long_set, short_set = bearish_t, bullish_t

		# Look forward up to max_forward weeks
		for offset in range(1, min(max_forward + 1, len(all_dates) - i)):
			date_forward = all_dates[i + offset]
			forward_rets = crsp[crsp["date"] == date_forward]

			long_rets  = forward_rets[forward_rets["ticker"].isin(long_set)]["weekly_ret"]
			short_rets = forward_rets[forward_rets["ticker"].isin(short_set)]["weekly_ret"]

			if len(long_rets) > 0 and len(short_rets) > 0:
				ls_ret = long_rets.mean() - short_rets.mean()
				results.append({
					"date_formation": date_t,
					"date_return": date_forward,
					"forward_weeks": offset,
					"n_long": len(long_set),
					"n_short": len(short_set),
					"long_ret": long_rets.mean(),
					"short_ret": short_rets.mean(),
					"ls_ret": ls_ret,
				})

	ls_panel = pd.DataFrame(results)
	ls_panel = ls_panel[ls_panel["date_formation"].dt.year.between(2019, 2023)].copy()

	long_label  = "Bullish" if momentum else "Bearish"
	short_label = "Bearish" if momentum else "Bullish"

	print("=" * 70)
	print(f"Long-Short Strategy: Sentiment ({'Momentum' if momentum else 'Contrarian'}) (Forward Returns t+1 to t+26)")
	print(f"LONG: {long_label} sentiment | SHORT: {short_label} sentiment")
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
	
	tag = "momentum" if momentum else "contrarian"
	out_fp = OUT_DIR / f"LS_sentiment_{tag}_forward_2019_2023.csv"
	ls_panel.to_csv(out_fp, index=False)
	print(f"\nSaved: {out_fp}")


if __name__ == "__main__":
	main()
