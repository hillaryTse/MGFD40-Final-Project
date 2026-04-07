"""
Weekly sentiment regressions (2019-2023)

Signal: Q5 dummy = 1 if stock is in top quintile of sentiment_value that week, 0 otherwise.
        Quintiles assigned within all stocks with sentiment each week.

Reg 1 (t):   abnormal_ret_i,t   = a + b1*Q5_i,t + e
Reg 2 (t+1): abnormal_ret_i,t+1 = a + b1*Q5_i,t + b2*lag_abnormal_ret_i,t-1 + e

sentiment_value = FinBERT positive score - negative score, ranging from -1 to +1.
Q5 captures the most bullish stocks each week (top 20% by sentiment_value).
abnormal_ret = weekly_ret - IWC benchmark return (Russell Microcap ETF)
HC3 heteroscedasticity-robust standard errors used throughout.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import yfinance as yf


# Project root is two levels up from this script
ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent
CRSP_WEEKLY_FP = ROOT / "original_data_crsp" / "crsp_weekly_monday_friday_4pm_2019_2024.csv"
SENTIMENT_FP = ROOT / "Finbert_Sentiment" / "finbert_sentiment_2019_2024.csv"


def load_sentiment() -> pd.DataFrame:
	"""Load FinBERT sentiment scores and return weekly sentiment_value per ticker."""
	if not SENTIMENT_FP.exists():
		raise FileNotFoundError(f"Missing sentiment file: {SENTIMENT_FP}")

	df = pd.read_csv(SENTIMENT_FP)

	# Normalise date column name
	if "date" in df.columns:
		df["date"] = pd.to_datetime(df["date"])
	elif "week_end" in df.columns:
		df["date"] = pd.to_datetime(df["week_end"])
	else:
		raise ValueError(f"No date column found in {SENTIMENT_FP}")

	keep = df[["date", "ticker", "sentiment_value"]].copy()
	keep["sentiment_value"] = pd.to_numeric(keep["sentiment_value"], errors="coerce").fillna(0)

	return keep[["date", "ticker", "sentiment_value"]]


def assign_q5(sentiment: pd.DataFrame) -> pd.DataFrame:
	"""Assign Q5 dummy (1 = top quintile by sentiment_value) within each week.

	Quintiles sorted on sentiment_value so Q5 = most bullish stocks.
	Weeks with fewer than 5 stocks are assigned Q5 = 0 (insufficient for quintile split).
	"""
	q5_list = []
	for _, grp in sentiment.groupby("date"):
		if len(grp) < 5:
			# Not enough stocks to form 5 quintiles - assign all to baseline
			q5_list.append(pd.Series(0, index=grp.index))
		else:
			quintile = pd.qcut(grp["sentiment_value"], q=5, labels=False, duplicates="drop")
			# Q5 = 1 for stocks in the highest sentiment quintile, 0 otherwise
			q5_list.append((quintile == quintile.max()).astype(int))
	sentiment = sentiment.copy()
	sentiment["Q5"] = pd.concat(q5_list).reindex(sentiment.index).fillna(0).astype(int)
	return sentiment


def load_iwc_weekly(min_date: pd.Timestamp, max_date: pd.Timestamp) -> pd.DataFrame:
	"""Download IWC (iShares Micro-Cap ETF) weekly returns as the Russell Microcap benchmark."""
	# Add buffer so weekly resampling covers all formation dates
	start = (min_date - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
	end   = (max_date + pd.Timedelta(days=14)).strftime("%Y-%m-%d")

	iwc_raw = yf.download("IWC", start=start, end=end, auto_adjust=True, progress=False)
	if iwc_raw.empty:
		raise RuntimeError("Failed to download IWC data from yfinance.")

	# Compound daily returns into Mon-Fri weekly returns ending on Friday
	iwc_weekly = (
		iwc_raw["Close"].squeeze()
		.pct_change()
		.resample("W-FRI")
		.apply(lambda x: (1 + x).prod() - 1)
		.reset_index()
	)
	iwc_weekly.columns = ["date", "benchmark_ret"]
	iwc_weekly["date"] = pd.to_datetime(iwc_weekly["date"])
	return iwc_weekly


def load_crsp_weekly() -> pd.DataFrame:
	"""Load pre-computed Mon-Fri weekly returns from CRSP (2019-2024)."""
	weekly = pd.read_csv(CRSP_WEEKLY_FP)
	if "week_end" in weekly.columns:
		weekly["week_end"] = pd.to_datetime(weekly["week_end"])
		weekly = weekly.rename(columns={"week_end": "date"})
	elif "date" in weekly.columns:
		weekly["date"] = pd.to_datetime(weekly["date"])
	else:
		raise ValueError(f"No date column found in {CRSP_WEEKLY_FP}")

	weekly = weekly[weekly["date"].dt.year.between(2019, 2024)].copy()

	if "weekly_ret" in weekly.columns:
		weekly_join = weekly[["date", "ticker", "weekly_ret"]].copy()
	else:
		raise ValueError(f"No 'weekly_ret' column found in {CRSP_WEEKLY_FP}")

	return weekly_join


def main() -> None:
	sentiment = load_sentiment()
	# Assign Q5 dummy based on sentiment_value quintile within each week
	sentiment = assign_q5(sentiment)

	min_date = sentiment["date"].min()
	max_date = sentiment["date"].max()

	crsp_weekly = load_crsp_weekly()
	iwc_weekly  = load_iwc_weekly(min_date, max_date)

	# Merge signal, returns, and benchmark; inner join keeps only stocks present in all three
	panel = sentiment.merge(crsp_weekly, on=["date", "ticker"], how="inner")
	panel = panel.merge(iwc_weekly, on="date", how="inner")
	# Abnormal return = stock return minus Russell Microcap benchmark return
	panel["abnormal_ret"] = panel["weekly_ret"] - panel["benchmark_ret"]

	panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
	# Lagged abnormal return (t-1) controls for momentum/mean reversion in Reg 2
	panel["lag_abnormal_ret"]  = panel.groupby("ticker")["abnormal_ret"].shift(1)
	# Lead abnormal return (t+1) is the dependent variable in Reg 2
	panel["lead_abnormal_ret"] = panel.groupby("ticker")["abnormal_ret"].shift(-1)

	# Drop rows with missing values for each regression separately
	reg1 = panel.dropna(subset=["abnormal_ret", "Q5"])
	reg2 = panel.dropna(subset=["lead_abnormal_ret", "Q5", "lag_abnormal_ret"])

	print("=" * 70)
	print(f"Panel observations: {len(panel):,}")
	print(f"Date range: {panel['date'].min().date()} to {panel['date'].max().date()}")
	print("=" * 70)

	# Reg 1: contemporaneous effect - does top-sentiment Q5 predict same-week abnormal returns?
	print("\nREG 1 (t): abnormal_ret ~ Q5")
	print(f"N = {len(reg1):,}")
	m1 = smf.ols("abnormal_ret ~ Q5", data=reg1).fit(cov_type="HC3")
	print(m1.summary())

	# Reg 2: predictive effect - does top-sentiment Q5 predict next-week abnormal returns?
	print("\nREG 2 (t+1): lead_abnormal_ret ~ Q5 + lag_abnormal_ret")
	print(f"N = {len(reg2):,}")
	m2 = smf.ols("lead_abnormal_ret ~ Q5 + lag_abnormal_ret", data=reg2).fit(cov_type="HC3")
	print(m2.summary())

	# Save full panel and regression coefficient summary
	# beta_Sentiment column name preserved for compatibility with LS_sentiment.py
	panel.to_csv(OUT_DIR / "sentiment_weekly_panel_2019_2023.csv", index=False)
	coef = pd.DataFrame(
		{
			"model":           ["reg1_t", "reg2_t_plus_1"],
			"beta_Sentiment":  [m1.params.get("Q5", np.nan),  m2.params.get("Q5", np.nan)],
			"pvalue_Sentiment":[m1.pvalues.get("Q5", np.nan), m2.pvalues.get("Q5", np.nan)],
			"N":               [int(m1.nobs), int(m2.nobs)],
			"r2":              [m1.rsquared, m2.rsquared],
		}
	)
	coef.to_csv(OUT_DIR / "sentiment_regression_results_2019_2023.csv", index=False)

	print("\nSaved:")
	print(OUT_DIR / "sentiment_weekly_panel_2019_2023.csv")
	print(OUT_DIR / "sentiment_regression_results_2019_2023.csv")


if __name__ == "__main__":
	main()
