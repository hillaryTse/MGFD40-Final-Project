"""
Weekly sentiment regressions (2019-2024)

Reg 1 (t):   abnormal_ret_i,t   = a + b1*Sentiment_i,t + e
Reg 2 (t+1): abnormal_ret_i,t+1 = a + b1*Sentiment_i,t + b2*lag_abnormal_ret_i,t-1 + e
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import yfinance as yf


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent
CRSP_WEEKLY_FP = ROOT / "original_data_crsp" / "crsp_weekly_monday_friday_4pm_2019_2024.csv"
SENTIMENT_FP = ROOT / "Finbert_Sentiment" / "finbert_sentiment_2019_2024.csv"


def load_sentiment() -> pd.DataFrame:
	if not SENTIMENT_FP.exists():
		raise FileNotFoundError(f"Missing sentiment file: {SENTIMENT_FP}")
	
	df = pd.read_csv(SENTIMENT_FP)

	# Handle date column - try common date column names
	if "date" in df.columns:
		df["date"] = pd.to_datetime(df["date"])
	elif "week_end" in df.columns:
		df["date"] = pd.to_datetime(df["week_end"])
	else:
		raise ValueError(f"No date column found in {SENTIMENT_FP}")

	# Keep relevant columns - use sentiment_value from finbert output
	keep = df[["date", "ticker", "sentiment_value"]].copy()
	keep["Sentiment"] = pd.to_numeric(keep["sentiment_value"], errors="coerce").fillna(0)
	
	return keep[["date", "ticker", "Sentiment"]]


def load_iwc_weekly(min_date: pd.Timestamp, max_date: pd.Timestamp) -> pd.DataFrame:
	start = (min_date - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
	end = (max_date + pd.Timedelta(days=14)).strftime("%Y-%m-%d")

	iwc_raw = yf.download("IWC", start=start, end=end, auto_adjust=True, progress=False)
	if iwc_raw.empty:
		raise RuntimeError("Failed to download IWC data from yfinance.")

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
	weekly = pd.read_csv(CRSP_WEEKLY_FP)
	if "week_end" in weekly.columns:
		weekly["week_end"] = pd.to_datetime(weekly["week_end"])
		weekly = weekly.rename(columns={"week_end": "date"})
	elif "date" in weekly.columns:
		weekly["date"] = pd.to_datetime(weekly["date"])
	else:
		raise ValueError(f"No date column found in {CRSP_WEEKLY_FP}")

	weekly = weekly[weekly["date"].dt.year.between(2019, 2024)].copy()
	
	# Handle return column name variations
	if "weekly_ret" in weekly.columns:
		weekly_join = weekly[["date", "ticker", "weekly_ret"]].copy()
	else:
		raise ValueError(f"No 'weekly_ret' column found in {CRSP_WEEKLY_FP}")
	
	return weekly_join


def main() -> None:
	sentiment = load_sentiment()

	min_date = sentiment["date"].min()
	max_date = sentiment["date"].max()

	crsp_weekly = load_crsp_weekly()
	iwc_weekly = load_iwc_weekly(min_date, max_date)

	panel = sentiment.merge(crsp_weekly, on=["date", "ticker"], how="inner")
	panel = panel.merge(iwc_weekly, on="date", how="inner")
	panel["abnormal_ret"] = panel["weekly_ret"] - panel["benchmark_ret"]

	panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
	panel["lag_abnormal_ret"] = panel.groupby("ticker")["abnormal_ret"].shift(1)
	panel["lead_abnormal_ret"] = panel.groupby("ticker")["abnormal_ret"].shift(-1)

	reg1 = panel.dropna(subset=["abnormal_ret", "Sentiment"])
	reg2 = panel.dropna(subset=["lead_abnormal_ret", "Sentiment", "lag_abnormal_ret"])

	print("=" * 70)
	print(f"Panel observations: {len(panel):,}")
	print(f"Date range: {panel['date'].min().date()} to {panel['date'].max().date()}")
	print("=" * 70)

	print("\nREG 1 (t): abnormal_ret ~ Sentiment")
	print(f"N = {len(reg1):,}")
	m1 = smf.ols("abnormal_ret ~ Sentiment", data=reg1).fit(cov_type="HC3")
	print(m1.summary())

	print("\nREG 2 (t+1): lead_abnormal_ret ~ Sentiment + lag_abnormal_ret")
	print(f"N = {len(reg2):,}")
	m2 = smf.ols("lead_abnormal_ret ~ Sentiment + lag_abnormal_ret", data=reg2).fit(cov_type="HC3")
	print(m2.summary())

	panel.to_csv(OUT_DIR / "sentiment_weekly_panel_2019_2024.csv", index=False)
	coef = pd.DataFrame(
		{
			"model": ["reg1_t", "reg2_t_plus_1"],
			"beta_Sentiment": [m1.params.get("Sentiment", np.nan), m2.params.get("Sentiment", np.nan)],
			"pvalue_Sentiment": [m1.pvalues.get("Sentiment", np.nan), m2.pvalues.get("Sentiment", np.nan)],
			"N": [int(m1.nobs), int(m2.nobs)],
			"r2": [m1.rsquared, m2.rsquared],
		}
	)
	coef.to_csv(OUT_DIR / "sentiment_regression_results_2019_2024.csv", index=False)

	print("\nSaved:")
	print(OUT_DIR / "sentiment_weekly_panel_2019_2024.csv")
	print(OUT_DIR / "sentiment_regression_results_2019_2024.csv")


if __name__ == "__main__":
	main()
