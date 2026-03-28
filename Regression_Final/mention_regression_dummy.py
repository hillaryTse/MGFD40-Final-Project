"""
Weekly mention regressions (2019-2024)

Reg 1 (t):   abnormal_ret_i,t   = a + b1*Mention_i,t + e
Reg 2 (t+1): abnormal_ret_i,t+1 = a + b1*Mention_i,t + b2*lag_abnormal_ret_i,t-1 + e
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import yfinance as yf


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent
CRSP_WEEKLY_FP = ROOT / "original_data_crsp" / "crsp_weekly_monday_friday_4pm_2019_2024.parquet"


def load_weekly_mentions() -> pd.DataFrame:
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

		keep = df[["date", "ticker", "mentions"]].copy()
		keep["mentions"] = pd.to_numeric(keep["mentions"], errors="coerce").fillna(0)
		parts.append(keep)

	mentions = pd.concat(parts, ignore_index=True)
	mentions = (
		mentions.groupby(["date", "ticker"], as_index=False)["mentions"]
		.sum()
		.rename(columns={"mentions": "Mention"})
	)
	return mentions


def load_weekly_no_mentions() -> pd.DataFrame:
	files = [
		ROOT / "2019_reddit_mentions" / "no_mentions_2019.csv",
		ROOT / "2020_reddit_mentions" / "no_mentions_2020.csv",
		ROOT / "2021_reddit_mentions" / "no_mentions_2021.csv",
		ROOT / "2022_reddit_mentions" / "no_mentions_2022.csv",
		ROOT / "2023_reddit_mentions" / "no_mentions_2023.csv",
		ROOT / "2024_Reddit_mentions" / "no_mentions_2024.csv",
	]

	parts = []
	for fp in files:
		if not fp.exists():
			raise FileNotFoundError(f"Missing no-mentions file: {fp}")
		df = pd.read_csv(fp, parse_dates=["date"], usecols=["date", "ticker"])
		df["Mention"] = 0.0
		parts.append(df)

	out = pd.concat(parts, ignore_index=True)
	return out


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
	weekly = pd.read_parquet(CRSP_WEEKLY_FP, columns=["week_end", "ticker", "weekly_ret"])
	weekly["week_end"] = pd.to_datetime(weekly["week_end"])
	weekly = weekly[weekly["week_end"].dt.year.between(2019, 2024)].copy()
	weekly_join = weekly.rename(columns={"week_end": "date"})
	return weekly_join


def main() -> None:
	mentions = load_weekly_mentions()
	no_mentions = load_weekly_no_mentions()

	signal_panel = pd.concat(
		[mentions[["date", "ticker", "Mention"]], no_mentions[["date", "ticker", "Mention"]]],
		ignore_index=True,
	)
	signal_panel = (
		signal_panel.groupby(["date", "ticker"], as_index=False)["Mention"]
		.max()
		.sort_values(["date", "ticker"])
	)

	min_date = signal_panel["date"].min()
	max_date = signal_panel["date"].max()

	crsp_weekly = load_crsp_weekly()
	iwc_weekly = load_iwc_weekly(min_date, max_date)

	panel = signal_panel.merge(crsp_weekly, on=["date", "ticker"], how="inner")
	panel = panel.merge(iwc_weekly, on="date", how="inner")
	panel["abnormal_ret"] = panel["weekly_ret"] - panel["benchmark_ret"]

	panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
	panel["lag_abnormal_ret"] = panel.groupby("ticker")["abnormal_ret"].shift(1)
	panel["lead_abnormal_ret"] = panel.groupby("ticker")["abnormal_ret"].shift(-1)

	reg1 = panel.dropna(subset=["abnormal_ret", "Mention"])
	reg2 = panel.dropna(subset=["lead_abnormal_ret", "Mention", "lag_abnormal_ret"])

	print("=" * 70)
	print(f"Panel observations: {len(panel):,}")
	print(f"Date range: {panel['date'].min().date()} to {panel['date'].max().date()}")
	print("=" * 70)

	print("\nREG 1 (t): abnormal_ret ~ Mention")
	print(f"N = {len(reg1):,}")
	m1 = smf.ols("abnormal_ret ~ Mention", data=reg1).fit(cov_type="HC3")
	print(m1.summary())

	print("\nREG 2 (t+1): lead_abnormal_ret ~ Mention + lag_abnormal_ret")
	print(f"N = {len(reg2):,}")
	m2 = smf.ols("lead_abnormal_ret ~ Mention + lag_abnormal_ret", data=reg2).fit(cov_type="HC3")
	print(m2.summary())

	panel.to_csv(OUT_DIR / "mention_weekly_panel_2019_2024.csv", index=False)
	coef = pd.DataFrame(
		{
			"model": ["reg1_t", "reg2_t_plus_1"],
			"beta_Mention": [m1.params.get("Mention", np.nan), m2.params.get("Mention", np.nan)],
			"pvalue_Mention": [m1.pvalues.get("Mention", np.nan), m2.pvalues.get("Mention", np.nan)],
			"N": [int(m1.nobs), int(m2.nobs)],
			"r2": [m1.rsquared, m2.rsquared],
		}
	)
	coef.to_csv(OUT_DIR / "mention_regression_results_2019_2024.csv", index=False)

	print("\nSaved:")
	print(OUT_DIR / "mention_weekly_panel_2019_2024.csv")
	print(OUT_DIR / "mention_regression_dummy_results_2019_2024.csv")


if __name__ == "__main__":
	main()

