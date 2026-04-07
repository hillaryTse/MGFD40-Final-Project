"""
Simple Long-Short Strategy: Sentiment (Quintile Sort)

Direction determined by t+1 beta from Sentiment_Regression.py results:
  beta > 0  ->  MOMENTUM:   LONG top bullish (Q5), SHORT top bearish (Q1)
  beta < 0  ->  CONTRARIAN: LONG top bearish (Q1), SHORT top bullish (Q5)

Reads beta from: Regression_Final/sentiment_regression_results_2019_2023.csv

- Week t: Rank stocks by mean weekly sentiment_value into quintiles
- Q5 = most positive sentiment (bullish), Q1 = most negative (bearish)
- Friday t+1 4pm: Form portfolio based on direction above
- Compute equally-weighted long/short returns for weeks t+1 to t+26
- Formation period: 2019-2023 (CRSP data extended to 2024 for forward returns)
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = Path(__file__).resolve().parent
SENTIMENT_FP = ROOT / "Finbert_Sentiment" / "finbert_sentiment_2019_2024.csv"
CRSP_WEEKLY_FP = ROOT / "original_data_crsp" / "crsp_weekly_monday_friday_4pm_2019_2024.csv"
REGRESSION_RESULTS_FP = ROOT / "Regression_Final" / "sentiment_regression_results_2019_2023.csv"


def get_strategy_direction() -> tuple[str, float, float]:
    """Read t+1 beta from Sentiment_Regression.py output and determine strategy direction.

    A negative beta implies top-sentiment stocks underperform next week (contrarian).
    A positive beta implies they outperform (momentum).
    """
    reg = pd.read_csv(REGRESSION_RESULTS_FP)
    row = reg[reg["model"] == "reg2_t_plus_1"].iloc[0]
    beta   = float(row["beta_Sentiment"])
    pvalue = float(row["pvalue_Sentiment"])
    direction = "momentum" if beta > 0 else "contrarian"
    print(f"t+1 beta (Q5) = {beta:.4f}  (p={pvalue:.4f})  ->  {direction.upper()} strategy")
    if pvalue > 0.05:
        print("  WARNING: t+1 beta not significant at 5% - strategy direction not statistically justified")
    return direction, beta, pvalue


def load_weekly_sentiment() -> pd.DataFrame:
    """Load FinBERT sentiment scores and aggregate to weekly mean per ticker.

    Daily Reddit posts are mapped to their Friday week-end date,
    then averaged across all posts for the same ticker that week.
    """
    sentiment = pd.read_csv(SENTIMENT_FP)
    sentiment["date"] = pd.to_datetime(sentiment["date"])

    # Map each daily date to the Friday of that Mon-Fri week
    sentiment["week_end"] = sentiment["date"] + pd.to_timedelta(
        (4 - sentiment["date"].dt.dayofweek) % 7, unit="D"
    )
    # Keep Friday week-ends only
    sentiment = sentiment[sentiment["week_end"].dt.dayofweek == 4].copy()

    # Average sentiment_value per ticker per week
    weekly_sent = sentiment.groupby(["week_end", "ticker"]).agg(
        sentiment_value=("sentiment_value", "mean")
    ).reset_index()

    weekly_sent = weekly_sent.rename(columns={"week_end": "date"})
    weekly_sent = weekly_sent.drop_duplicates(subset=["date", "ticker"])
    return weekly_sent


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
    # Keep Friday week-ends only
    weekly = weekly[weekly["date"].dt.dayofweek == 4].copy()

    if "weekly_ret" not in weekly.columns:
        raise ValueError(f"No 'weekly_ret' column found in {CRSP_WEEKLY_FP}")

    return weekly[["date", "ticker", "weekly_ret"]].copy()


def main() -> None:
    direction, beta, pvalue = get_strategy_direction()
    momentum = direction == "momentum"

    sentiment = load_weekly_sentiment()
    crsp      = load_crsp_weekly()

    all_dates = sorted(crsp["date"].unique())
    # Pre-index by date for O(1) lookup - avoids repeated full DataFrame scans in the loop
    crsp_by_date = {d: grp for d, grp in crsp.groupby("date")}
    sent_by_date = {d: grp[["ticker", "sentiment_value"]].copy() for d, grp in sentiment.groupby("date")}

    results = []
    max_forward = 26  # hold for up to 26 weeks (approx. 6 months)

    for i, date_t in enumerate(all_dates[:-1]):
        sent_t = sent_by_date.get(date_t, pd.DataFrame(columns=["ticker", "sentiment_value"]))
        crsp_t = crsp_by_date.get(date_t, pd.DataFrame(columns=["ticker", "weekly_ret"]))

        # Only use stocks that have both sentiment and price data this week
        stocks_with_both = set(sent_t["ticker"].unique()) & set(crsp_t["ticker"].unique())
        sent_t = sent_t[sent_t["ticker"].isin(stocks_with_both)].copy()

        # Need at least 5 stocks to form quintiles
        if len(sent_t) < 5:
            continue

        # Quintile sort on sentiment_value within this week
        sent_t["quintile"] = pd.qcut(sent_t["sentiment_value"], q=5, labels=False, duplicates="drop")
        max_q = sent_t["quintile"].max()
        min_q = sent_t["quintile"].min()

        top_bullish_t = set(sent_t[sent_t["quintile"] == max_q]["ticker"])  # Q5: most positive
        top_bearish_t = set(sent_t[sent_t["quintile"] == min_q]["ticker"])  # Q1: most negative

        if len(top_bullish_t) == 0 or len(top_bearish_t) == 0:
            continue

        # Assign long and short legs based on t+1 beta direction
        # momentum   (beta > 0): long Q5 (bullish), short Q1 (bearish)
        # contrarian (beta < 0): long Q1 (bearish), short Q5 (bullish)
        if momentum:
            long_set, short_set = top_bullish_t, top_bearish_t
        else:
            long_set, short_set = top_bearish_t, top_bullish_t

        # Compute equally-weighted portfolio returns for each forward week
        for offset in range(1, min(max_forward + 1, len(all_dates) - i)):
            date_forward = all_dates[i + offset]
            forward_rets = crsp_by_date.get(date_forward, pd.DataFrame(columns=["ticker", "weekly_ret"]))

            long_rets  = forward_rets[forward_rets["ticker"].isin(long_set)]["weekly_ret"]
            short_rets = forward_rets[forward_rets["ticker"].isin(short_set)]["weekly_ret"]

            if len(long_rets) > 0 and len(short_rets) > 0:
                ls_ret = long_rets.mean() - short_rets.mean()
                results.append({
                    "date_formation": date_t,
                    "date_return":    date_forward,
                    "forward_weeks":  offset,
                    "n_long":         len(long_set),
                    "n_short":        len(short_set),
                    "long_ret":       long_rets.mean(),
                    "short_ret":      short_rets.mean(),
                    "ls_ret":         ls_ret,
                })

    ls_panel = pd.DataFrame(results)
    # Restrict formation dates to 2019-2023; forward returns extend into 2024
    ls_panel = ls_panel[ls_panel["date_formation"].dt.year.between(2019, 2023)].copy()

    long_label  = "Top Bullish (Q5)" if momentum else "Top Bearish (Q1)"
    short_label = "Top Bearish (Q1)" if momentum else "Top Bullish (Q5)"

    print("=" * 70)
    print(f"Long-Short Strategy: Sentiment Quintile ({'Momentum' if momentum else 'Contrarian'}) (Forward Returns t+1 to t+26)")
    print(f"LONG: {long_label} | SHORT: {short_label}")
    print(f"Total observations: {len(ls_panel)}")
    print(f"Date range (formation): {ls_panel['date_formation'].min().date()} to {ls_panel['date_formation'].max().date()}")
    print(f"Forward weeks range: 1 to {ls_panel['forward_weeks'].max()}")
    print("=" * 70)

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

    # Full forward-horizon panel (t+1 to t+26) for decay analysis
    out_fp = OUT_DIR / f"LS_sentiment_{tag}_forward_2019_2023.csv"
    ls_panel.to_csv(out_fp, index=False)
    print(f"\nSaved: {out_fp}")

    # t+1 only file used by FF3_regression.py and plot_longshort.py
    ls_weekly = ls_panel[ls_panel["forward_weeks"] == 1].copy()
    ls_weekly.to_csv(OUT_DIR / f"LS_sentiment_{tag}_weekly_2019_2023.csv", index=False)
    print(f"Saved: {OUT_DIR / f'LS_sentiment_{tag}_weekly_2019_2023.csv'}")


if __name__ == "__main__":
    main()
