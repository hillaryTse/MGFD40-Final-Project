"""
Simple Long-Short Strategy: Mentions (Quintile Sort)

Direction determined by t+1 beta from Mention_Regression.py results:
  beta > 0  ->  MOMENTUM:   LONG top quintile (Q5), SHORT no mention
  beta < 0  ->  CONTRARIAN: LONG no mention,    SHORT top quintile (Q5)

Reads beta from: Regression_Final/mention_regression_results_2019_2023.csv

- Week t: Rank mentioned stocks by mention count into quintiles
- Friday t+1 4pm: Form portfolio based on direction above
- Compute equally-weighted long/short returns for weeks t+1 to t+26
- Formation period: 2019-2023 (CRSP data extended to 2024 for forward returns)
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = Path(__file__).resolve().parent
CRSP_WEEKLY_FP = ROOT / "original_data_crsp" / "crsp_weekly_monday_friday_4pm_2019_2024.csv"
REGRESSION_RESULTS_FP = ROOT / "Regression_Final" / "mention_regression_results_2019_2023.csv"


def get_strategy_direction() -> tuple[str, float, float]:
    """Read t+1 beta from Mention_Regression.py output and determine strategy direction.

    A negative beta implies high-mention stocks underperform next week (contrarian).
    A positive beta implies they outperform (momentum).
    """
    reg = pd.read_csv(REGRESSION_RESULTS_FP)
    row = reg[reg["model"] == "reg2_t_plus_1"].iloc[0]
    beta   = float(row["beta_Mention"])
    pvalue = float(row["pvalue_Mention"])
    direction = "momentum" if beta > 0 else "contrarian"
    print(f"t+1 beta (Q5) = {beta:.4f}  (p={pvalue:.4f})  ->  {direction.upper()} strategy")
    if pvalue > 0.05:
        print("  WARNING: t+1 beta not significant at 5% - strategy direction not statistically justified")
    return direction, beta, pvalue


def load_weekly_mentions() -> pd.DataFrame:
    """Load and aggregate weekly Reddit mention counts per ticker (2019-2024)."""
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
        # Normalise date column name across yearly files
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
    # Keep Friday week-ends only (Mon-Fri week ending)
    mentions = mentions[mentions["date"].dt.dayofweek == 4].copy()
    # Sum mentions per ticker-week in case of duplicate rows
    mentions = mentions.groupby(["date", "ticker"], as_index=False)["mentions"].sum()
    return mentions


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
    direction, _, _ = get_strategy_direction()
    momentum = direction == "momentum"

    mentions = load_weekly_mentions()
    crsp     = load_crsp_weekly()

    all_dates = sorted(crsp["date"].unique())
    # Pre-index by date for O(1) lookup - avoids repeated full DataFrame scans in the loop
    crsp_by_date = {d: grp for d, grp in crsp.groupby("date")}
    ment_by_date = {d: grp[["ticker", "mentions"]].copy() for d, grp in mentions.groupby("date")}

    results = []
    max_forward = 26  # hold for up to 26 weeks (approx. 6 months)

    for i, date_t in enumerate(all_dates[:-1]):
        ment_t = ment_by_date.get(date_t, pd.DataFrame(columns=["ticker", "mentions"]))
        crsp_t = crsp_by_date.get(date_t, pd.DataFrame(columns=["ticker", "weekly_ret"]))

        # No-mention stocks = all CRSP stocks that week minus mentioned stocks
        all_stocks_t = set(crsp_t["ticker"].unique())
        no_mention_t = all_stocks_t - set(ment_t["ticker"])

        if len(ment_t) == 0 or len(no_mention_t) == 0:
            continue

        # Quintile sort on mention count; Q5 = most mentioned stocks
        ment_t = ment_t.copy()
        ment_t["quintile"] = pd.qcut(ment_t["mentions"], q=5, labels=False, duplicates="drop")
        top_quintile_t = set(ment_t[ment_t["quintile"] == ment_t["quintile"].max()]["ticker"])

        if len(top_quintile_t) == 0:
            continue

        # Assign long and short legs based on t+1 beta direction
        # momentum   (beta > 0): long Q5, short no-mention
        # contrarian (beta < 0): long no-mention, short Q5
        if momentum:
            long_set, short_set = top_quintile_t, no_mention_t
        else:
            long_set, short_set = no_mention_t, top_quintile_t

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
                    "n_long":         len(top_quintile_t),
                    "n_short":        len(no_mention_t),
                    "long_ret":       long_rets.mean(),
                    "short_ret":      short_rets.mean(),
                    "ls_ret":         ls_ret,
                })

    ls_panel = pd.DataFrame(results)
    # Restrict formation dates to 2019-2023; forward returns extend into 2024
    ls_panel = ls_panel[ls_panel["date_formation"].dt.year.between(2019, 2023)].copy()

    print("=" * 70)
    print(f"Long-Short Strategy: Mentions (Forward Returns t+1 to t+26)")
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

    # Full forward-horizon panel (t+1 to t+26) for decay analysis
    ls_panel.to_csv(OUT_DIR / "LS_mentions_forward_2019_2023.csv", index=False)
    print(f"\nSaved: {OUT_DIR / 'LS_mentions_forward_2019_2023.csv'}")

    # t+1 only file used by FF3_regression.py and plot_longshort.py
    ls_weekly = ls_panel[ls_panel["forward_weeks"] == 1].copy()
    ls_weekly.to_csv(OUT_DIR / "LS_mentions_weekly_2019_2023.csv", index=False)
    print(f"Saved: {OUT_DIR / 'LS_mentions_weekly_2019_2023.csv'}")


if __name__ == "__main__":
    main()
