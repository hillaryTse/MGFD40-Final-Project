"""
Fama-French 3-Factor Regression on Long-Short Portfolio Returns
---------------------------------------------------------------
Regresses weekly L/S returns for:
  (1) Mentions strategy  : long mentioned, short unmentioned
  (2) Sentiment strategy : long bearish, short bullish (contrarian)

Model: LS_ret_i,t = alpha + b1*(Mkt-RF)_t + b2*SMB_t + b3*HML_t + e_t

Alpha = abnormal return unexplained by FF3 factors.
HC3 robust standard errors.
"""

from pathlib import Path

import io
import zipfile

import requests
import pandas as pd
import statsmodels.formula.api as smf

OUT_DIR  = Path(__file__).resolve().parent
LS_DIR   = OUT_DIR.parent / "Long-Short"

MENTIONS_FP  = LS_DIR / "LS_mentions_weekly_2019_2023.csv"
SENTIMENT_FP = LS_DIR / "LS_sentiment_contrarian_weekly_2019_2023.csv"


def load_ff3_weekly() -> pd.DataFrame:
    """Download weekly FF3 factors directly from Ken French's data library."""
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_weekly_CSV.zip"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        raw = zf.read(zf.namelist()[0]).decode("utf-8")

    # File has a text header then a CSV block starting with the header row ",Mkt-RF,SMB,HML,RF"
    lines = raw.splitlines()
    data_start = next(i for i, l in enumerate(lines) if l.strip()[:8].isdigit())
    data_end   = next((i for i in range(data_start, len(lines))
                       if lines[i].strip() and not lines[i].strip()[:8].isdigit()),
                      len(lines))
    csv_block = "\n".join(lines[data_start:data_end])

    ff = pd.read_csv(io.StringIO(csv_block), header=None,
                     names=["date", "Mkt_RF", "SMB", "HML", "RF"])
    ff["date"] = pd.to_datetime(ff["date"].astype(str).str.strip(), format="%Y%m%d")
    for col in ["Mkt_RF", "SMB", "HML", "RF"]:
        ff[col] = pd.to_numeric(ff[col], errors="coerce") / 100
    return ff[["date", "Mkt_RF", "SMB", "HML", "RF"]].dropna()


def run_ff3(name: str, ls_weekly: pd.DataFrame, ff3: pd.DataFrame) -> pd.Series:
    """Merge L/S returns with FF3 factors and run regression. Returns coef summary."""
    df = ls_weekly[["date_return", "ls_ret"]].copy()
    df = df.rename(columns={"date_return": "date"})
    df["date"] = pd.to_datetime(df["date"])

    panel = df.merge(ff3, on="date", how="inner")
    # L/S portfolio is dollar-neutral so RF subtraction is optional,
    # but included for consistency with FF framework
    panel["ls_excess"] = panel["ls_ret"] - panel["RF"]

    n = len(panel)
    model = smf.ols("ls_excess ~ Mkt_RF + SMB + HML", data=panel).fit(cov_type="HC3")

    print("=" * 65)
    print(f"FF3 Regression — {name}  (N={n:,} weeks)")
    print("=" * 65)
    print(model.summary())

    return pd.Series({
        "strategy":   name,
        "N":          n,
        "alpha":      model.params["Intercept"],
        "alpha_tstat":model.tvalues["Intercept"],
        "alpha_pval": model.pvalues["Intercept"],
        "b_MktRF":    model.params["Mkt_RF"],
        "b_SMB":      model.params["SMB"],
        "b_HML":      model.params["HML"],
        "R2":         model.rsquared,
    })


def main() -> None:
    mentions  = pd.read_csv(MENTIONS_FP,  parse_dates=["date_return"])
    sentiment = pd.read_csv(SENTIMENT_FP, parse_dates=["date_return"])

    print("Downloading FF3 weekly factors from Ken French...")
    ff3 = load_ff3_weekly()
    ff3 = ff3[ff3["date"].between("2019-01-01", "2024-01-31")]
    print(f"FF3 loaded: {ff3['date'].min().date()} to {ff3['date'].max().date()}\n")

    results = []
    results.append(run_ff3("Mentions (Long mentioned / Short unmentioned)", mentions,  ff3))
    results.append(run_ff3("Sentiment Contrarian (Long bearish / Short bullish)", sentiment, ff3))

    summary = pd.DataFrame(results)

    print("\n" + "=" * 65)
    print("ALPHA SUMMARY")
    print("=" * 65)
    for _, row in summary.iterrows():
        sig = "***" if row["alpha_pval"] < 0.01 else ("**" if row["alpha_pval"] < 0.05 else ("*" if row["alpha_pval"] < 0.10 else ""))
        print(f"  {row['strategy']}")
        print(f"    Alpha = {row['alpha']:.4f}  t={row['alpha_tstat']:.3f}  p={row['alpha_pval']:.4f}  {sig}")
    print("=" * 65)

    summary.to_csv(OUT_DIR / "FF3_regression_results.csv", index=False)
    print(f"\nSaved: {OUT_DIR / 'FF3_regression_results.csv'}")


if __name__ == "__main__":
    main()
