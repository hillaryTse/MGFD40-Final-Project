"""
Generate academic-style summary table for FF3 regressions.

Reads results from FF3_regression_results.csv (produced by FF3_regression.py).
Renders a publication-style table using matplotlib and saves to FF3F/plots/FF3_table.png.
Run FF3_regression.py first to ensure the CSV is up to date.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

OUT_DIR  = Path(__file__).resolve().parent
PLOT_DIR = OUT_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)


def stars(p):
    """Return significance stars based on p-value thresholds."""
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""


# Load regression results - run FF3_regression.py first if this file is missing
_res = pd.read_csv(OUT_DIR / "FF3_regression_results.csv")

# Build strategy column headers and data dicts from CSV rows
STRATEGIES = [s.replace(" (", "\n(") for s in _res["strategy"].tolist()]
DATA = [
    {
        "Alpha":  (row["alpha"],   row["alpha_tstat"],   row["alpha_pval"]),
        "Mkt-RF": (row["b_MktRF"], row["b_MktRF_tstat"], row["b_MktRF_pval"]),
        "SMB":    (row["b_SMB"],   row["b_SMB_tstat"],   row["b_SMB_pval"]),
        "HML":    (row["b_HML"],   row["b_HML_tstat"],   row["b_HML_pval"]),
        "R2": row["R2"], "N": int(row["N"]),
    }
    for _, row in _res.iterrows()
]
VARS = ["Alpha", "Mkt-RF", "SMB", "HML"]

# --- Layout constants ---
fig = plt.figure(figsize=(7, 5.5), facecolor="white")
ax  = fig.add_axes([0, 0, 1, 1])
ax.axis("off")

FONT    = "DejaVu Serif"
LEFT    = 0.18
COL_W   = 0.36
TOP     = 0.72   # y-position of first data row
ROW_H   = 0.088  # vertical spacing between rows
HDR_H   = 0.10   # height reserved for column headers

# x-centres for each strategy column
col_xs  = [LEFT + i * COL_W + COL_W / 2 for i in range(2)]
label_x = 0.14   # x-position for row labels (left-aligned)


def hline(y, lw=0.8, color="black"):
    """Draw a horizontal rule across the table."""
    ax.axhline(y, xmin=0.03, xmax=0.97, color=color, linewidth=lw)


ax.text(0.5, 0.97,
        "Table 1. Fama-French 3-Factor Regressions on Long-Short Portfolio Returns",
        ha="center", va="top", fontsize=10, fontweight="bold",
        fontfamily=FONT, transform=ax.transAxes)

# Model equation displayed below title, above the top rule
ax.text(0.5, 0.90,
        r"$LS\text{-}ret_t - RF_t = \alpha + \beta_1(Mkt\text{-}RF)_t + \beta_2 SMB_t + \beta_3 HML_t + \varepsilon_t$",
        ha="center", va="top", fontsize=8.5, color="#333333",
        transform=ax.transAxes)

hline(0.83, lw=1.2)

# Strategy column headers - one per L/S strategy
for x, strat in zip(col_xs, STRATEGIES):
    ax.text(x, 0.83 - HDR_H * 0.45, strat,
            ha="center", va="center", fontsize=8.5, fontweight="bold",
            fontfamily=FONT, transform=ax.transAxes,
            multialignment="center")

# Sub-rule separating headers from data rows
hline(TOP + ROW_H * 0.3, lw=0.6, color="#555555")

# Factor rows: coefficient on top, t-statistic below in smaller font
for r, var in enumerate(VARS):
    y_coef  = TOP - ROW_H * (r + 0.55)
    y_tstat = TOP - ROW_H * (r + 0.88)

    # Row label (Alpha, Mkt-RF, SMB, HML)
    ax.text(label_x, y_coef, var,
            ha="right", va="center", fontsize=9,
            fontfamily=FONT, transform=ax.transAxes)

    for x, d in zip(col_xs, DATA):
        coef, t, p = d[var]
        s = stars(p)
        # Coefficient with significance stars
        ax.text(x, y_coef,  f"{coef:.4f}{s}",
                ha="center", va="center", fontsize=9,
                fontfamily=FONT, transform=ax.transAxes)
        # t-statistic in parentheses
        ax.text(x, y_tstat, f"({t:.3f})",
                ha="center", va="center", fontsize=8, color="#444444",
                fontfamily=FONT, transform=ax.transAxes)

# Thin divider before R2 and N block
div_y = TOP - ROW_H * (len(VARS) + 0.1)
hline(div_y, lw=0.5, color="#888888")

# R2 and observation count rows
stats_rows = [("R²", "R2", ".3f"), ("Observations", "N", "d")]
for s, (label, key, fmt) in enumerate(stats_rows):
    y = div_y - ROW_H * (s + 0.6)
    ax.text(label_x, y, label,
            ha="right", va="center", fontsize=9,
            fontfamily=FONT, transform=ax.transAxes)
    for x, d in zip(col_xs, DATA):
        ax.text(x, y, format(d[key], fmt),
                ha="center", va="center", fontsize=9,
                fontfamily=FONT, transform=ax.transAxes)

bottom_y = div_y - ROW_H * (len(stats_rows) + 0.1)
hline(bottom_y, lw=1.2)

# Footer: standard errors and significance thresholds
ax.text(0.5, bottom_y - 0.04,
        "t-statistics in parentheses. HC3 heteroscedasticity-robust standard errors.\n"
        "* p < 0.10   ** p < 0.05   *** p < 0.01",
        ha="center", va="top", fontsize=7.5, color="#555555",
        style="italic", fontfamily=FONT, transform=ax.transAxes,
        multialignment="center")

out_fp = PLOT_DIR / "FF3_table.png"
plt.savefig(out_fp, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved: {out_fp}")
plt.show()
