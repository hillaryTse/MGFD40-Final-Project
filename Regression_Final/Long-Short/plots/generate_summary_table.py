"""
Summary Statistics Tables
--------------------------
Table 2: Reddit Mentions Coverage by quintile (Q1-Q5) and no-mention group
Table 3: FinBERT Sentiment Coverage by quintile (Q1-Q5)
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT     = Path(__file__).resolve().parent.parent.parent.parent
PLOT_DIR = Path(__file__).resolve().parent
PLOT_DIR.mkdir(exist_ok=True)

YEAR_TO_FOLDER = {
    2019: "2019_reddit_mentions",
    2020: "2020_reddit_mentions",
    2021: "2021_reddit_mentions",
    2022: "2022_reddit_mentions",
    2023: "2023_reddit_mentions",
}

# ── Load data ─────────────────────────────────────────────────────────────────
mention_parts, no_mention_parts = [], []
for year, folder in YEAR_TO_FOLDER.items():
    m = pd.read_csv(ROOT / folder / f"weekly_mentions_{year}.csv",
                    usecols=["week_end", "ticker", "mentions"])
    m["date"] = pd.to_datetime(m["week_end"])
    mention_parts.append(m[["date", "ticker", "mentions"]])

    n = pd.read_csv(ROOT / folder / f"no_mentions_{year}.csv",
                    usecols=["date", "ticker"], parse_dates=["date"])
    no_mention_parts.append(n)

mentions   = pd.concat(mention_parts,   ignore_index=True)
no_mention = pd.concat(no_mention_parts, ignore_index=True)
mentions   = mentions.groupby(["date", "ticker"], as_index=False)["mentions"].sum()

# ── Quintile sort per week ────────────────────────────────────────────────────
def assign_quintile(grp):
    if len(grp) < 5:
        grp = grp.copy()
        grp["quintile"] = 1
        return grp
    grp = grp.copy()
    grp["quintile"] = pd.qcut(grp["mentions"].rank(method="first"), q=5, labels=False) + 1
    return grp

mentions = mentions.groupby("date", group_keys=False).apply(assign_quintile)

# ── Build summary rows ────────────────────────────────────────────────────────
total_mentions   = mentions["mentions"].sum()
total_stockweeks = len(mentions) + len(no_mention)

QUINTILE_LABELS = {1: "1 (low)", 2: "2", 3: "3", 4: "4", 5: "5 (high)"}
rows = []
for q in [1, 2, 3, 4, 5]:
    sub = mentions[mentions["quintile"] == q]
    rows.append([
        QUINTILE_LABELS[q],
        f"{len(sub):,}",
        f"{int(sub['mentions'].sum()):,}",
        f"{sub['mentions'].sum() / total_mentions * 100:.0f}",
        f"{sub['mentions'].mean():.1f}",
    ])

rows.append(["No mention", f"{len(no_mention):,}", "0", "0", "0.0"])
rows.append([
    "Total",
    f"{total_stockweeks:,}",
    f"{int(total_mentions):,}",
    "100",
    f"{total_mentions / len(mentions):.1f}",
])

COLS = ["Reddit quintile", "Stock-week obs.", "Total mentions", "% of mentions", "Avg. mentions"]

# ── Academic table plot ───────────────────────────────────────────────────────
FONT    = "DejaVu Serif"
N_ROWS  = len(rows)
N_COLS  = len(COLS)

COL_XS  = [0.22, 0.38, 0.53, 0.68, 0.83]
LABEL_X = 0.21
TOP     = 0.67
ROW_H   = 0.082

fig = plt.figure(figsize=(9, 5), facecolor="white")
ax  = fig.add_axes([0, 0, 1, 1])
ax.axis("off")

def hline(y, lw=0.8, color="black"):
    ax.axhline(y, xmin=0.03, xmax=0.97, color=color, linewidth=lw)

# Title
ax.text(0.5, 0.97,
        "Table 2. Summary Statistics on Reddit Mentions Coverage",
        ha="center", va="top", fontsize=10.5, fontweight="bold",
        fontfamily=FONT, transform=ax.transAxes)

# Caption
ax.text(0.5, 0.90,
        "This table reports summary statistics on Reddit mention coverage by quintile.\n"
        "Each week, stocks with at least one mention are ranked into quintiles (Q1–Q5) by total mention count.\n"
        "Sample period: January 2019 to December 2023.",
        ha="center", va="top", fontsize=8, color="#222222",
        fontfamily=FONT, transform=ax.transAxes, multialignment="center", linespacing=1.6)

# Top rule
hline(0.80, lw=1.2)

# Column headers
for x, col in zip(COL_XS, COLS):
    ax.text(x, 0.80 - 0.05, col,
            ha="center", va="center", fontsize=8.5, fontweight="bold",
            fontfamily=FONT, transform=ax.transAxes, multialignment="center")

# Sub-rule under headers
hline(TOP + ROW_H * 0.35, lw=0.6, color="#555555")

# Data rows (Q1–Q5 + No mention)
for r, row in enumerate(rows[:-1]):
    y = TOP - ROW_H * (r + 0.55)
    bold = r == 4  # Q5 bold (the long leg)
    for x, val in zip(COL_XS, row):
        ax.text(x, y, val,
                ha="center", va="center", fontsize=9,
                fontweight="bold" if bold else "normal",
                fontfamily=FONT, transform=ax.transAxes)

# Thin divider before Total
div_y = TOP - ROW_H * (N_ROWS - 1 - 0.1)
hline(div_y, lw=0.5, color="#888888")

# Total row
for x, val in zip(COL_XS, rows[-1]):
    ax.text(x, div_y - ROW_H * 0.6, val,
            ha="center", va="center", fontsize=9, fontweight="bold",
            fontfamily=FONT, transform=ax.transAxes)

# Bottom rule
bottom_y = div_y - ROW_H * 1.15
hline(bottom_y, lw=1.2)

# Footer note
ax.text(0.5, bottom_y - 0.035,
        "Quintiles assigned within each week. Q5 (high) forms the long leg of the L/S portfolio.\n"
        "No mention = stocks in CRSP universe with zero Reddit mentions that week (short leg).",
        ha="center", va="top", fontsize=7.5, color="#555555", style="italic",
        fontfamily=FONT, transform=ax.transAxes, multialignment="center")

out_fp = PLOT_DIR / "summary_mentions_table.png"
plt.savefig(out_fp, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved: {out_fp}")


# ══════════════════════════════════════════════════════════════════════════════
# TABLE 3 — SENTIMENT SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
SENTIMENT_FP = ROOT / "Finbert_Sentiment" / "finbert_sentiment_2019_2024.csv"

senti = pd.read_csv(SENTIMENT_FP, usecols=["date", "ticker", "sentiment_value", "top_label"])
senti["date"] = pd.to_datetime(senti["date"])
senti["week_end"] = senti["date"] + pd.to_timedelta((4 - senti["date"].dt.dayofweek) % 7, unit="D")
senti = senti[senti["week_end"].dt.dayofweek == 4].copy()

weekly_senti = senti.groupby(["week_end", "ticker"]).agg(
    sentiment_value=("sentiment_value", "mean"),
    pct_positive=("top_label", lambda x: (x == "positive").mean() * 100),
).reset_index().rename(columns={"week_end": "date"})
weekly_senti = weekly_senti.drop_duplicates(subset=["date", "ticker"])


def assign_senti_quintile(grp):
    if len(grp) < 5:
        grp = grp.copy()
        grp["quintile"] = 1
        return grp
    grp = grp.copy()
    grp["quintile"] = pd.qcut(grp["sentiment_value"].rank(method="first"), q=5, labels=False) + 1
    return grp

weekly_senti = weekly_senti.groupby("date", group_keys=False).apply(assign_senti_quintile)

QUINTILE_LABELS = {1: "1 (most bearish)", 2: "2", 3: "3", 4: "4", 5: "5 (most bullish)"}
SCOLS = ["Sentiment quintile", "Stock-week obs.", "Avg. sentiment", "% positive articles"]
srows = []
for q in [1, 2, 3, 4, 5]:
    sub = weekly_senti[weekly_senti["quintile"] == q]
    srows.append([
        QUINTILE_LABELS[q],
        f"{len(sub):,}",
        f"{sub['sentiment_value'].mean():.4f}",
        f"{sub['pct_positive'].mean():.1f}",
    ])
srows.append([
    "Total",
    f"{len(weekly_senti):,}",
    f"{weekly_senti['sentiment_value'].mean():.4f}",
    f"{weekly_senti['pct_positive'].mean():.1f}",
])

SCOL_XS = [0.24, 0.44, 0.62, 0.80]
TOP_S   = 0.67
ROW_H_S = 0.082

fig3 = plt.figure(figsize=(9, 5), facecolor="white")
ax3  = fig3.add_axes([0, 0, 1, 1])
ax3.axis("off")

def hline3(y, lw=0.8, color="black"):
    ax3.axhline(y, xmin=0.03, xmax=0.97, color=color, linewidth=lw)

ax3.text(0.5, 0.97,
         "Table 3. Summary Statistics on FinBERT Sentiment Coverage",
         ha="center", va="top", fontsize=10.5, fontweight="bold",
         fontfamily=FONT, transform=ax3.transAxes)

ax3.text(0.5, 0.90,
         "Each week, stocks with sentiment data are ranked into quintiles (Q1–Q5) by mean sentiment_value.\n"
         "Q1 = most bearish, Q5 = most bullish. % positive = share of articles labelled positive by FinBERT.\n"
         "Sample period: January 2019 to December 2023.",
         ha="center", va="top", fontsize=8, color="#222222",
         fontfamily=FONT, transform=ax3.transAxes, multialignment="center", linespacing=1.6)

hline3(0.80, lw=1.2)

for x, col in zip(SCOL_XS, SCOLS):
    ax3.text(x, 0.80 - 0.05, col,
             ha="center", va="center", fontsize=8.5, fontweight="bold",
             fontfamily=FONT, transform=ax3.transAxes, multialignment="center")

hline3(TOP_S + ROW_H_S * 0.35, lw=0.6, color="#555555")

for r, row in enumerate(srows[:-1]):
    y = TOP_S - ROW_H_S * (r + 0.55)
    bold = r == 4  # Q5 bold (long leg in momentum, short in contrarian)
    for x, val in zip(SCOL_XS, row):
        ax3.text(x, y, val,
                 ha="center", va="center", fontsize=9,
                 fontweight="bold" if bold else "normal",
                 fontfamily=FONT, transform=ax3.transAxes)

div_y3 = TOP_S - ROW_H_S * (len(srows) - 1 - 0.1)
hline3(div_y3, lw=0.5, color="#888888")

for x, val in zip(SCOL_XS, srows[-1]):
    ax3.text(x, div_y3 - ROW_H_S * 0.6, val,
             ha="center", va="center", fontsize=9, fontweight="bold",
             fontfamily=FONT, transform=ax3.transAxes)

bottom_y3 = div_y3 - ROW_H_S * 1.15
hline3(bottom_y3, lw=1.2)

ax3.text(0.5, bottom_y3 - 0.035,
         "Quintiles assigned within each week using rank-based sort to handle ties.\n"
         "Q1 forms the long leg (contrarian strategy); Q5 forms the short leg.",
         ha="center", va="top", fontsize=7.5, color="#555555", style="italic",
         fontfamily=FONT, transform=ax3.transAxes, multialignment="center")

out_fp3 = PLOT_DIR / "summary_sentiment_table.png"
fig3.savefig(out_fp3, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved: {out_fp3}")

plt.show()
