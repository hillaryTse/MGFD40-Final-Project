"""
Long-Short Portfolio Plots
--------------------------
Mentions   : (1) Cumulative long/short/L-S returns over time
             (2) Forward-horizon decay — avg L/S spread at weeks t+1 to t+26
                 (plotted until mean L/S first crosses back to 0)

Sentiment  : (1) Cumulative long/short/L-S returns over time
             (forward horizon omitted — t+1 beta not significant)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

PLOT_DIR = Path(__file__).resolve().parent
OUT_DIR  = PLOT_DIR.parent
PLOT_DIR.mkdir(exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
mentions_weekly  = pd.read_csv(OUT_DIR / "LS_mentions_weekly_2019_2023.csv",   parse_dates=["date_formation", "date_return"])
mentions_forward = pd.read_csv(OUT_DIR / "LS_mentions_forward_2019_2023.csv",  parse_dates=["date_formation", "date_return"])
senti_weekly     = pd.read_csv(OUT_DIR / "LS_sentiment_contrarian_weekly_2019_2023.csv", parse_dates=["date_formation", "date_return"])

# ── Helper: cumulative return series from a weekly file ───────────────────────
def cum_rets(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with cumulative long, short, ls columns indexed by date_return."""
    d = df.sort_values("date_return").copy()
    d["cum_long"]  = (1 + d["long_ret"]).cumprod() - 1
    d["cum_short"] = (1 + d["short_ret"]).cumprod() - 1
    d["cum_ls"]    = (1 + d["ls_ret"]).cumprod() - 1
    return d


def fill_ls(ax, x, y):
    """Green/red fill under L/S cumulative curve."""
    ax.fill_between(x, y, 0, where=(y >= 0), alpha=0.20, color="green")
    ax.fill_between(x, y, 0, where=(y  < 0), alpha=0.20, color="red")


def style_ax(ax, title, ylabel="Cumulative Return"):
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=1))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — MENTIONS  (3 panels)
# ══════════════════════════════════════════════════════════════════════════════
mc = cum_rets(mentions_weekly)

# Forward-horizon: mean L/S by offset week
fwd = (
    mentions_forward
    .groupby("forward_weeks")["ls_ret"]
    .agg(mean="mean", sem=lambda x: x.std() / np.sqrt(len(x)))
    .reset_index()
)

# Trim to first reversal (mean crosses 0 from positive, or all if never crosses)
first_cross = fwd.index[(fwd["mean"].shift(1) > 0) & (fwd["mean"] <= 0)]
cutoff = int(fwd.loc[first_cross[0], "forward_weeks"]) if len(first_cross) else fwd["forward_weeks"].max()
fwd_trim = fwd[fwd["forward_weeks"] <= cutoff].copy()

fig1, axes = plt.subplots(2, 1, figsize=(13, 10))
fig1.suptitle("Long-Short Strategy: Reddit Mentions (2019–2023)", fontsize=13, fontweight="bold")

# Panel 1 — long vs short cumulative
axes[0].plot(mc["date_return"], mc["cum_long"],  label="Long (mentioned)",    color="steelblue",  linewidth=1.5)
axes[0].plot(mc["date_return"], mc["cum_short"], label="Short (no mention)",  color="darkorange", linewidth=1.5)
style_ax(axes[0], "Cumulative Returns — Long Leg vs Short Leg")

# Panel 2 — L/S spread cumulative
axes[1].plot(mc["date_return"], mc["cum_ls"], color="navy", linewidth=2, label="L/S Spread")
fill_ls(axes[1], mc["date_return"], mc["cum_ls"].values)
style_ax(axes[1], "Cumulative Returns — Long-Short Portfolio")
axes[1].set_xlabel("Date")

plt.tight_layout()
fig1.savefig(PLOT_DIR / "plot_mentions_longshort.png", dpi=150)
print("Saved: plots/plot_mentions_longshort.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — SENTIMENT CONTRARIAN  (2 panels)
# ══════════════════════════════════════════════════════════════════════════════
sc = cum_rets(senti_weekly)

fig2, axes2 = plt.subplots(2, 1, figsize=(13, 8))
fig2.suptitle("Long-Short Strategy: Sentiment Contrarian — Long Bearish / Short Bullish (2019–2023)",
              fontsize=13, fontweight="bold")

# Panel 1 — long vs short cumulative
axes2[0].plot(sc["date_return"], sc["cum_long"],  label="Long (bearish senti)",  color="steelblue",  linewidth=1.5)
axes2[0].plot(sc["date_return"], sc["cum_short"], label="Short (bullish senti)", color="darkorange",  linewidth=1.5)
style_ax(axes2[0], "Cumulative Returns — Long Leg vs Short Leg")

# Panel 2 — L/S spread cumulative
axes2[1].plot(sc["date_return"], sc["cum_ls"], color="navy", linewidth=2, label="L/S Spread")
fill_ls(axes2[1], sc["date_return"], sc["cum_ls"].values)
style_ax(axes2[1], "Cumulative Returns — Long-Short Portfolio")
axes2[1].set_xlabel("Date")

plt.tight_layout()
fig2.savefig(PLOT_DIR / "plot_sentiment_longshort.png", dpi=150)
print("Saved: plots/plot_sentiment_longshort.png")

plt.show()
