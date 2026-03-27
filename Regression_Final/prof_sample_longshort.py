"""
Long-Short Strategy Based on Reddit Attention (Weekly Rebalancing)
------------------------------------------------------------------
Universe  : 4 stocks (A, B, C, D)
Signal    : Each week, 2 stocks have HIGH reddit attention (short them),
            2 stocks have LOW reddit attention (buy them).
Horizon   : 24 weekly rebalancing periods.

Intuition : Retail social-media attention → short-term overvaluation →
            we fade high-attention stocks and go long neglected stocks.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

np.random.seed(42)

# ── Parameters ────────────────────────────────────────────────────────────────
STOCKS = ["A", "B", "C", "D"]
N_WEEKS = 24
N_STOCKS = len(STOCKS)

# Weekly return assumptions
# High-attention stocks: slightly negative expected return (reversal effect)
# Low-attention  stocks: slightly positive expected return
MU_HIGH = -0.003  # mean weekly return for high-attention stocks
MU_LOW = 0.003  # mean weekly return for low-attention  stocks
SIGMA = 0.03  # weekly volatility (same for all)

# ── Simulate weekly returns ───────────────────────────────────────────────────
# Each week, randomly assign 2 stocks to HIGH attention, 2 to LOW attention
attention_matrix = np.zeros((N_WEEKS, N_STOCKS), dtype=int)  # 1 = high, 0 = low
returns_matrix = np.zeros((N_WEEKS, N_STOCKS))

for t in range(N_WEEKS):
    high_idx = np.random.choice(N_STOCKS, size=2, replace=False)
    attention_matrix[t, high_idx] = 1  # mark high-attention

    for s in range(N_STOCKS):
        mu = MU_HIGH if attention_matrix[t, s] == 1 else MU_LOW
        returns_matrix[t, s] = np.random.normal(mu, SIGMA)

# ── Build DataFrames ──────────────────────────────────────────────────────────
weeks = [f"W{t + 1:02d}" for t in range(N_WEEKS)]

df_ret = pd.DataFrame(returns_matrix, index=weeks, columns=STOCKS)
df_att = pd.DataFrame(attention_matrix, index=weeks, columns=STOCKS)

# ── Construct the Long-Short Portfolio ───────────────────────────────────────
# Weights: +0.5 on each low-attention stock, -0.5 on each high-attention stock
# (dollar-neutral: sum of weights = 0)

weights_matrix = np.where(df_att == 0, 0.5, -0.5)  # shape (N_WEEKS, N_STOCKS)
df_weights = pd.DataFrame(weights_matrix, index=weeks, columns=STOCKS)

# Portfolio return each week = sum(weight_i * return_i)
portfolio_returns = (df_weights.values * df_ret.values).sum(axis=1)
df_port = pd.Series(portfolio_returns, index=weeks, name="LongShort")

# Cumulative returns
cum_port = (1 + df_port).cumprod() - 1
cum_stocks = (1 + df_ret).cumprod() - 1

# ── Performance Summary ───────────────────────────────────────────────────────
total_return = cum_port.iloc[-1]
ann_vol = df_port.std() * np.sqrt(52)
sharpe = (df_port.mean() / df_port.std()) * np.sqrt(52)

print("=" * 50)
print("  Long-Short Strategy — Performance Summary")
print("=" * 50)
print(f"  Total Return  : {total_return:+.2%}")
print(f"  Ann. Volatility: {ann_vol:.2%}")
print(f"  Ann. Sharpe   : {sharpe:.2f}")
print("=" * 50)

print("\nWeekly weights (+ = long, - = short):")
print(df_weights.to_string())

print("\nWeekly returns:")
print(df_ret.round(4).to_string())

print("\nPortfolio return per week:")
print(df_port.round(4).to_string())

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Top: cumulative returns of individual stocks
for col in STOCKS:
    axes[0].plot(cum_stocks[col], label=f"Stock {col}", alpha=0.7)
axes[0].set_title("Cumulative Returns — Individual Stocks")
axes[0].set_ylabel("Cumulative Return")
axes[0].legend(loc="upper left")
axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
axes[0].grid(True, alpha=0.3)

# Bottom: long-short portfolio cumulative return
axes[1].plot(cum_port, color="navy", linewidth=2, label="Long-Short Portfolio")
axes[1].fill_between(
    range(N_WEEKS),
    cum_port.values,
    0,
    where=(cum_port.values >= 0),
    alpha=0.2,
    color="green",
)
axes[1].fill_between(
    range(N_WEEKS),
    cum_port.values,
    0,
    where=(cum_port.values < 0),
    alpha=0.2,
    color="red",
)
axes[1].set_title("Cumulative Returns — Long-Short Portfolio")
axes[1].set_ylabel("Cumulative Return")
axes[1].set_xlabel("Week")
axes[1].set_xticks(range(N_WEEKS))
axes[1].set_xticklabels(weeks, rotation=45, fontsize=7)
axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("longshort_strategy.png", dpi=150)
plt.show()
print("\nChart saved to longshort_strategy.png")
