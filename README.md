# MGFD40 💸

## A Penny for Your Thoughts: Reddit Investor Attention, Sentiment Contagion, and Return Predictability in Penny Stocks

> 🪙 Penny stocks fly under the radar of traditional media so retail investors turn to Reddit.

This project explores whether Reddit ticker mentions and sentiment can predict penny stock returns. We run predictive regressions to determine whether high-attention or high-sentiment stocks outperform the following week, use the results to form a long-short trading strategy 📈📉, and test whether it generates abnormal returns (alpha) using Fama-French 3-factor regressions.

---

## 📁 Project Structure

```
[year]_reddit_mentions/
    weekly_mentions_[year].csv      # Weekly mention counts per ticker
    all_mentions_[year].csv         # All raw mention records
    no_mentions_[year].csv          # Stocks with no mentions that week

All_Year_Reddit_Data/
    Reddit_Data_All_Years/
        reddit_fetch_2007_2024.py           # Scrapes raw Reddit posts mentioning penny stock tickers
        reddit-fetch-fixes.py               # Fixes/patches applied to the fetch script
        reddit_data_2013_2024.csv           # Raw Reddit post data (2013–2024)
        reddit_data_2013_2024.parquet       # Same data in parquet format
    no_mentions_date.py                     # Identifies stocks with no Reddit mentions by date

Finbert_Sentiment/
    calc_finbert_sentiment_2019_2024.py   # Runs FinBERT on Reddit posts
    finbert_sentiment_2019_2024.csv       # Output sentiment scores per post

original_data_crsp/
    crsp_weekly_monday_friday_4pm_2019_2024.csv   # Weekly stock returns (Mon–Fri)
    crsp_daily_*.csv / .parquet                   # Raw daily CRSP data
    compustat_quarterly_*.csv                     # Compustat fundamentals
    crsp_compu_link_table_*.csv                   # CRSP–Compustat link table

Regression_Final/
    Mention_Regression.py                         # Predictive regression: mentions → returns
    Sentiment_Regression.py                       # Predictive regression: sentiment → returns
    mention_regression_results_2019_2023.csv      # Regression output (mentions)
    sentiment_regression_results_2019_2023.csv    # Regression output (sentiment)
    mention_weekly_panel_2019_2023.csv            # Weekly panel used for mention regression
    sentiment_weekly_panel_2019_2023.csv          # Weekly panel used for sentiment regression

    Long-Short/
        LS_mentions.py                            # Builds long-short portfolio from mention quintiles
        LS_sentiment.py                           # Builds long-short portfolio from sentiment quintiles
        LS_mentions_weekly_2019_2023.csv          # t+1 weekly LS returns (mentions)
        LS_sentiment_*_weekly_2019_2023.csv       # t+1 weekly LS returns (sentiment)
        LS_*_forward_2019_2023.csv                # LS returns for t+1 to t+26 forward weeks

    FF3F/
        FF3_regression.py                         # Regresses LS returns on Fama-French 3 factors
        FF3_regression_results.csv                # Alpha, betas, t-stats
        plot_ff3_table.py                         # Plots FF3 results table
        plots/
            FF3_table.png                         # FF3 regression results table

    Long-Short/plots/
        plot_longshort.py                         # Generates long-short performance plots
        generate_summary_table.py                 # Generates summary stats tables
        plot_mentions_longshort.png               # LS portfolio performance chart (mentions)
        plot_sentiment_longshort.png              # LS portfolio performance chart (sentiment)
        summary_mentions_table.png                # Summary stats table (mentions)
        summary_sentiment_table.png               # Summary stats table (sentiment)
```

---

## In VS Code: Do Ctrl+Shift+V on this README.md to view preview

## Installation

Choose **Option A** (GitHub Desktop — recommended for beginners) or **Option B** (command line) for Installation and Contributing.

---

### Option A: GitHub Desktop (Recommended)

#### A1. Install GitHub Desktop

Download and install from [desktop.github.com](https://desktop.github.com/).

#### A2. Sign in to GitHub

Open GitHub Desktop → **File > Options > Accounts** → sign in with your GitHub account.

#### A3. Clone the repository

1. In GitHub Desktop, click **File > Clone repository**
2. Select the **URL** tab and paste: `https://github.com/hillaryTse/MGFD40-Final-Project.git`
3. Choose a local path and click **Clone**

#### A4. Install Python

Download **Python 3.10 or newer** from [python.org/downloads](https://www.python.org/downloads/).
On Windows, check **"Add Python to PATH"** during installation.

#### A5. Install required libraries

Open a terminal (or the terminal inside VS Code) in the project folder and run:

```bash
pip install pandas pyarrow praw requests
```

---

### Option B: Command Line

#### B1. Install Git

Download from [git-scm.com](https://git-scm.com/downloads) and install.

#### B2. Clone the repository

```bash
git clone https://github.com/hillaryTse/MGFD40-Final-Project.git
cd MGFD40
```

#### B3. Set main as upstream

This ensures `git pull` pulls from the main branch on GitHub by default:

```bash
git branch --set-upstream-to=origin/main main
```

#### B4. Install Python

Download **Python 3.10 or newer** from [python.org/downloads](https://www.python.org/downloads/).
On Windows, check **"Add Python to PATH"** during installation.

#### B5. Install required libraries

```bash
pip install pandas pyarrow praw requests
```

---

## Contributing

### Option A: GitHub Desktop (Recommended)

#### A1. Pull the latest changes

In GitHub Desktop, click **Fetch origin** (top bar), then **Pull origin** if there are new changes.

#### A2. Create a branch

Recommended if more than one person is working, or if you want someone to review before merging:

1. In GitHub Desktop, click **Current Branch** at the top → **New Branch**
2. Name it `your-name/feature-description` and click **Create Branch**

#### A3. Make your changes

Make your changes in VSCode.

#### A4. Commit your changes

1. In GitHub Desktop, review changed files in the left panel
2. Write a short summary in the **Summary** box (bottom-left)
3. Click **Commit to your-name/feature-description**

#### A5. Push your branch

Click **Push origin** (top bar).

#### A6. Open a Pull Request

Only if you created a branch:

1. Click **Create Pull Request** — this opens GitHub in your browser
2. Add a description and submit for team review

> If you did **not** create a branch, just commit and click **Push origin** — no Pull Request needed.

---

### Option B: Command Line

#### B1. Pull the latest changes

```bash
git pull origin
```

#### B2. Create a branch

Do this step if you want someone to review before saving to the repository, or if more than one person is working on code at the same time:

```bash
git checkout -b your-name/feature-description
```

#### B3. Set upstream to main

This ensures `git pull` on your branch pulls from the main branch on GitHub:

```bash
git branch --set-upstream-to=origin/main your-name/feature-description
```

#### B5. Stage and commit your changes

```bash
git add file.py file.csv file.parquet folder/
git commit -m "Brief description of what you did"
```

#### B6. Push your branch

If you created a branch:

```bash
git push -u origin your-name/feature-description
```

If you did not create a branch:

```bash
git push origin
```

#### B7. Open a Pull Request

Open a **Pull Request** on GitHub for the team to review before merging. Not required if you did not create a branch.

---

### Guidelines

- Write clear commit messages explaining what and why
