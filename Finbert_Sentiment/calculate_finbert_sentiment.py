"""Compute FinBERT sentiment scores for Reddit posts.

This script reads:
  - Data/reddit_data_2007_2024.csv (contains `content` text)
  - 2024 Reddit mentions/top25_most_mentioned_2024.csv
  - 2024 Reddit mentions/bottom25_least_mentioned_2024.csv
  - 2024 Reddit mentions/no_mentions_2024.csv

It filters the posts to only those with tickers in the above lists, scores each post with FinBERT,
and saves both per-post sentiment and per-ticker aggregated summaries.

Requirements:
  - python (run from the workspace root)
  - transformers

Usage:
  python calculate_finbert_sentiment.py
"""

import os
from pathlib import Path

import pandas as pd
from transformers import pipeline
import torch


def load_ticker_groups(root: Path):
    """Load the 3 ticker files and return a merged DataFrame with a `group` column."""
    mention_dir = root / "2024 Reddit mentions"

    files = [
        (mention_dir / "top25_most_mentioned_2024.csv", "top25"),
        (mention_dir / "bottom25_least_mentioned_2024.csv", "bottom25"),
        (mention_dir / "no_mentions_2024.csv", "no_mentions"),
    ]

    all_rows = []
    for path, group in files:
        if not path.exists():
            raise FileNotFoundError(f"Missing expected ticker list: {path}")
        df = pd.read_csv(path, usecols=["ticker"])
        df["group"] = group
        all_rows.append(df)

    return pd.concat(all_rows, ignore_index=True)


def main():
    root = Path(__file__).resolve().parent.parent

    out_dir = root / "Finbert_Sentiment_2024"
    out_dir.mkdir(exist_ok=True)

    print("Loading ticker groups...")
    ticker_groups = load_ticker_groups(root)
    ticker_groups = ticker_groups.drop_duplicates(subset=["ticker", "group"])
    target_tickers = set(ticker_groups["ticker"])

    print(f"Tickers to score: {len(target_tickers)}")

    print("Loading reddit posts...")
    reddit_path = root / "Data" / "reddit_data_2007_2024.csv"
    reddit = pd.read_csv(reddit_path, usecols=["date", "ticker", "content"], parse_dates=["date"])  # noqa: E501

    # Filter to only requested tickers
    reddit = reddit[reddit["ticker"].isin(target_tickers)].copy()

    # Drop empty/NaN content
    reddit = reddit.dropna(subset=["content"]).reset_index(drop=True)

    print(f"Posts after filtering: {len(reddit):,}")

    # Setup FinBERT pipeline
    device = 0 if torch.cuda.is_available() else -1
    print(f"Initializing FinBERT pipeline (device={device})...")
    nlp = pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert",
        return_all_scores=True,
        device=device,
    )

    # Compute sentiment per row in batches
    batch_size = 32
    results = []
    for start in range(0, len(reddit), batch_size):
        batch_texts = reddit.loc[start : start + batch_size - 1, "content"].astype(str).tolist()
        scored = nlp(batch_texts, truncation=True, max_length=512)
        results.extend(scored)
        if (start // batch_size) % 10 == 0:
            print(f"  processed {min(start + batch_size, len(reddit)):,}/{len(reddit):,} posts")

    # Convert to DataFrame
    def _row_to_scores(row_scores):
        # HuggingFace pipeline sometimes returns:
        #   - a dict for a single input (e.g., {'label': 'neutral', 'score': 0.87})
        #   - a list of dicts when return_all_scores=True (not always honored)
        if isinstance(row_scores, dict):
            row_scores = [row_scores]

        # Build a map from label -> score, defaulting missing labels to 0
        d = {label: 0.0 for label in ["positive", "negative", "neutral"]}
        for item in row_scores:
            label = item.get("label", "").lower()
            score = float(item.get("score", 0.0))
            if label in d:
                d[label] = score

        d["sentiment_value"] = d["positive"] - d["negative"]
        # The top label is the highest-scoring one (if we only got a single label, it will still work)
        top_item = max(row_scores, key=lambda r: float(r.get("score", 0.0)))
        d["top_label"] = top_item.get("label")
        d["top_score"] = float(top_item.get("score", 0.0))
        return d

    sentiment_df = pd.DataFrame([_row_to_scores(r) for r in results])
    reddit_scored = pd.concat([reddit.reset_index(drop=True), sentiment_df], axis=1)

    # Add group metadata
    ticker_to_group = ticker_groups.set_index("ticker")["group"].to_dict()
    reddit_scored["group"] = reddit_scored["ticker"].map(ticker_to_group)

    # Save detailed results
    out_path = out_dir / "reddit_finbert_sentiment_posts.csv"
    reddit_scored.to_csv(out_path, index=False)
    print(f"Saved per-post sentiment CSV: {out_path}")

    # Compute a daily sentiment value per ticker (average of posts per day)
    # Then compute the mean of daily sentiment values (each day weighted equally).
    daily_sentiment = (
        reddit_scored
        .groupby(["ticker", "date"])["sentiment_value"]
        .mean()
        .reset_index(name="daily_sentiment_value")
    )

    mean_daily_sentiment = (
        daily_sentiment
        .groupby("ticker")["daily_sentiment_value"]
        .mean()
        .rename("mean_daily_sentiment_value")
    )

    # Compute weekly averages per ticker using week-ending Friday to match returns data.
    weekly_sentiment = (
        reddit_scored
        .groupby(["ticker", pd.Grouper(key="date", freq="W-FRI")])
        .agg(
            weekly_sentiment_value=("sentiment_value", "mean"),
            weekly_top_score=("top_score", "mean"),
        )
        .reset_index()
    )

    mean_weekly_sentiment = (
        weekly_sentiment
        .groupby("ticker")["weekly_sentiment_value"]
        .mean()
        .rename("mean_weekly_sentiment_value")
    )

    mean_weekly_confidence = (
        weekly_sentiment
        .groupby("ticker")["weekly_top_score"]
        .mean()
        .rename("mean_weekly_confidence")
    )

    # Aggregate by ticker + group
    agg = (
        reddit_scored
        .groupby(["group", "ticker"])
        .agg(
            post_count=("sentiment_value", "count"),
            mean_sentiment_value=("sentiment_value", "mean"),
            mean_positive=("positive", "mean"),
            mean_negative=("negative", "mean"),
            mean_neutral=("neutral", "mean"),
            mean_top_score=("top_score", "mean"),
        )
        .reset_index()
    )

    # Add the daily-mean sentiment value to the ticker summary
    agg = agg.merge(mean_daily_sentiment, on="ticker", how="left")
    agg = agg.merge(mean_weekly_sentiment, on="ticker", how="left")
    agg = agg.merge(mean_weekly_confidence, on="ticker", how="left")

    # Classify the overall sentiment direction for easier reading
    # NOTE: the +/-0.01 cutoff is a small tolerance to avoid labeling tiny floating
    # noise as positive/negative. It can be adjusted depending on how sensitive
    # you want the classification to be.
    def _overall_sentiment(v: float) -> str:
        if v > 0.01:
            return "positive"
        if v < -0.01:
            return "negative"
        return "neutral"

    agg["overall_sentiment"] = agg["mean_sentiment_value"].apply(_overall_sentiment)

    # Order output for readability
    agg = agg.sort_values(
        ["group", "overall_sentiment", "mean_sentiment_value"],
        ascending=[True, False, False],
    )

    out_summary = out_dir / "reddit_finbert_sentiment_by_ticker.csv"
    agg.to_csv(out_summary, index=False)
    print(f"Saved per-ticker sentiment summary CSV: {out_summary}")


if __name__ == "__main__":
    main()
