"""Compute FinBERT sentiment scores for Reddit posts (2019-2024).

Sources:
  - All_Year_Reddit_Data/Reddit_Data_All_Years/reddit_data_2013_2024.parquet
  - All_Year_Reddit_Data/{year}_reddit_mentions/reddit_mentions_{year}.csv  (2019-2023)
  - 2024_reddit_mentions/reddit_mentions_2024.csv  (full 2024)

Scores subject+content for all tickers mentioned in 2019-2024.
Long posts are chunked into 510-token segments and scores averaged equally.
sentiment_value = positive_score - negative_score, ranging from -1 to +1.
Results are written in batches with checkpointing for crash recovery.

Usage:
  python Finbert_Sentiment/calc_finbert_sentiment_2019_2024.py
"""

import os
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, pipeline
from tqdm import tqdm

# Use all available CPU cores for torch inference
torch.set_num_threads(os.cpu_count())
# Disable HuggingFace tokenizer parallelism to avoid conflict with torch threading
os.environ["TOKENIZERS_PARALLELISM"] = "false"

YEARS       = list(range(2019, 2025))
BATCH_SIZE  = 32     # chunks per inference batch
MACRO_BATCH = 1000   # posts per checkpoint write to CSV
CHUNK_SIZE  = 510    # tokens per chunk (512 - 2 reserved for [CLS]/[SEP] special tokens)
MODEL       = "ProsusAI/finbert"


def load_mentioned_tickers(root: Path) -> set:
    """Return the set of all tickers mentioned in Reddit posts across 2019-2024.

    Only posts mentioning these tickers will be scored, avoiding unnecessary inference
    on irrelevant posts from the full Reddit archive.
    """
    frames = []
    for year in range(2019, 2024):
        p = root / "All_Year_Reddit_Data" / f"{year}_reddit_mentions" / f"reddit_mentions_{year}.csv"
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")
        frames.append(pd.read_csv(p, usecols=["ticker"]))
    p2024 = root / "2024_reddit_mentions" / "reddit_mentions_2024.csv"
    if not p2024.exists():
        raise FileNotFoundError(f"Missing: {p2024}")
    frames.append(pd.read_csv(p2024, usecols=["ticker"]))
    return set(pd.concat(frames, ignore_index=True)["ticker"].unique())


def texts_to_chunks(texts, tokenizer):
    """Split texts into <=510-token chunks. Returns flat chunk list and per-post chunk counts.

    FinBERT (BERT-based) has a hard 512-token input limit. Posts exceeding this are split
    into CHUNK_SIZE-token segments so no content is silently truncated.
    """
    all_chunks, chunk_counts = [], []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) <= CHUNK_SIZE:
            # Post fits in one pass - no chunking needed
            all_chunks.append(text)
            chunk_counts.append(1)
        else:
            # Split token IDs into segments and decode back to text for the pipeline
            chunks = [
                tokenizer.decode(ids[i : i + CHUNK_SIZE])
                for i in range(0, len(ids), CHUNK_SIZE)
            ]
            all_chunks.extend(chunks)
            chunk_counts.append(len(chunks))
    return all_chunks, chunk_counts


def score_chunks(chunks, nlp):
    """Run FinBERT inference on all chunks in mini-batches.

    Returns a list of {positive, negative, neutral} score dicts, one per chunk.
    return_all_scores=True ensures we get all three label probabilities, not just the top one.
    """
    scores = []
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        for result in nlp(batch, truncation=True, max_length=512):
            if isinstance(result, dict):
                result = [result]
            scores.append({item["label"].lower(): float(item["score"]) for item in result})
    return scores


def aggregate_scores(chunk_scores, chunk_counts):
    """Average chunk scores equally per post, return list of score dicts.

    sentiment_value = positive - negative, ranging from -1 (most bearish) to +1 (most bullish).
    For multi-chunk posts, scores are averaged equally across all chunks.
    """
    rows, idx = [], 0
    for n in chunk_counts:
        post = chunk_scores[idx : idx + n]
        idx += n
        pos = sum(c.get("positive", 0.0) for c in post) / n
        neg = sum(c.get("negative", 0.0) for c in post) / n
        neu = sum(c.get("neutral",  0.0) for c in post) / n
        label_scores = {"positive": pos, "negative": neg, "neutral": neu}
        top_label = max(label_scores, key=label_scores.get)
        rows.append({
            "positive":        pos,
            "negative":        neg,
            "neutral":         neu,
            "sentiment_value": pos - neg,   # main signal used in regressions
            "top_label":       top_label,
            "top_score":       label_scores[top_label],
        })
    return rows


def main():
    root     = Path(__file__).resolve().parent.parent
    out_dir  = root / "Finbert_Sentiment"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "finbert_sentiment_2019_2024.csv"

    print("Loading mentioned tickers (2019-2024)...")
    target_tickers = load_mentioned_tickers(root)
    print(f"Tickers: {len(target_tickers)}")

    print("Loading Reddit posts...")
    reddit = pd.read_parquet(
        root / "All_Year_Reddit_Data" / "Reddit_Data_All_Years" / "reddit_data_2013_2024.parquet",
        columns=["date", "ticker", "subject", "content"],
    )
    reddit["date"] = pd.to_datetime(reddit["date"])
    # Filter to posts mentioning our tickers within the study period
    reddit = reddit[
        reddit["ticker"].isin(target_tickers) &
        reddit["date"].dt.year.isin(YEARS)
    ].dropna(subset=["content"]).reset_index(drop=True)
    # Concatenate subject and content as the text input to FinBERT
    reddit["text"] = reddit["subject"].fillna("") + " " + reddit["content"].fillna("")
    print(f"Posts to score: {len(reddit):,}")

    # Resume from checkpoint: count rows already written to avoid re-scoring on restart
    start_idx = 0
    if out_path.exists():
        start_idx = sum(1 for _ in open(out_path)) - 1  # subtract header row
        start_idx = max(start_idx, 0)
        if start_idx > 0:
            print(f"Resuming from post {start_idx:,}")

    # device=-1 forces CPU inference (no GPU required)
    print(f"Initializing FinBERT (CPU, threads={os.cpu_count()})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    nlp = pipeline(
        "sentiment-analysis",
        model=MODEL,
        tokenizer=tokenizer,
        return_all_scores=True,  # return probabilities for all three labels
        device=-1,
    )

    total = len(reddit)
    with tqdm(total=total - start_idx, desc="Posts scored", unit="post") as pbar:
        for batch_start in range(start_idx, total, MACRO_BATCH):
            batch_end = min(batch_start + MACRO_BATCH, total)
            batch_df  = reddit.iloc[batch_start:batch_end]

            chunks, chunk_counts = texts_to_chunks(batch_df["text"].tolist(), tokenizer)
            post_scores = aggregate_scores(score_chunks(chunks, nlp), chunk_counts)

            result_df = pd.concat(
                [batch_df[["date", "ticker", "content"]].reset_index(drop=True),
                 pd.DataFrame(post_scores)],
                axis=1,
            )
            # Write header only on the first batch; subsequent batches append without header
            result_df.to_csv(out_path, mode="a", header=batch_start == 0, index=False)

            pbar.update(len(batch_df))

    print(f"\nDone. Saved: {out_path}")


if __name__ == "__main__":
    main()
