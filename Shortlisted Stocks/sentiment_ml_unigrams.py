"""
ML Unigram Sentiment Analyser — Top 25 & Bottom 25 Stocks
==========================================================
Reads pre-fetched Reddit data from parquet (2007-2024).
Filters to 2024 rows for the top/bottom 25 tickers.
Uses ML financial unigram dictionaries.

SETUP
-----
1. pip install pandas pyarrow

2. Place these files in the same folder as this script:
   - top25_stocks.csv
   - bottom25_stocks.csv

3. Run:
       python sentiment_ml_unigrams.py
"""

import os
import re
import csv
import json
import pandas as pd
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────
START_DATE = "2024-01-01"
END_DATE   = "2025-01-01"   # exclusive — covers all of Dec 31 2024

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PARQUET_PATH       = os.path.join(SCRIPT_DIR, "../Data/reddit_data_2007_2024.parquet")
TOP25_CSV          = os.path.join(SCRIPT_DIR, "top25_stocks.csv")
BOTTOM25_CSV       = os.path.join(SCRIPT_DIR, "bottom25_stocks.csv")
POSITIVE_DICT_FILE = os.path.join(SCRIPT_DIR, "../dictionary/ML_positive_unigram.txt")
NEGATIVE_DICT_FILE = os.path.join(SCRIPT_DIR, "../dictionary/ML_negative_unigram.txt")
OUTPUT_DIR         = os.path.join(SCRIPT_DIR, "output-unigrams")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUT_PATH      = os.path.join(OUTPUT_DIR, "sentiment_summary_unigram_top25.csv")
FULL_OUT_PATH = os.path.join(OUTPUT_DIR, "sentiment_full_unigram_top25.json")

MIN_TOKENS = 5


# ── Load Tickers ──────────────────────────────────────────────────────────────
df_top25    = pd.read_csv(TOP25_CSV)
df_bottom25 = pd.read_csv(BOTTOM25_CSV)

TOP25_TICKERS    = set(df_top25["ticker"].str.upper().tolist())
BOTTOM25_TICKERS = set(df_bottom25["ticker"].str.upper().tolist())
TICKERS          = TOP25_TICKERS | BOTTOM25_TICKERS

TICKER_MENTIONS = {}
TICKER_MENTIONS.update(zip(df_top25["ticker"].str.upper(),    df_top25["mentions"]))
TICKER_MENTIONS.update(zip(df_bottom25["ticker"].str.upper(), df_bottom25["mentions"]))

print(f"Loaded {len(TOP25_TICKERS)} top tickers:    {', '.join(sorted(TOP25_TICKERS))}")
print(f"Loaded {len(BOTTOM25_TICKERS)} bottom tickers: {', '.join(sorted(BOTTOM25_TICKERS))}\n")


# ── Load Dictionaries ─────────────────────────────────────────────────────────
def load_dictionary(filepath):
    try:
        with open(filepath, "r") as f:
            terms = {line.strip().lower() for line in f if line.strip()}
        print(f"  Loaded {len(terms):,} terms from {filepath}")
        return terms
    except FileNotFoundError:
        print(f"  {filepath} not found — check the file path")
        return set()


print("Loading ML unigram dictionaries...")
POSITIVE_DICT = load_dictionary(POSITIVE_DICT_FILE)
NEGATIVE_DICT = load_dictionary(NEGATIVE_DICT_FILE)

if not POSITIVE_DICT and not NEGATIVE_DICT:
    raise SystemExit("No dictionary files found. Check file paths and rerun.")
print()


# ── Text Scoring ──────────────────────────────────────────────────────────────
def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if t]


def score_text(text):
    tokens = tokenize(text)

    if len(tokens) < MIN_TOKENS:
        return None

    matched_pos = [t for t in tokens if t in POSITIVE_DICT]
    matched_neg = [t for t in tokens if t in NEGATIVE_DICT]

    pos_hits = len(matched_pos)
    neg_hits = len(matched_neg)
    total    = len(tokens)
    score    = (pos_hits - neg_hits) / total

    if pos_hits > neg_hits:
        sentiment = "bullish"
    elif neg_hits > pos_hits:
        sentiment = "bearish"
    else:
        sentiment = "neutral"

    return {
        "pos_hits":        pos_hits,
        "neg_hits":        neg_hits,
        "total_tokens":    total,
        "sentiment_score": round(score, 6),
        "sentiment":       sentiment,
        "matched_pos":     matched_pos[:10],
        "matched_neg":     matched_neg[:10],
    }


# ── Sentiment Aggregation ─────────────────────────────────────────────────────
def compute_sentiment(df_posts, tickers):
    results = []

    for ticker in sorted(tickers):
        subset = df_posts[df_posts["ticker"] == ticker]
        if subset.empty:
            print(f"  ${ticker}: no posts found in dataset for this period")
            continue

        counts         = defaultdict(int)
        scores         = []
        total_pos_hits = 0
        total_neg_hits = 0
        total_tokens   = 0
        skipped        = 0
        samples        = {"bullish": [], "bearish": []}

        for _, row in subset.iterrows():
            text   = f"{row.get('subject', '')} {row.get('content', '')}".strip()
            result = score_text(text)
            if result is None:
                skipped += 1
                continue

            counts[result["sentiment"]] += 1
            scores.append(result["sentiment_score"])
            total_pos_hits += result["pos_hits"]
            total_neg_hits += result["neg_hits"]
            total_tokens   += result["total_tokens"]

            sent = result["sentiment"]
            if sent in ("bullish", "bearish") and len(samples[sent]) < 5:
                samples[sent].append({
                    "date":        str(row.get("date", "")),
                    "text":        text[:150],
                    "pos_hits":    result["pos_hits"],
                    "neg_hits":    result["neg_hits"],
                    "sent_score":  result["sentiment_score"],
                    "matched_pos": result["matched_pos"],
                    "matched_neg": result["matched_neg"],
                })

        total_signals = counts["bullish"] + counts["bearish"] + counts["neutral"]
        if total_signals == 0:
            print(f"  ${ticker}: posts found but all too short to score")
            continue

        bull_pct     = round(counts["bullish"] / total_signals * 100, 1)
        bear_pct     = round(counts["bearish"] / total_signals * 100, 1)
        neut_pct     = round(counts["neutral"] / total_signals * 100, 1)
        avg_score    = round(sum(scores) / len(scores), 6) if scores else 0.0
        corpus_score = round(
            (total_pos_hits - total_neg_hits) / total_tokens, 6
        ) if total_tokens > 0 else 0.0

        overall = (
            "bullish" if counts["bullish"] > counts["bearish"] else
            "bearish" if counts["bearish"] > counts["bullish"] else
            "neutral"
        )

        print(f"  ${ticker:<6} posts={len(subset)}  Bull={bull_pct}%  "
              f"Bear={bear_pct}%  score={corpus_score:+.4f}  {overall.upper()}")

        results.append({
            "ticker":                 ticker,
            "mentions":               TICKER_MENTIONS.get(ticker, 0),
            "posts_found":            len(subset),
            "total_signals":          total_signals,
            "skipped_too_short":      skipped,
            "bullish_count":          counts["bullish"],
            "bearish_count":          counts["bearish"],
            "neutral_count":          counts["neutral"],
            "bullish_pct":            bull_pct,
            "bearish_pct":            bear_pct,
            "neutral_pct":            neut_pct,
            "total_pos_hits":         total_pos_hits,
            "total_neg_hits":         total_neg_hits,
            "total_tokens":           total_tokens,
            "avg_sentiment_score":    avg_score,
            "corpus_sentiment_score": corpus_score,
            "overall_sentiment":      overall,
            "sample_bullish":         samples["bullish"],
            "sample_bearish":         samples["bearish"],
        })

    return results


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # Step 1: Load and filter parquet
    print(f"=== STEP 1: Loading parquet | {START_DATE} to {END_DATE} ===\n")
    df = pd.read_parquet(PARQUET_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= START_DATE) & (df["date"] < END_DATE)]
    df["ticker"] = df["ticker"].str.upper()
    df = df[df["ticker"].isin(TICKERS)].reset_index(drop=True)
    print(f"Loaded {len(df):,} rows covering {df['ticker'].nunique()} tickers in 2024\n")
    print(df.head(10))

    # Step 2: Score sentiment per group
    fields = [
        "ticker", "mentions", "posts_found", "total_signals", "skipped_too_short",
        "bullish_count", "bearish_count", "neutral_count",
        "bullish_pct", "bearish_pct", "neutral_pct",
        "total_pos_hits", "total_neg_hits", "total_tokens",
        "avg_sentiment_score", "corpus_sentiment_score", "overall_sentiment",
    ]

    for label, tickers, out_csv, out_json in [
        ("Top 25",    TOP25_TICKERS,    OUT_PATH,                                                  FULL_OUT_PATH),
        ("Bottom 25", BOTTOM25_TICKERS, OUT_PATH.replace("unigram_top25", "unigram_bottom25"),
                                        FULL_OUT_PATH.replace("unigram_top25", "unigram_bottom25")),
    ]:
        print(f"\n=== STEP 2: Scoring sentiment — {label} ===\n")
        results = compute_sentiment(df, tickers)

        if not results:
            print(f"\nNo results for {label} — tickers not found in dataset for this period.")
            continue

        with open(out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved {out_json}")

        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved {out_csv}\n")

        sorted_results = sorted(results, key=lambda x: x["corpus_sentiment_score"], reverse=True)

        print(f"── Sentiment Leaderboard — {label} (r/pennystocks · 2024) ─────────────────")
        print(f"{'Ticker':<8} {'Posts':>6} {'Signals':>8} {'Bull%':>7} {'Bear%':>7} "
              f"{'PosHits':>8} {'NegHits':>8} {'Score':>10}  Verdict")
        print("─" * 76)
        for r in sorted_results:
            print(
                f"{r['ticker']:<8} {r['posts_found']:>6} {r['total_signals']:>8} "
                f"{r['bullish_pct']:>6.1f}% {r['bearish_pct']:>6.1f}% "
                f"{r['total_pos_hits']:>8} {r['total_neg_hits']:>8} "
                f"{r['corpus_sentiment_score']:>+10.4f}  {r['overall_sentiment'].upper()}"
            )

        print(f"\n── Top 5 Most Bullish ({label}) ────────────────")
        for r in sorted_results[:5]:
            print(f"  ${r['ticker']:<6} score={r['corpus_sentiment_score']:+.4f}  "
                  f"{r['bullish_pct']}% bullish  pos_hits={r['total_pos_hits']}")

        print(f"\n── Top 5 Most Bearish ({label}) ────────────────")
        for r in sorted_results[-5:][::-1]:
            print(f"  ${r['ticker']:<6} score={r['corpus_sentiment_score']:+.4f}  "
                  f"{r['bearish_pct']}% bearish  neg_hits={r['total_neg_hits']}")
