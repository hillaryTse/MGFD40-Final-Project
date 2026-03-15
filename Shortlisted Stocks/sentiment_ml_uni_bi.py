"""
ML Combined Unigram + Bigram Sentiment Analyser — Top 25 & Bottom 25 Stocks
============================================================================
Reads pre-fetched Reddit data from parquet (2007-2024).
Filters to 2024 rows for the top/bottom 25 tickers.
Scores each post with both ML unigram and bigram dictionaries,
then averages the two normalised scores into one combined score.

SETUP
-----
1. pip install pandas pyarrow

2. Place these files in the same folder as this script:
   - top25_stocks.csv
   - bottom25_stocks.csv

3. Run:
       python sentiment_ml_uni_bi.py
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

PARQUET_PATH     = os.path.join(SCRIPT_DIR, "../Data/reddit_data_2007_2024.parquet")
TOP25_CSV        = os.path.join(SCRIPT_DIR, "top25_stocks.csv")
BOTTOM25_CSV     = os.path.join(SCRIPT_DIR, "bottom25_stocks.csv")
POS_UNIGRAM_FILE = os.path.join(SCRIPT_DIR, "../dictionary/ML_positive_unigram.txt")
NEG_UNIGRAM_FILE = os.path.join(SCRIPT_DIR, "../dictionary/ML_negative_unigram.txt")
POS_BIGRAM_FILE  = os.path.join(SCRIPT_DIR, "../dictionary/ML_positive_bigram.txt")
NEG_BIGRAM_FILE  = os.path.join(SCRIPT_DIR, "../dictionary/ML_negative_bigram.txt")
OUTPUT_DIR       = os.path.join(SCRIPT_DIR, "output-uni-bi")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUT_PATH      = os.path.join(OUTPUT_DIR, "sentiment_summary_uni_bi_top25.csv")
FULL_OUT_PATH = os.path.join(OUTPUT_DIR, "sentiment_full_uni_bi_top25.json")

MIN_TOKENS  = 5
MIN_BIGRAMS = 5


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


print("Loading ML dictionaries...")
POS_UNIGRAM = load_dictionary(POS_UNIGRAM_FILE)
NEG_UNIGRAM = load_dictionary(NEG_UNIGRAM_FILE)
POS_BIGRAM  = load_dictionary(POS_BIGRAM_FILE)
NEG_BIGRAM  = load_dictionary(NEG_BIGRAM_FILE)

if not any([POS_UNIGRAM, NEG_UNIGRAM, POS_BIGRAM, NEG_BIGRAM]):
    raise SystemExit("No dictionary files found. Check file paths and rerun.")
print()


# ── Text Scoring ──────────────────────────────────────────────────────────────
def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if t]


def get_bigrams(tokens):
    return [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]


def score_text(text):
    tokens  = tokenize(text)
    bigrams = get_bigrams(tokens)

    has_unigrams = len(tokens)  >= MIN_TOKENS
    has_bigrams  = len(bigrams) >= MIN_BIGRAMS

    if not has_unigrams and not has_bigrams:
        return None

    # Unigram score
    if has_unigrams:
        uni_pos   = [t for t in tokens  if t in POS_UNIGRAM]
        uni_neg   = [t for t in tokens  if t in NEG_UNIGRAM]
        uni_score = (len(uni_pos) - len(uni_neg)) / len(tokens)
    else:
        uni_pos = uni_neg = []
        uni_score = 0.0

    # Bigram score
    if has_bigrams:
        bi_pos   = [b for b in bigrams if b in POS_BIGRAM]
        bi_neg   = [b for b in bigrams if b in NEG_BIGRAM]
        bi_score = (len(bi_pos) - len(bi_neg)) / len(bigrams)
    else:
        bi_pos = bi_neg = []
        bi_score = 0.0

    # Combined: average of the two normalised scores
    n_components   = (1 if has_unigrams else 0) + (1 if has_bigrams else 0)
    combined_score = (uni_score + bi_score) / n_components

    if combined_score > 0:
        sentiment = "bullish"
    elif combined_score < 0:
        sentiment = "bearish"
    else:
        sentiment = "neutral"

    return {
        "uni_pos_hits":    len(uni_pos),
        "uni_neg_hits":    len(uni_neg),
        "total_tokens":    len(tokens),
        "uni_score":       round(uni_score,      6),
        "bi_pos_hits":     len(bi_pos),
        "bi_neg_hits":     len(bi_neg),
        "total_bigrams":   len(bigrams),
        "bi_score":        round(bi_score,       6),
        "combined_score":  round(combined_score, 6),
        "sentiment":       sentiment,
        "matched_uni_pos": uni_pos[:10],
        "matched_uni_neg": uni_neg[:10],
        "matched_bi_pos":  bi_pos[:10],
        "matched_bi_neg":  bi_neg[:10],
    }


# ── Sentiment Aggregation ─────────────────────────────────────────────────────
def compute_sentiment(df_posts, tickers):
    results = []

    for ticker in sorted(tickers):
        subset = df_posts[df_posts["ticker"] == ticker]
        if subset.empty:
            print(f"  ${ticker}: no posts found in dataset for this period")
            continue

        counts          = defaultdict(int)
        combined_scores = []
        total_uni_pos   = 0
        total_uni_neg   = 0
        total_bi_pos    = 0
        total_bi_neg    = 0
        total_tokens    = 0
        total_bigrams   = 0
        skipped         = 0
        samples         = {"bullish": [], "bearish": []}

        for _, row in subset.iterrows():
            text   = f"{row.get('subject', '')} {row.get('content', '')}".strip()
            result = score_text(text)
            if result is None:
                skipped += 1
                continue

            counts[result["sentiment"]] += 1
            combined_scores.append(result["combined_score"])
            total_uni_pos += result["uni_pos_hits"]
            total_uni_neg += result["uni_neg_hits"]
            total_bi_pos  += result["bi_pos_hits"]
            total_bi_neg  += result["bi_neg_hits"]
            total_tokens  += result["total_tokens"]
            total_bigrams += result["total_bigrams"]

            sent = result["sentiment"]
            if sent in ("bullish", "bearish") and len(samples[sent]) < 5:
                samples[sent].append({
                    "date":            str(row.get("date", "")),
                    "text":            text[:150],
                    "combined_score":  result["combined_score"],
                    "uni_score":       result["uni_score"],
                    "bi_score":        result["bi_score"],
                    "matched_uni_pos": result["matched_uni_pos"],
                    "matched_uni_neg": result["matched_uni_neg"],
                    "matched_bi_pos":  result["matched_bi_pos"],
                    "matched_bi_neg":  result["matched_bi_neg"],
                })

        total_signals = counts["bullish"] + counts["bearish"] + counts["neutral"]
        if total_signals == 0:
            print(f"  ${ticker}: posts found but all too short to score")
            continue

        bull_pct     = round(counts["bullish"] / total_signals * 100, 1)
        bear_pct     = round(counts["bearish"] / total_signals * 100, 1)
        neut_pct     = round(counts["neutral"] / total_signals * 100, 1)
        avg_combined = round(sum(combined_scores) / len(combined_scores), 6) if combined_scores else 0.0

        # Corpus-level combined score
        uni_corpus = (total_uni_pos - total_uni_neg) / total_tokens  if total_tokens  > 0 else 0.0
        bi_corpus  = (total_bi_pos  - total_bi_neg)  / total_bigrams if total_bigrams > 0 else 0.0
        n = (1 if total_tokens > 0 else 0) + (1 if total_bigrams > 0 else 0)
        corpus_combined = round((uni_corpus + bi_corpus) / n, 6) if n > 0 else 0.0

        overall = (
            "bullish" if counts["bullish"] > counts["bearish"] else
            "bearish" if counts["bearish"] > counts["bullish"] else
            "neutral"
        )

        print(f"  ${ticker:<6} posts={len(subset)}  Bull={bull_pct}%  "
              f"Bear={bear_pct}%  score={corpus_combined:+.4f}  {overall.upper()}")

        results.append({
            "ticker":                ticker,
            "mentions":              TICKER_MENTIONS.get(ticker, 0),
            "posts_found":           len(subset),
            "total_signals":         total_signals,
            "skipped_too_short":     skipped,
            "bullish_count":         counts["bullish"],
            "bearish_count":         counts["bearish"],
            "neutral_count":         counts["neutral"],
            "bullish_pct":           bull_pct,
            "bearish_pct":           bear_pct,
            "neutral_pct":           neut_pct,
            "total_uni_pos_hits":    total_uni_pos,
            "total_uni_neg_hits":    total_uni_neg,
            "total_bi_pos_hits":     total_bi_pos,
            "total_bi_neg_hits":     total_bi_neg,
            "total_tokens":          total_tokens,
            "total_bigrams":         total_bigrams,
            "avg_combined_score":    avg_combined,
            "corpus_combined_score": corpus_combined,
            "overall_sentiment":     overall,
            "sample_bullish":        samples["bullish"],
            "sample_bearish":        samples["bearish"],
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
        "total_uni_pos_hits", "total_uni_neg_hits",
        "total_bi_pos_hits",  "total_bi_neg_hits",
        "total_tokens", "total_bigrams",
        "avg_combined_score", "corpus_combined_score", "overall_sentiment",
    ]

    for label, tickers, out_csv, out_json in [
        ("Top 25",    TOP25_TICKERS,    OUT_PATH,                                              FULL_OUT_PATH),
        ("Bottom 25", BOTTOM25_TICKERS, OUT_PATH.replace("uni_bi_top25", "uni_bi_bottom25"),
                                        FULL_OUT_PATH.replace("uni_bi_top25", "uni_bi_bottom25")),
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

        sorted_results = sorted(results, key=lambda x: x["corpus_combined_score"], reverse=True)

        print(f"── Sentiment Leaderboard — {label} (r/pennystocks · 2024) ─────────────────")
        print(f"{'Ticker':<8} {'Posts':>6} {'Signals':>8} {'Bull%':>7} {'Bear%':>7} {'Score':>10}  Verdict")
        print("─" * 68)
        for r in sorted_results:
            print(
                f"{r['ticker']:<8} {r['posts_found']:>6} {r['total_signals']:>8} "
                f"{r['bullish_pct']:>6.1f}% {r['bearish_pct']:>6.1f}% "
                f"{r['corpus_combined_score']:>+10.4f}  {r['overall_sentiment'].upper()}"
            )

        print(f"\n── Top 5 Most Bullish ({label}) ────────────────")
        for r in sorted_results[:5]:
            print(f"  ${r['ticker']:<6} score={r['corpus_combined_score']:+.4f}  "
                  f"{r['bullish_pct']}% bullish")

        print(f"\n── Top 5 Most Bearish ({label}) ────────────────")
        for r in sorted_results[-5:][::-1]:
            print(f"  ${r['ticker']:<6} score={r['corpus_combined_score']:+.4f}  "
                  f"{r['bearish_pct']}% bearish")
