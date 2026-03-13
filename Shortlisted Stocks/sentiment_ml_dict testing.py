"""
ML Dictionary Sentiment Scraper — Top 25 Stocks
=================================================
Scans r/pennystocks via Arctic Shift API only (no Reddit API needed)
Date range: January 1 2024 - December 31 2024
Uses Loughran-McDonald (ML) financial bigram dictionaries.

SETUP
-----
1. pip install requests pandas

2. Place these files in the same folder as this script:
   - ML_positive_bigram_20151231.txt
   - ML_negative_bigram_20151231.txt
   - top25_stocks.csv

3. Run:
       python sentiment_ml_dict.py
"""

import os
import re
import csv
import json
import time
import requests
import pandas as pd
from datetime import datetime, timezone
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────
ARCTIC_BASE      = "https://arctic-shift.photon-reddit.com/api"
SUBREDDIT        = "pennystocks"
START_DATE       = "2024-01-01"
END_DATE         = "2024-12-31"
BATCH_SIZE       = 100
CHECKPOINT_EVERY = 500

TICKERS_CSV        = "top25_stocks.csv"
POSITIVE_DICT_FILE = "ML_positive_bigram_20151231.txt"
NEGATIVE_DICT_FILE = "ML_negative_bigram_20151231.txt"
POSTS_CACHE        = "posts_cache.csv"
CHECKPOINT_FILE    = "checkpoint.txt"
OUT_PATH           = "sentiment_summary.csv"
FULL_OUT_PATH      = "sentiment_full.json"

MIN_BIGRAMS = 5


# ── Helpers ───────────────────────────────────────────────────────────────────
def ts_to_date(ts):
    return datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")


def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            val = f.read().strip()
            return int(val) if val else 0
    return 0


def save_checkpoint(ts):
    with open(CHECKPOINT_FILE, "w") as f:
        f.write(str(ts))


def append_posts(records):
    file_exists = os.path.exists(POSTS_CACHE)
    with open(POSTS_CACHE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "date", "ticker", "post_id", "title", "selftext",
            "score", "num_comments", "created_utc"
        ])
        if not file_exists:
            writer.writeheader()
        writer.writerows(records)


# ── Load Tickers ──────────────────────────────────────────────────────────────
df_tickers = pd.read_csv(TICKERS_CSV)
TICKERS = set(df_tickers["ticker"].str.upper().tolist())
TICKER_MENTIONS = dict(zip(df_tickers["ticker"].str.upper(), df_tickers["mentions"]))
print(f"Loaded {len(TICKERS)} tickers: {', '.join(sorted(TICKERS))}\n")


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
POSITIVE_DICT = load_dictionary(POSITIVE_DICT_FILE)
NEGATIVE_DICT = load_dictionary(NEGATIVE_DICT_FILE)

if not POSITIVE_DICT and not NEGATIVE_DICT:
    raise SystemExit("No dictionary files found. Check file paths and rerun.")
print()


# ── Ticker Extraction ─────────────────────────────────────────────────────────
def extract_tickers(text):
    found = []
    text_upper = text.upper()
    for ticker in TICKERS:
        pattern = r'(?<![A-Z\$])' + re.escape(ticker) + r'(?![A-Z])'
        if re.search(pattern, text_upper):
            found.append(ticker)
    return found


# ── Arctic Shift Scraper ──────────────────────────────────────────────────────
def scrape_arctic_shift(start_date, end_date, batch_size=BATCH_SIZE):
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    end_ts   = int(datetime.strptime(end_date,   "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())

    saved = load_checkpoint()
    if saved and saved > start_ts:
        after = saved
        print(f"Resuming from checkpoint: {ts_to_date(after)}")
    else:
        after = start_ts
        print(f"Starting fresh from {start_date}")

    print(f"Fetching r/{SUBREDDIT} posts up to {end_date} via Arctic Shift...")

    pending     = []
    total_saved = 0

    while after < end_ts:
        data = None
        for attempt in range(5):
            try:
                params = {
                    "subreddit": SUBREDDIT,
                    "after":     after,
                    "before":    end_ts,
                    "limit":     batch_size,
                    "sort":      "asc",
                }
                resp = requests.get(
                    f"{ARCTIC_BASE}/posts/search",
                    params=params,
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json().get("data", [])
                break
            except (requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout) as e:
                wait = 2 ** attempt
                print(f"  Connection error (attempt {attempt+1}/5): {e}. Retrying in {wait}s...")
                time.sleep(wait)

        if data is None:
            print("  Failed after 5 retries — saving progress and exiting.")
            if pending:
                append_posts(pending)
                total_saved += len(pending)
            save_checkpoint(after)
            print(f"  Checkpoint saved at {ts_to_date(after)}. Re-run to resume.")
            return total_saved

        if not data:
            break

        for post in data:
            text    = f"{post.get('title', '')} {post.get('selftext', '')}"
            tickers = extract_tickers(text)
            for ticker in tickers:
                pending.append({
                    "date":         ts_to_date(post["created_utc"]),
                    "ticker":       ticker,
                    "post_id":      post["id"],
                    "title":        post.get("title", ""),
                    "selftext":     post.get("selftext", ""),
                    "score":        post.get("score", 0),
                    "num_comments": post.get("num_comments", 0),
                    "created_utc":  post["created_utc"],
                })

        after = data[-1]["created_utc"] + 1

        if len(pending) >= CHECKPOINT_EVERY:
            append_posts(pending)
            total_saved += len(pending)
            save_checkpoint(after)
            print(f"  [{ts_to_date(after)}] Saved {total_saved} mentions total (checkpointed)")
            pending = []
        else:
            print(f"  Fetched up to {ts_to_date(after)} — {total_saved + len(pending)} mentions so far")

        time.sleep(0.5)

    if pending:
        append_posts(pending)
        total_saved += len(pending)

    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

    return total_saved


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

    if len(bigrams) < MIN_BIGRAMS:
        return None

    matched_pos = [b for b in bigrams if b in POSITIVE_DICT]
    matched_neg = [b for b in bigrams if b in NEGATIVE_DICT]

    pos_hits = len(matched_pos)
    neg_hits = len(matched_neg)
    total    = len(bigrams)
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
        "total_bigrams":   total,
        "sentiment_score": round(score, 6),
        "sentiment":       sentiment,
        "matched_pos":     matched_pos[:10],
        "matched_neg":     matched_neg[:10],
    }


# ── Sentiment Aggregation ─────────────────────────────────────────────────────
def compute_sentiment(df_posts):
    results = []

    for ticker in sorted(TICKERS):
        subset = df_posts[df_posts["ticker"] == ticker]
        if subset.empty:
            print(f"  ${ticker}: no posts found — not mentioned in r/{SUBREDDIT} during this period")
            continue

        counts         = defaultdict(int)
        scores         = []
        total_pos_hits = 0
        total_neg_hits = 0
        total_bigrams  = 0
        skipped        = 0
        samples        = {"bullish": [], "bearish": []}

        for _, row in subset.iterrows():
            text   = f"{row.get('title', '')} {row.get('selftext', '')}".strip()
            result = score_text(text)
            if result is None:
                skipped += 1
                continue

            counts[result["sentiment"]] += 1
            scores.append(result["sentiment_score"])
            total_pos_hits += result["pos_hits"]
            total_neg_hits += result["neg_hits"]
            total_bigrams  += result["total_bigrams"]

            sent = result["sentiment"]
            if sent in ("bullish", "bearish") and len(samples[sent]) < 5:
                samples[sent].append({
                    "date":         row.get("date", ""),
                    "text":         text[:150],
                    "reddit_score": int(row.get("score", 0)),
                    "pos_hits":     result["pos_hits"],
                    "neg_hits":     result["neg_hits"],
                    "sent_score":   result["sentiment_score"],
                    "matched_pos":  result["matched_pos"],
                    "matched_neg":  result["matched_neg"],
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
            (total_pos_hits - total_neg_hits) / total_bigrams, 6
        ) if total_bigrams > 0 else 0.0

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
            "total_bigrams":          total_bigrams,
            "avg_sentiment_score":    avg_score,
            "corpus_sentiment_score": corpus_score,
            "overall_sentiment":      overall,
            "sample_bullish":         samples["bullish"],
            "sample_bearish":         samples["bearish"],
        })

    return results


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # Step 1: Scrape posts from Arctic Shift
    print(f"=== STEP 1: Scraping r/{SUBREDDIT} | {START_DATE} to {END_DATE} ===\n")
    total = scrape_arctic_shift(START_DATE, END_DATE)

    # Step 2: Deduplicate cache
    df = pd.read_csv(POSTS_CACHE)
    before = len(df)
    df = df.drop_duplicates(subset=["post_id", "ticker"])
    df = df.sort_values("date").reset_index(drop=True)
    df.to_csv(POSTS_CACHE, index=False)
    print(f"\nPosts cache: {len(df)} unique ticker-mention rows "
          f"(removed {before - len(df)} dupes)\n")
    print(df.head(10))

    # Step 3: Score sentiment per ticker
    print(f"\n=== STEP 2: Scoring sentiment with ML dictionaries ===\n")
    results = compute_sentiment(df)

    if not results:
        print("\nNo results — tickers may not have been mentioned in "
              f"r/{SUBREDDIT} during this period.")
    else:
        # Save full JSON with samples
        with open(FULL_OUT_PATH, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved {FULL_OUT_PATH}")

        # Save summary CSV
        fields = [
            "ticker", "mentions", "posts_found", "total_signals", "skipped_too_short",
            "bullish_count", "bearish_count", "neutral_count",
            "bullish_pct", "bearish_pct", "neutral_pct",
            "total_pos_hits", "total_neg_hits", "total_bigrams",
            "avg_sentiment_score", "corpus_sentiment_score", "overall_sentiment",
        ]
        with open(OUT_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved {OUT_PATH}\n")

        # Leaderboard
        sorted_results = sorted(
            results, key=lambda x: x["corpus_sentiment_score"], reverse=True
        )

        print("── Sentiment Leaderboard (r/pennystocks · 2024) ────────────────────────────")
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

        print("\n── Top 5 Most Bullish ────────────────")
        for r in sorted_results[:5]:
            print(f"  ${r['ticker']:<6} score={r['corpus_sentiment_score']:+.4f}  "
                  f"{r['bullish_pct']}% bullish  pos_hits={r['total_pos_hits']}")

        print("\n── Top 5 Most Bearish ────────────────")
        for r in sorted_results[-5:][::-1]:
            print(f"  ${r['ticker']:<6} score={r['corpus_sentiment_score']:+.4f}  "
                  f"{r['bearish_pct']}% bearish  neg_hits={r['total_neg_hits']}")
