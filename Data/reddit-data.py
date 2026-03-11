"""
Reddit Penny Stock Mention Scraper
-----------------------------------
Strategy:
  - PRAW: fetch recent r/pennystocks posts (limited to ~1000)
  - Arctic Shift API: fetch historical posts 2007-2024 (no auth required)

Output: CSV with columns [date, ticker, post_id, title, score, num_comments, source]
Checkpoints every 5000 records so crashes don't lose progress.
"""

import praw
import requests
import re
import time
import os
import pandas as pd
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Reddit API credentials (fill these in from https://www.reddit.com/prefs/apps)
# ---------------------------------------------------------------------------
REDDIT_CLIENT_ID     = "YOUR_CLIENT_ID"
REDDIT_CLIENT_SECRET = "YOUR_CLIENT_SECRET"
REDDIT_USER_AGENT    = "penny_stock_scraper/0.1 by YOUR_USERNAME"

SUBREDDIT    = "pennystocks"
OUT_PATH     = "reddit_penny_mentions.csv"
CHECKPOINT   = "scrape_checkpoint.txt"   # stores last successfully saved timestamp
CHECKPOINT_EVERY = 5000                  # save to CSV every N new records

# Regex: match $TICKER or standalone 1-5 uppercase letter words
TICKER_PATTERN = re.compile(r'\$([A-Z]{1,5})\b|(?<!\w)([A-Z]{2,5})(?!\w)')
FALSE_POSITIVES = {
    "I", "A", "THE", "AND", "OR", "NOT", "FOR", "IS", "ARE", "TO",
    "IN", "ON", "AT", "BE", "DO", "GO", "IT", "MY", "NO", "OF", "SO",
    "UP", "US", "WE", "DD", "OP", "IMO", "BUY", "SELL", "HOLD", "ETF",
    "CEO", "IPO", "SEC", "OTC", "NYSE", "NASDAQ", "EPS", "PE", "ATH",
    "YOY", "QOQ", "FOMO", "YOLO", "MOON", "GG", "NFT", "USD", "GDP",
    "EOD", "EOW", "PT", "SL", "TP", "RH", "WSB", "LOL", "TBH", "IMO",
}

def extract_tickers(text):
    if not text:
        return []
    tickers = set()
    for m in TICKER_PATTERN.finditer(text):
        ticker = (m.group(1) or m.group(2)).upper()
        if ticker not in FALSE_POSITIVES:
            tickers.add(ticker)
    return list(tickers)

def ts_to_date(utc_ts):
    return datetime.fromtimestamp(utc_ts, tz=timezone.utc).strftime("%Y-%m-%d")

def save_checkpoint(after_ts):
    with open(CHECKPOINT, "w") as f:
        f.write(str(after_ts))

def load_checkpoint():
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT) as f:
            val = f.read().strip()
            if val:
                return int(val)
    return None

def append_records(records):
    """Append a batch of records to the output CSV."""
    df = pd.DataFrame(records)
    write_header = not os.path.exists(OUT_PATH)
    df.to_csv(OUT_PATH, mode="a", header=write_header, index=False)

# ---------------------------------------------------------------------------
# PRAW: recent posts (up to ~1000)
# ---------------------------------------------------------------------------
def scrape_praw(limit=1000):
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
    )
    subreddit = reddit.subreddit(SUBREDDIT)
    records = []
    for post in subreddit.new(limit=limit):
        text = f"{post.title} {post.selftext}"
        tickers = extract_tickers(text)
        for ticker in tickers:
            records.append({
                "date":         ts_to_date(post.created_utc),
                "ticker":       ticker,
                "post_id":      post.id,
                "title":        post.title,
                "score":        post.score,
                "num_comments": post.num_comments,
                "source":       "praw",
            })
    return records

# ---------------------------------------------------------------------------
# Arctic Shift API: historical posts
# ---------------------------------------------------------------------------
ARCTIC_BASE = "https://arctic-shift.photon-reddit.com/api"

def scrape_arctic_shift(start_date, end_date, batch_size=100):
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    end_ts   = int(datetime.strptime(end_date,   "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())

    # Resume from checkpoint if available
    saved = load_checkpoint()
    if saved and saved > start_ts:
        after = saved
        print(f"Resuming from checkpoint: {ts_to_date(after)}")
    else:
        after = start_ts
        print(f"Starting fresh from {start_date}")

    print(f"Fetching r/{SUBREDDIT} posts up to {end_date} via Arctic Shift...")

    pending = []
    total_saved = 0

    while after < end_ts:
        # Retry loop with exponential backoff
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
                break  # success
            except (requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout) as e:
                wait = 2 ** attempt
                print(f"  Connection error (attempt {attempt+1}/5): {e}. Retrying in {wait}s...")
                time.sleep(wait)
        else:
            print("  Failed after 5 retries — saving progress and exiting.")
            if pending:
                append_records(pending)
                total_saved += len(pending)
            save_checkpoint(after)
            print(f"  Checkpoint saved at {ts_to_date(after)}. Re-run to resume.")
            return total_saved

        if not data:
            break

        for post in data:
            text = f"{post.get('title', '')} {post.get('selftext', '')}"
            tickers = extract_tickers(text)
            for ticker in tickers:
                pending.append({
                    "date":         ts_to_date(post["created_utc"]),
                    "ticker":       ticker,
                    "post_id":      post["id"],
                    "title":        post.get("title", ""),
                    "score":        post.get("score", 0),
                    "num_comments": post.get("num_comments", 0),
                    "source":       "arctic_shift",
                })

        after = data[-1]["created_utc"] + 1

        # Checkpoint every CHECKPOINT_EVERY records
        if len(pending) >= CHECKPOINT_EVERY:
            append_records(pending)
            total_saved += len(pending)
            save_checkpoint(after)
            print(f"  [{ts_to_date(after)}] Saved {total_saved} mentions total (checkpointed)")
            pending = []
        else:
            print(f"  Fetched up to {ts_to_date(after)} — {total_saved + len(pending)} mentions so far")

        time.sleep(0.5)

    # Save any remaining records
    if pending:
        append_records(pending)
        total_saved += len(pending)

    # Clear checkpoint on successful completion
    if os.path.exists(CHECKPOINT):
        os.remove(CHECKPOINT)

    return total_saved

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    total = scrape_arctic_shift("2007-01-01", "2024-12-31")

    # Deduplicate the full output file
    df = pd.read_csv(OUT_PATH)
    before = len(df)
    df = df.drop_duplicates(subset=["post_id", "ticker"])
    df = df.sort_values("date").reset_index(drop=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"\nDone. {len(df)} unique ticker-mention rows in {OUT_PATH} (removed {before - len(df)} dupes)")
    print(df.head(10))
