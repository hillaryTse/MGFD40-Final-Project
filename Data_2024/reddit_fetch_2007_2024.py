"""
Reddit Penny Stock Mention Scraper
-----------------------------------
Strategy:
  - Arctic Shift API: fetch historical posts 2007-2024 (no auth required)

Ticker extraction:
  - $TICKER  : dollar-prefixed 1-5 uppercase chars — always kept, even if word
               appears in the false-positives list (e.g. $UP is valid)
  - STANDALONE: 2-5 uppercase chars not preceded/followed by a word char —
               filtered against FALSE_POSITIVES
  - Source   : post title + selftext only (no comments)

Output: CSV with columns [date, ticker, subject, content]
Checkpoints every 5000 records so crashes don't lose progress.
"""

import requests
import re
import time
import os
import pandas as pd
from datetime import datetime, timezone

SUBREDDIT        = "pennystocks"
OUT_PATH         = "reddit_data_2007_2024.csv"
CHECKPOINT       = "scrape_checkpoint.txt"
CHECKPOINT_EVERY = 5000

# Group 1: $TICKER  (dollar sign required)  — 1-5 uppercase letters
# Group 2: STANDALONE uppercase word        — 2-5 uppercase letters
TICKER_PATTERN = re.compile(r'\$([A-Z]{1,5})\b|(?<!\w)([A-Z]{2,5})(?!\w)')

FALSE_POSITIVES = {
    "I", "A", "THE", "AND", "OR", "NOT", "FOR", "IS", "ARE", "TO", "AI",
    "IN", "ON", "AT", "BE", "DO", "GO", "IT", "MY", "NO", "OF", "SO",
    "UP", "US", "WE", "DD", "OP", "IMO", "BUY", "SELL", "HOLD", "ETF",
    "CEO", "IPO", "SEC", "OTC", "NYSE", "NASDAQ", "EPS", "PE", "ATH",
    "YOY", "QOQ", "FOMO", "YOLO", "MOON", "GG", "NFT", "USD", "GDP",
    "EOD", "EOW", "PT", "SL", "TP", "RH", "WSB", "LOL", "TBH", "IMO",
}

def extract_tickers(text):
    """Extract ticker symbols from post text.

    Dollar-prefixed tickers ($TICKER) bypass the false-positives filter.
    Standalone uppercase words are filtered against FALSE_POSITIVES.
    """
    if not text:
        return []
    tickers = set()
    for m in TICKER_PATTERN.finditer(text):
        if m.group(1):
            # $TICKER — always valid regardless of false-positives list
            tickers.add(m.group(1).upper())
        elif m.group(2):
            # Standalone uppercase word — apply filter
            word = m.group(2).upper()
            if word not in FALSE_POSITIVES:
                tickers.add(word)
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
    df = pd.DataFrame(records)
    write_header = not os.path.exists(OUT_PATH)
    df.to_csv(OUT_PATH, mode="a", header=write_header, index=False)

# ---------------------------------------------------------------------------
# Arctic Shift API: historical posts
# ---------------------------------------------------------------------------
ARCTIC_BASE = "https://arctic-shift.photon-reddit.com/api"

def scrape_arctic_shift(start_date, end_date, batch_size=100):
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    end_ts   = int(datetime.strptime(end_date,   "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())

    saved = load_checkpoint()
    if saved and saved > start_ts:
        after = saved
        print(f"Resuming from checkpoint: {ts_to_date(after)}")
    else:
        after = start_ts
        print(f"Starting fresh from {start_date}")

    print(f"Fetching r/{SUBREDDIT} posts up to {end_date} (exclusive) via Arctic Shift...")

    pending = []
    total_saved = 0

    while after < end_ts:
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
            except requests.exceptions.HTTPError as e:
                wait = 2 ** attempt
                print(f"  HTTP error (attempt {attempt+1}/5): {e}. Retrying in {wait}s...")
                time.sleep(wait)
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
            subject = post.get("title", "").replace("\n", " ").replace("\r", " ")
            content = post.get("selftext", "").replace("\n", " ").replace("\r", " ")
            # Scan title + selftext for tickers (post content only, no comments)
            tickers = extract_tickers(f"{subject} {content}")
            for ticker in tickers:
                pending.append({
                    "date":    ts_to_date(post["created_utc"]),
                    "ticker":  ticker,
                    "subject": subject,
                    "content": content,
                })

        after = data[-1]["created_utc"] + 1

        if len(pending) >= CHECKPOINT_EVERY:
            append_records(pending)
            total_saved += len(pending)
            save_checkpoint(after)
            print(f"  [{ts_to_date(after)}] Saved {total_saved} mentions total (checkpointed)")
            pending = []
        else:
            print(f"  Fetched up to {ts_to_date(after)} — {total_saved + len(pending)} mentions so far")

        time.sleep(0.5)

    if pending:
        append_records(pending)
        total_saved += len(pending)

    if os.path.exists(CHECKPOINT):
        os.remove(CHECKPOINT)

    return total_saved

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # end_date="2025-01-01" is exclusive, so all of Dec 31 2024 is included
    total = scrape_arctic_shift("2007-01-01", "2025-01-01")

    df = pd.read_csv(OUT_PATH)
    before = len(df)
    df = df.drop_duplicates(subset=["subject", "ticker"])
    df = df.sort_values("date").reset_index(drop=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"\nDone. {len(df)} unique ticker-mention rows in {OUT_PATH} (removed {before - len(df)} dupes)")
    print(df.head(10))
