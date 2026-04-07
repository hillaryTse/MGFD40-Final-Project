"""
Reddit Penny Stock Mention Scraper
-----------------------------------
Strategy:
  - Arctic Shift API: fetch historical posts 2007-2024 (no auth required)

Ticker extraction:
  - $TICKER  : dollar-prefixed 1-5 uppercase chars — always kept, even if word
               appears in the false-positives list (e.g. $UP is valid)
  - STANDALONE: 2-5 uppercase chars not preceded/followed by a word char,
               filtered against FALSE_POSITIVES
  - Source   : post title + selftext only (no comments)

Output: CSV with columns [date, ticker, subject, content]
Checkpoints every 5000 records so crashes do not lose progress.
"""

import requests
import re
import time
import os
import pandas as pd
from datetime import datetime, timezone

SUBREDDIT        = "pennystocks"
OUT_PATH         = "reddit_data_2007_2024.csv"
CHECKPOINT       = "scrape_checkpoint.txt"    # stores last fetched timestamp for crash recovery
CHECKPOINT_EVERY = 5000                       # flush to CSV and save checkpoint every N mentions

# Group 1: $TICKER  (dollar sign required)  - 1-5 uppercase letters
# Group 2: STANDALONE uppercase word        - 2-5 uppercase letters not adjacent to other word chars
TICKER_PATTERN = re.compile(r'\$([A-Z]{1,5})\b|(?<!\w)([A-Z]{2,5})(?!\w)')

# Common Reddit terms and financial abbreviations that match the ticker pattern but are not tickers.
# Verified manually against CRSP ticker universe before adding.
# Note: $PREFIXED tickers (e.g. $BUY) bypass this filter and are always kept.
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
    Returns a deduplicated list of ticker strings.
    """
    if not text:
        return []
    tickers = set()
    for m in TICKER_PATTERN.finditer(text):
        if m.group(1):
            # $TICKER - always valid regardless of false-positives list
            tickers.add(m.group(1).upper())
        elif m.group(2):
            # Standalone uppercase word - apply filter
            word = m.group(2).upper()
            if word not in FALSE_POSITIVES:
                tickers.add(word)
    return list(tickers)


def ts_to_date(utc_ts):
    """Convert a UTC Unix timestamp to a YYYY-MM-DD date string."""
    return datetime.fromtimestamp(utc_ts, tz=timezone.utc).strftime("%Y-%m-%d")


def save_checkpoint(after_ts):
    """Write the last successfully fetched timestamp to disk for crash recovery."""
    with open(CHECKPOINT, "w") as f:
        f.write(str(after_ts))


def load_checkpoint():
    """Return the last checkpointed timestamp, or None if no checkpoint exists."""
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT) as f:
            val = f.read().strip()
            if val:
                return int(val)
    return None


def append_records(records):
    """Append a list of mention dicts to the output CSV, writing the header only on first write."""
    df = pd.DataFrame(records)
    write_header = not os.path.exists(OUT_PATH)
    df.to_csv(OUT_PATH, mode="a", header=write_header, index=False)


# ---------------------------------------------------------------------------
# Arctic Shift API: historical posts
# ---------------------------------------------------------------------------
ARCTIC_BASE = "https://arctic-shift.photon-reddit.com/api"


def scrape_arctic_shift(start_date, end_date, batch_size=100):
    """Fetch all r/pennystocks posts between start_date and end_date via Arctic Shift API.

    Paginates forward in time using the `after` timestamp parameter.
    Retries up to 5 times with exponential backoff on network errors.
    Saves checkpoint after every CHECKPOINT_EVERY mentions so the run can be resumed.
    """
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    end_ts   = int(datetime.strptime(end_date,   "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())

    # Resume from last checkpoint if available, otherwise start from the beginning
    saved = load_checkpoint()
    if saved and saved > start_ts:
        after = saved
        print(f"Resuming from checkpoint: {ts_to_date(after)}")
    else:
        after = start_ts
        print(f"Starting fresh from {start_date}")

    print(f"Fetching r/{SUBREDDIT} posts up to {end_date} (exclusive) via Arctic Shift...")

    pending     = []   # mentions not yet flushed to CSV
    total_saved = 0

    while after < end_ts:
        # Retry loop with exponential backoff on failure
        for attempt in range(5):
            try:
                params = {
                    "subreddit": SUBREDDIT,
                    "after":     after,
                    "before":    end_ts,
                    "limit":     batch_size,
                    "sort":      "asc",      # fetch chronologically oldest first
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
            # All 5 attempts failed - save progress and exit cleanly
            print("  Failed after 5 retries - saving progress and exiting.")
            if pending:
                append_records(pending)
                total_saved += len(pending)
            save_checkpoint(after)
            print(f"  Checkpoint saved at {ts_to_date(after)}. Re-run to resume.")
            return total_saved

        # Empty response means we have reached the end of available posts
        if not data:
            break

        for post in data:
            subject = post.get("title",    "").replace("\n", " ").replace("\r", " ")
            content = post.get("selftext", "").replace("\n", " ").replace("\r", " ")
            # Scan title + selftext for tickers; one row per ticker mentioned in the post
            tickers = extract_tickers(f"{subject} {content}")
            for ticker in tickers:
                pending.append({
                    "date":    ts_to_date(post["created_utc"]),
                    "ticker":  ticker,
                    "subject": subject,
                    "content": content,
                })

        # Advance pagination cursor to just after the last post's timestamp
        after = data[-1]["created_utc"] + 1

        # Flush to disk and checkpoint periodically to guard against crashes
        if len(pending) >= CHECKPOINT_EVERY:
            append_records(pending)
            total_saved += len(pending)
            save_checkpoint(after)
            print(f"  [{ts_to_date(after)}] Saved {total_saved} mentions total (checkpointed)")
            pending = []
        else:
            print(f"  Fetched up to {ts_to_date(after)} - {total_saved + len(pending)} mentions so far")

        time.sleep(0.5)   # rate-limit: be polite to the Arctic Shift API

    # Flush any remaining records not yet written
    if pending:
        append_records(pending)
        total_saved += len(pending)

    # Remove checkpoint file on clean completion
    if os.path.exists(CHECKPOINT):
        os.remove(CHECKPOINT)

    return total_saved


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # end_date is exclusive, so "2025-01-01" captures all posts through Dec 31 2024
    total = scrape_arctic_shift("2007-01-01", "2025-01-01")

    # Post-processing: remove duplicate ticker mentions from the same post
    df = pd.read_csv(OUT_PATH)
    before = len(df)
    df = df.drop_duplicates(subset=["subject", "ticker"])
    df = df.sort_values("date").reset_index(drop=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"\nDone. {len(df)} unique ticker-mention rows in {OUT_PATH} (removed {before - len(df)} dupes)")
    print(df.head(10))
