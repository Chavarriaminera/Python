"""
Step 1 — First-Pass Sentiment Labeling
--------------------------------------

This script:
1. Reads raw Bitcoin tweets in chunks (so memory doesn’t get overloaded).
2. Removes retweets (if an `is_retweet` column exists).
3. Uses VADER sentiment analysis to get a sentiment score for each tweet.
4. Creates a new column called `sent_label` with values: positive, neutral, or negative.
5. Saves the processed data into a new CSV file.

Input:
- A CSV file containing at least a "text" column (tweets).
- Optional "is_retweet" column to filter out retweets.

Output:
- A new CSV file with sentiment scores and labels.

Author: Cin 
"""
# --- SETUP --------------------------------------------------------------------
import os, sys, time, math, shutil
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from pathlib import Path  # Pathlib lets us handle filesystem paths robustly across OSes

# --- Prettify console output ----------------------------------------
# If colorama isn't installed, we fall back to plain text (no colors).
try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
except Exception:  # colorama not available
    class _NoColor:
        def __getattr__(self, _): return ""
    Fore = Style = _NoColor()

# Small helpers for consistent “pretty” prints
def info(msg):    print(f"{Fore.CYAN}ℹ️  {msg}{Style.RESET_ALL}")
def good(msg):    print(f"{Fore.GREEN}✓ {msg}{Style.RESET_ALL}")
def warn(msg):    print(f"{Fore.YELLOW}⚠️  {msg}{Style.RESET_ALL}")
def done(msg):    print(f"{Fore.GREEN}✅ {msg}{Style.RESET_ALL}")
def emph(label):  return f"{Fore.YELLOW}{label}{Style.RESET_ALL}"

# --- VADER setup ---
# download the Vader sentiment lexicon. We are using this one because it contains words, slang, emojis,
# and their sentiment scores. This only downloads once; subsequent runs should be fast.
nltk.download("vader_lexicon", quiet=True)

# create a vader sentiment analyzer. This lets us turn raw tweet text into usable sentiment numbers.
sia = SentimentIntensityAnalyzer()

# --- Paths (edit if needed) ---
INPUT_FILE = Path("C:/Users/chava/OneDrive/Documents/Data Science Projects/Bitcoin/Step0_bitcoin_tweets_clean.csv")
OUTPUT_FILE = Path("C:/Users/chava/OneDrive/Documents/Data Science Projects/Bitcoin/step1_sentiment.csv")

CHUNKSIZE = 100_000  # number of rows per chunk; tune this to fit available memory

# ------------------------------ PROCESSING FUNCTION ----------------------------

def get_sentiment_label(compound_score: float) -> str:
    """
    Convert VADER compound score into a weak label:
    - >= 0.05 → positive
    - <= -0.05 → negative
    - otherwise → neutral
    """
    if compound_score >= 0.05:
        return "positive"
    elif compound_score <= -0.05:
        return "negative"
    else:
        return "neutral"

# --- INPUT SANITY CHECK -------------------------------------------------------
if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Input file not found:\n{INPUT_FILE}")

# Ensure output folder exists (mkdir on parent of the OUTPUT_FILE path).
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# --- MAIN LOOP ----------------------------------------------------------------
# We only need a couple of columns; this reduces IO and dtype headaches.
# If your file definitely has these columns, set them explicitly; otherwise leave as None to read all.
##usecols = ["text", "is_retweet"]

results = []        # will collect processed chunks (reduced columns) to concat at the end
processed_rows = 0  # running total of rows processed (after RT drop + NaN drop)
chunk_idx = 0       # chunk counter
t0 = time.time()

print(f"{Fore.MAGENTA}┌──────────────────────────────────────────────┐{Style.RESET_ALL}")
print(f"{Fore.MAGENTA}│{Style.RESET_ALL}  {Fore.CYAN}▶️  Step 1 — Sentiment labeling started...{Style.RESET_ALL}   {Fore.MAGENTA}│{Style.RESET_ALL}")
print(f"{Fore.MAGENTA}└──────────────────────────────────────────────┘{Style.RESET_ALL}")
print(f"   {emph('Reading:')} {INPUT_FILE}")
print(f"   {emph('Writing:')} {OUTPUT_FILE}")
print(f"   {emph('Chunk size:')} {CHUNKSIZE:,} rows\n")

# read the dataset in chunks so the entire file does not load into memory at once.
# Use encoding compatibility that works across pandas versions.
read_csv_kwargs = dict(
    chunksize=CHUNKSIZE,
    low_memory=False,   # avoids mixed-type inference warnings
    encoding="utf-8"
    #usecols=usecols
)
# Prefer pandas>=2 arg; fallback gracefully for older versions.
try:
    read_csv_kwargs["encoding_errors"] = "ignore"
except Exception:
    read_csv_kwargs["errors"] = "ignore"

for chunk in pd.read_csv(INPUT_FILE, **read_csv_kwargs):
    chunk_idx += 1
    tick = time.time()

    # 1) Drop retweets if the column exists (normalize typical truthy strings/numbers).
    if "is_retweet" in chunk.columns:
        rt_raw = chunk["is_retweet"].astype(str).str.strip().str.lower()
        is_rt = rt_raw.isin({"true", "1", "t", "yes", "y"})
        chunk = chunk.loc[~is_rt].copy()

    # 2) Drop rows with no tweet text.
    if "text" not in chunk.columns:
        warn(f"Chunk {chunk_idx:>3}: missing 'text' column — skipped")
        continue
    chunk = chunk.dropna(subset=["text"])
    if chunk.empty:
        warn(f"Chunk {chunk_idx:>3}: empty after filtering — skipped")
        continue

    # 3) Apply VADER to each tweet text (compound ∈ [-1, 1]).
    # Using str(t) to be safe in case of non-string values slipping through.
    chunk["compound"] = chunk["text"].apply(lambda t: sia.polarity_scores(str(t))["compound"])

    # 4) Turn the compound score into a simple sentiment label
    #    NOTE: keeping column name as 'sent_label' to match the docstring.
    chunk["sent_label"] = chunk["compound"].apply(get_sentiment_label)
    
    ## Ensure consistent header
    if chunk_idx == 1:
        fixed_header = list(chunk.columns)
    else:
        for col in fixed_header:
            if col not in chunk.columns:
                chunk[col] = ""
        chunk = chunk[fixed_header]

    # 5) Keep only the important columns for Step 1 output
    results.append(chunk)
    processed_rows += len(chunk)
    tock = time.time()
    good(f"Chunk {chunk_idx:>3} processed "
         f"({Fore.MAGENTA}{processed_rows:,}{Style.RESET_ALL} rows total, "
         f"{(tock - tick):.2f}s)")

# --- WRITE OUTPUT + SUMMARY ----------------------------------------------------
if results:
    # merge all partial DataFrames into one
    final = pd.concat(results, ignore_index=True)

    # save the combined dataset with sentiment scores + labels
    final.to_csv(OUTPUT_FILE, index=False)
    elapsed = time.time() - t0
    done("Step 1 complete.")
    info(f"Saved → {OUTPUT_FILE}")
    info(f"Total time: {elapsed:.2f}s\n")

    # --- Sentiment distribution summary ---
    # Total number of processed tweets
    total = len(final)

    # Count how many of each label (positive/neutral/negative)
    vc = final["sent_label"].value_counts(dropna=False)

    # Pull counts safely (default 0 if missing)
    pos = int(vc.get("positive", 0))
    neu = int(vc.get("neutral", 0))
    neg = int(vc.get("negative", 0))

    # Calculate percentages relative to total
    pos_pct = (pos / total) * 100 if total else 0.0
    neu_pct = (neu / total) * 100 if total else 0.0
    neg_pct = (neg / total) * 100 if total else 0.0

    # Print a nice formatted summary
    print(f"{Fore.MAGENTA}┌────────── Sentiment Distribution ──────────┐{Style.RESET_ALL}")
    print(
        "   "
        f"{Fore.GREEN}positive: {pos:,} ({pos_pct:.2f}%){Style.RESET_ALL} | "
        f"{Fore.CYAN}neutral: {neu:,} ({neu_pct:.2f}%){Style.RESET_ALL} | "
        f"{Fore.RED}negative: {neg:,} ({neg_pct:.2f}%){Style.RESET_ALL}"
    )
    print(f"{Fore.MAGENTA}└──────────────────────────────────────────────┘{Style.RESET_ALL}")
else:
    # If no chunks were processed (bad file or no 'text' column or all filtered as RTs)
    warn("No rows were processed. Check that your CSV has a 'text' column and non-RT rows.")
