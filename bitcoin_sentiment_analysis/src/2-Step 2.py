
"""
Step 2 — Data Cleaning & Preparation (Chunked + Timed)
------------------------------------------------------

What this script does:
1) Streams Step 1 output in CHUNK_SIZE pieces (so we don't run out of memory).
2) Cleans tweet text while PRESERVING emoji meaning by converting emojis to words.
3) Tokenizes, removes stopwords, lemmatizes.
4) CALCULATES VADER COMPOUND SCORE.
5) COLLECTS ALL COMPOUND SCORES.
6) Writes each processed chunk (including score) directly to disk.

*** NEW STEP: SECOND PASS (After all chunks are processed) ***
7) Calculates 7-LEVEL GLOBAL QUANTILE THRESHOLDS from all collected scores.
8) Reads the temporary output, applies the 'quantile_sentiment' label, and overwrites the final output CSV.
9) Prints a comprehensive summary report, including the new quantile distribution.

... (rest of original description) ...
"""


# ------------------------------- IMPORTS --------------------------------------
import sys
import re
import time
import string
from pathlib import Path

import numpy as np
import pandas as pd
import nltk
import emoji
import matplotlib.pyplot as plt
from collections import Counter

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import wordpunct_tokenize # no punkt download required

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Make stdout line-buffered so print() shows immediately (esp. on Windows/Spyder)
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

# ------------------------------- NLTK SETUP -----------------------------------
# Only these are needed; we avoid punkt/punkt_tab by using wordpunct_tokenize.
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# ------------------------------- BANNER ART -----------------------------------
BITCOIN_ART = r"""
          ⣀⣤⣴⣶⣾⣿⣿⣿⣿⣷⣶⣦⣤⣀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⣠⣴⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣦⣄⠀⠀⠀⠀⠀
⠀⠀⠀⣠⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣄⠀⠀⠀
⠀⠀⣴⣿⣿⣿⣿⣿⣿⣿⠟⠿⠿⡿⠀⢰⣿⠁⢈⣿⣿⣿⣿⣿⣿⣿⣿⣦⠀⠀
⠀⣼⣿⣿⣿⣿⣿⣿⣿⣿⣤⣄⠀⠀⠀⠈⠉⠀⠸⠿⣿⣿⣿⣿⣿⣿⣿⣿⣧⠀
⢰⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡏⠀⠀⢠⣶⣶⣤⡀⠀⠈⢻⣿⣿⣿⣿⣿⣿⣿⡆
⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠃⠀⠀⠼⣿⣿⡿⠃⠀⠀⢸⣿⣿⣿⣿⣿⣿⣿⣷
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡟⠀⠀⢀⣀⣀⠀⠀⠀⠀⢴⣿⣿⣿⣿⣿⣿⣿⣿⣿
⢿⣿⣿⣿⣿⣿⣿⣿⢿⣿⠁⠀⠀⣼⣿⣿⣿⣦⠀⠀⠈⢻⣿⣿⣿⣿⣿⣿⣿⡿
⠸⣿⣿⣿⣿⣿⣿⣏⠀⠀⠀⠀⠀⠛⠛⠿⠟⠋⠀⠀⠀⣾⣿⣿⣿⣿⣿⣿⣿⠇
⠀⢻⣿⣿⣿⣿⣿⣿⣿⣿⠇⠀⣤⡄⠀⣀⣀⣀⣀⣠⣾⣿⣿⣿⣿⣿⣿⣿⡟⠀
⠀⠀⠻⣿⣿⣿⣿⣿⣿⣿⣄⣰⣿⠁⢀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠟⠀⠀
⠀⠀⠀⠙⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠋⠀⠀⠀
⠀⠀⠀⠀⠀⠙⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠟⠋⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠉⠛⠻⠿⢿⣿⣿⣿⣿⡿⠿⠟⠛⠉⠀⠀⠀⠀⠀⠀⠀⠀
"""

# ------------------------------- PATHS / CONFIG -------------------------------
INPUT_FILE  = Path("C:/Users/chava/OneDrive/Documents/Data Science Projects/Bitcoin/step1_sentiment.csv")
OUTPUT_FILE = Path("C:/Users/chava/OneDrive/Documents/Data Science Projects/Bitcoin/Step2bitcoin_tweets_clean.csv")
TEMP_OUTPUT_FILE = Path(str(OUTPUT_FILE).replace(".csv", "_temp.csv")) # Intermediate file for first pass

CHUNK_SIZE = 50_000      # smaller -> first progress line sooner
SPECIAL_PROGRESS_EVERY = 3 # milestone banner every N chunks (0 = off)
DO_FAST_ROW_COUNT = True     # show ETA by fast counting in big chunks

# Sentiment config
SENTIMENT_COL = "sentiment"   # change if your column name differs
SENTIMENT_ALIASES = {
    "pos": "positive", "positive": "positive", "+": "positive",
    "neu": "neutral",  "neutral":  "neutral",  "0": "neutral",
    "neg": "negative", "negative": "negative", "-": "negative",
}

# Console colors for pretty summaries
RESET = "\033[0m"
PURPLE = "\033[95m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
RED    = "\033[91m"


# ------------------------------ SENTIMENT SETUP -------------------------------
analyzer=SentimentIntensityAnalyzer()
ALL_COMPOUND_SCORES: list[float] = [] # Global list to collect all scores for global quantile calculation

# ------------------------------ PRINT HELPERS --------------------------------
# (print_milestone and print_sentiment_summary remain the same)

def print_milestone(idx: int) -> None:
    print(f"\n🎀 Milestone: {idx} chunks processed — keep going!\n{BITCOIN_ART}", flush=True)

def print_sentiment_summary(counts: dict[str, int]) -> None:
    total = sum(counts.values())
    print("")  # spacer
    if total == 0:
        print(f"{PURPLE}⎧ Sentiment Distribution ⎫{RESET}\n  (no sentiment column/values found)", flush=True)
        return

    def pct(n): 
        return f"{(100.0 * n / total):.2f}%"

    pos = counts.get("positive", 0)
    neu = counts.get("neutral", 0)
    neg = counts.get("negative", 0)

    line = (
        f"{PURPLE}⎧ Sentiment Distribution ⎫{RESET}\n"
        f"  {GREEN}positive: {pos:,} ({pct(pos)}){RESET}  |  "
        f"{CYAN}neutral: {neu:,} ({pct(neu)}){RESET}  |  "
        f"{RED}negative: {neg:,} ({pct(neg)}){RESET}"
    )
    print(line, flush=True)

# --------------------------- CLEANING HELPERS ---------------------------------
# (demojize_to_words, clean_tweet, tokenize_safe, and preprocess_text remain the same)

def demojize_to_words(text: str) -> str:
    """Convert emojis to readable tokens without colons."""
    t = emoji.demojize(text, language="en")        # "... :rocket: ..."
    t = re.sub(r":([a-zA-Z0-9_]+):", r" \1 ", t)       # -> " rocket "
    return t

def clean_tweet(text: str) -> str:
    """
    Normalize tweet text:
      - lowercase
      - emojis -> words
      - remove URLs and @mentions
      - keep hashtag word but drop '#'
      - strip punctuation & numbers
      - collapse whitespace
    """
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = demojize_to_words(text)
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = text.replace("#", " ")
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize_safe(text: str) -> list[str]:
    """
    Primary: wordpunct_tokenize (no downloads). Filter to alphabetic tokens.
    Fallbacks try punkt/punkt_tab only if ever needed.
    """
    tokens = [t for t in wordpunct_tokenize(text) if t.isalpha()]
    if tokens:
        return tokens
    # ultra-rare fallback
    try:
        nltk.download("punkt", quiet=True)
        try:
            nltk.download("punkt_tab", quiet=True)
        except Exception:
            pass
        from nltk.tokenize import word_tokenize
        return [t for t in word_tokenize(text) if t.isalpha()]
    except Exception:
        return re.findall(r"[A-Za-z]+", text)

def preprocess_text(text: str, lemmatizer: WordNetLemmatizer, stop_words: set) -> str:
    """Tokenize -> remove stopwords -> lemmatize -> rejoin."""
    if not text:
        return ""
    tokens = tokenize_safe(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)


# ------------------------------ QUANTILE LABELING ------------------------------

def compute_quantile_bins(scores: list[float], num_bins: int = 7) -> list[float]:
    """Calculates the quantile thresholds (including min/max)."""
    if not scores:
        return []
    # np.linspace(0, 1, 8) gives [0.0, 0.1428..., 0.2857..., ..., 1.0] for 7 bins
    return np.quantile(scores, q=np.linspace(0, 1, num_bins + 1)).tolist()

def label_from_quantiles(score: float, thresholds: list[float]) -> str:
    """Assigns the 7-level label based on the calculated global thresholds."""
    if not thresholds:
        return "neutral"
        
    labels = [
        "extremely negative", "very negative", "negative",
        "neutral", "positive", "very positive", "extremely positive"
    ]
    
    # Ensure score is within the range of thresholds (min <= score <= max)
    score = np.clip(score, thresholds[0], thresholds[-1])

    for i in range(len(thresholds) - 1):
        # We check for: thresholds[i] <= score < thresholds[i+1]
        # The last bin includes the max score: thresholds[6] <= score <= thresholds[7]
        if i == len(thresholds) - 2:
            if thresholds[i] <= score <= thresholds[i + 1]:
                return labels[i]
        elif thresholds[i] <= score < thresholds[i + 1]:
            return labels[i]
            
    return labels[-1] # Should be caught by the above, but as a fallback.
    
# ------------------------------ EMOJI HELPER ----------------------------------
def extract_emojis(s: str) -> list[str]:
    """
    Get distinct emojis from a string without requiring a specific emoji version.
    """
    try:
        # Prefer library helper if available
        from emoji import distinct_emoji_list
        return distinct_emoji_list(s)
    except Exception:
        # Fallback: use emoji.EMOJI_DATA keys to detect single-char emojis
        return [ch for ch in s if ch in getattr(emoji, "EMOJI_DATA", {})]

# ------------------------------ UTIL FUNCTIONS --------------------------------
def estimate_total_rows_fast(csv_path: Path, text_col_exists: bool) -> int | None:
    """FAST total-row estimate using big chunks to provide ETA."""
    try:
        print("⏳ Estimating total rows...", flush=True)
        usecols = ["text"] if text_col_exists else None
        total = 0
        for cc in pd.read_csv(csv_path, usecols=usecols, chunksize=200_000):
            total += len(cc)
        print(f"    Estimated total rows: {total:,}", flush=True)
        return total
    except Exception as e:
        print(f"⚠️ Row estimate skipped: {e}", flush=True)
        return None

# --------------------------------- MAIN ---------------------------------------
def main():
    global ALL_COMPOUND_SCORES
    print("Script Started", flush=True)
    print("🔧 Checking files…", flush=True)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    # Clean up output files (using TEMP_OUTPUT_FILE for the first pass)
    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()
    if TEMP_OUTPUT_FILE.exists():
        TEMP_OUTPUT_FILE.unlink()

    print("🧠 Preparing NLP objects…", flush=True)
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    print("📥 Reading preview…", flush=True)
    preview = pd.read_csv(INPUT_FILE, nrows=1)
    has_text_col = "text" in preview.columns

    total_rows = estimate_total_rows_fast(INPUT_FILE, has_text_col) if DO_FAST_ROW_COUNT else None

    print("📂 Starting Step 2 (chunked: Pass 1/2)…", flush=True)
    print(f"    Chunk size: {CHUNK_SIZE:,}", flush=True)
    print(f"    Intermediate Output → {TEMP_OUTPUT_FILE}", flush=True)
    print("🚦 Entering chunk loop…", flush=True)

    # -------------------- Running totals for stats --------------------
    sent_counts = {"positive": 0, "neutral": 0, "negative": 0}
    tweet_lengths: list[int] = []
    word_counts: list[int] = []
    word_counter: Counter = Counter()
    hashtags: Counter = Counter()
    emoji_counts: Counter = Counter()

    rows_processed = 0
    chunk_index = 0
    start_all = time.perf_counter()

    # ========================== PASS 1: CLEANING & SCORE COLLECTION ==========================
    for chunk in pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE):
        chunk_index += 1
        t0 = time.perf_counter()

        if "text" not in chunk.columns:
            raise KeyError("Input data must have a 'text' column.")

        # ---------------- CLEAN ----------------
        chunk["clean_text"] = chunk["text"].apply(clean_tweet)

        # TOKENIZE + LEMMATIZE
        chunk["clean_text"] = chunk["clean_text"].apply(
            lambda x: preprocess_text(x, lemmatizer, stop_words)
        )
        
        # ---------------- SENTIMENT SCORE ----------------
        # Calculate VADER compound score on the *original* text (VADER best practice)
        chunk["compound_score"] = chunk["text"].apply(
            lambda x: analyzer.polarity_scores(str(x))["compound"]
        )
        
        # Collect scores for global quantile calculation later
        ALL_COMPOUND_SCORES.extend(chunk["compound_score"].dropna().tolist())

        # ---------------- WRITE (to temp file) ----------------
        write_header = not TEMP_OUTPUT_FILE.exists()
        # Only saving necessary columns (including the new score) to save time/space
        cols_to_save = list(chunk.columns)
        if "clean_text" not in cols_to_save: cols_to_save.append("clean_text")
        if "compound_score" not in cols_to_save: cols_to_save.append("compound_score")
        
        chunk[cols_to_save].to_csv(TEMP_OUTPUT_FILE, index=False, mode="a", header=write_header)

        # ---------------- STATS (streaming) ----------------
        # ... (rest of the stats collection is the same)
        if SENTIMENT_COL in chunk.columns:
            s = (
                chunk[SENTIMENT_COL]
                .astype(str).str.strip().str.lower()
                .map(lambda x: SENTIMENT_ALIASES.get(x, x))
            )
            sent_counts["positive"] += int((s == "positive").sum())
            sent_counts["neutral"]  += int((s == "neutral").sum())
            sent_counts["negative"] += int((s == "negative").sum())

        lens = chunk["clean_text"].str.len().dropna().tolist()
        tweet_lengths.extend(lens)
        wcs = chunk["clean_text"].str.split().str.len().dropna().tolist()
        word_counts.extend(wcs)

        for text in chunk["clean_text"].dropna():
            word_counter.update(text.split())

        for raw in chunk["text"].dropna():
            for tok in str(raw).split():
                if tok.startswith("#"):
                    hashtags[tok.lower()] += 1
            for e in extract_emojis(str(raw)):
                emoji_counts[e] += 1

        # ---------------- PROGRESS ----------------
        rows_in_chunk = len(chunk)
        rows_processed += rows_in_chunk
        dt = time.perf_counter() - t0
        rate = rows_in_chunk / dt if dt > 0 else float("inf")

        if total_rows:
            remaining = max(total_rows - rows_processed, 0)
            eta_sec = remaining / rate if rate > 0 else 0
            eta_min = eta_sec / 60.0
            pct_done = 100.0 * rows_processed / total_rows
            print(
                f"✅ Chunk {chunk_index:>3} | {rows_in_chunk:,} rows in {dt:,.2f}s | "
                f"{rate:,.0f} rows/s | {rows_processed:,}/{total_rows:,} ({pct_done:,.2f}%) | "
                f"ETA ~ {eta_min:,.1f} min",
                flush=True,
            )
        else:
            print(
                f"✅ Chunk {chunk_index:>3} | {rows_in_chunk:,} rows in {dt:,.2f}s | "
                f"{rate:,.0f} rows/s | total processed: {rows_processed:,}",
                flush=True,
            )

        if SPECIAL_PROGRESS_EVERY and (chunk_index % SPECIAL_PROGRESS_EVERY == 0):
            print_milestone(chunk_index)


    # ========================== PASS 2: QUANTILE LABELING & FINAL SAVE ==========================
    total_time = time.perf_counter() - start_all
    print("\n-------------------------------------------------------", flush=True)
    print("📊 Pass 1 complete. Calculating global quantile thresholds...", flush=True)
    
    if not ALL_COMPOUND_SCORES:
        print("🛑 Error: No compound scores collected. Skipping quantile labeling.", flush=True)
        return

    # 1. Calculate Global Thresholds
    global_thresholds = compute_quantile_bins(ALL_COMPOUND_SCORES, num_bins=7)
    
    print(f"✅ Global 7-Quantile Thresholds calculated (min to max):", flush=True)
    print(f"   {['{:.3f}'.format(t) for t in global_thresholds]}", flush=True)
    print("...Starting Pass 2: Labeling and Final Save.", flush=True)

    # 2. Read temp file in chunks, apply label, and save to final file
    quantile_counts = Counter()
    final_chunk_index = 0
    
    for chunk in pd.read_csv(TEMP_OUTPUT_FILE, chunksize=CHUNK_SIZE):
        final_chunk_index += 1
        
        # Apply the final quantile label using the global thresholds
        chunk["quantile_sentiment"] = chunk["compound_score"].apply(
            lambda x: label_from_quantiles(x, global_thresholds)
        )
        
        quantile_counts.update(chunk["quantile_sentiment"].dropna())
        
        # Save to final output file
        write_header = not OUTPUT_FILE.exists()
        chunk.to_csv(OUTPUT_FILE, index=False, mode="a", header=write_header)
        
        print(f"   | Pass 2: Wrote chunk {final_chunk_index} to final file.", flush=True)

    # 3. Clean up temporary file
    TEMP_OUTPUT_FILE.unlink()
    
    # 4. Final summary and timing
    total_time_final = time.perf_counter() - start_all
    mins_final = total_time_final / 60.0
    print("\n-------------------------------------------------------", flush=True)
    print("💾 All data written.", flush=True)
    print(f"🏁 Step 2 complete (Total Two Passes) in {total_time_final:,.2f}s (~{mins_final:,.1f} min).", flush=True)
    print(f"➡️  Final Output file: {OUTPUT_FILE}", flush=True)


    # =================== FINAL SUMMARY REPORT ===================
    print_sentiment_summary(sent_counts)

    print(f"\n{PURPLE}⎧ Quantile Sentiment Distribution (7-Level) ⎫{RESET}")
    total_q = sum(quantile_counts.values())

    def pct_q(n): 
        return f"{(100.0 * n / total_q):.2f}%"

    for label in [
        "extremely negative", "very negative", "negative",
        "neutral", "positive", "very positive", "extremely positive"
    ]:
        count = quantile_counts.get(label, 0)
        print(f"  {label:<18}: {count:,} ({pct_q(count)})", flush=True)
        
    # Plotting the distribution of scores (optional but useful)
    plt.figure(figsize=(10, 6))
    plt.hist(ALL_COMPOUND_SCORES, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    for i, t in enumerate(global_thresholds):
        color = 'red' if i in [0, 7] else 'orange'
        plt.axvline(x=t, color=color, linestyle='--', linewidth=1)
    plt.title("VADER Compound Score Distribution with 7-Quantile Thresholds")
    plt.xlabel("Compound Score")
    plt.ylabel("Frequency")
    plt.show()

    print(f"\n{PURPLE}⎧ Descriptive Statistics ⎫{RESET}")
    if tweet_lengths:
        print(f"  Length (chars)   — avg: {np.mean(tweet_lengths):.1f} | "
              f"median: {np.median(tweet_lengths):.1f} | "
              f"min: {int(np.min(tweet_lengths))} | max: {int(np.max(tweet_lengths))}", flush=True)
    if word_counts:
        print(f"  Word count       — avg: {np.mean(word_counts):.2f} | "
              f"median: {np.median(word_counts):.1f}", flush=True)

    # Top words
    print(f"\n  🔝 Top 20 words:")
    for w, c in word_counter.most_common(20):
        print(f"    {w}: {c:,}", flush=True)

    # Top hashtags
    if hashtags:
        print(f"\n  #️⃣ Top 10 hashtags:")
        for h, c in hashtags.most_common(10):
            print(f"    {h}: {c:,}", flush=True)

    # Top emojis
    if emoji_counts:
        print(f"\n  😊 Top 10 emojis:")
        for e, c in emoji_counts.most_common(10):
            print(f"    {e}: {c:,}", flush=True)


# --------------------------------- ENTRYPOINT ---------------------------------
if __name__ == "__main__":
    main()