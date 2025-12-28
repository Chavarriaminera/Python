# ============================================================
# Step 0 — Clean Tweets (RAM-safe, keep ALL columns)  v2
# - Streams RAW CSV in chunks
# - Drops retweets (counts them)
# - Keeps every original column (no column selection)
# - Skips malformed lines (counts them)
# - Adds time features from 'date': month, weekday/hour (UTC)
# - Optional: also adds weekday/hour in America/Phoenix
# - Writes ONE cleaned CSV + final summary
# - Progress pings; Hello Kitty every 3rd bucket
# ============================================================

from pathlib import Path
import pandas as pd

# ---------- Settings -----------------------------------------------------------
INPUT      = r"C:\Users\chava\OneDrive\Documents\Data Science Projects\Bitcoin\bitcoin_tweets.csv"
TEXT_COL   = "text"         # tweet text column name
DATE_COL   = "date"         # timestamp column name in your raw file
CHUNK      = 100_000        # rows per chunk
PROGRESS_EVERY = 200_000    # print after ~this many parsed rows (None/0 to disable)

# Derive features from DATE_COL (non-destructive; we do NOT overwrite 'date')
ADD_MONTH_FROM_DATE = True                  # add 'month' (YYYY-MM)
ADD_UTC_BINS        = True                  # add 'weekday_utc' (0=Mon..6=Sun), 'hour_utc' (0..23)

# Optional: also compute local time bins (America/Phoenix). Requires Python 3.9+ (zoneinfo)
ADD_LOCAL_BINS      = True
LOCAL_TZ_NAME       = "America/Phoenix"     # change if you want a different local tz

INPUT_PATH = Path(INPUT)
CLEAN_OUT  = INPUT_PATH.with_name("Step0_bitcoin_tweets_clean.csv")

ASCII_KITTY = r"""
⠀ ⢠⡾⠲⠶⣤⣀⣠⣤⣤⣤⡿⠛⠿⡴⠾⠛⢻⡆⠀⠀⠀
⠀⠀⠀⣼⠁⠀⠀⠀⠉⠁⠀⢀⣿⠐⡿⣿⠿⣶⣤⣤⣷⡀⠀⠀
⠀⠀⠀⢹⡶⠀⠀⠀⠀⠀⠀⠈⢯⣡⣿⣿⣀⣸⣿⣦⢓⡟⠀⠀
⠀⠀⢀⡿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠹⣍⣭⣾⠁⠀⠀
⠀⣀⣸⣇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣸⣷⣤⡀. ₊˚✧♡🌸I'm tired grandpa🌸♡✧˚₊‧
⠈⠉⠹⣏⡁⠀⢸⣿⠀⠀⠀⢀⡀⠀⠀⠀⣿⠆⠀⢀⣸⣇⣀⠀
⠀⠐⠋⢻⣅⣄⢀⣀⣀⡀⠀⠯⠽⠂⢀⣀⣀⡀⠀⣤⣿⠀⠉⠀
⠀⠀⠴⠛⠙⣳⠋⠉⠉⠙⣆⠀⠀⢰⡟⠉⠈⠙⢷⠟⠉⠙⠂⠀
⠀⠀⠀⠀⠀⢻⣄⣠⣤⣴⠟⠛⠛⠛⢧⣤⣤⣀⡾
"""

# ---------- Optional: timezone support ----------------------------------------
# We'll try zoneinfo (built-in on Python 3.9+). If unavailable, we silently skip local bins.
try:
    from zoneinfo import ZoneInfo
    HAVE_ZONEINFO = True
except Exception:
    HAVE_ZONEINFO = False
    if ADD_LOCAL_BINS:
        print("⚠️  zoneinfo not available — local time bins will be skipped.")

# ---------- Helpers ------------------------------------------------------------
def drop_retweets(df: pd.DataFrame, text_col: str) -> tuple[pd.DataFrame, int]:
    """
    Remove retweets and report how many were removed.
    - If 'is_retweet' exists, keep rows that look false-like (NOT RTs).
    - Else, fall back to 'RT' prefix in the text.
    """
    before = len(df)
    if "is_retweet" in df.columns:
        vals = df["is_retweet"].astype(str).str.strip().str.lower()
        false_like = {"false", "0", "no", "f", "nan", ""}  # treat empty/"nan" as not RT
        df = df[vals.isin(false_like)]
    elif text_col in df.columns:
        df = df[~df[text_col].astype(str).str.startswith("RT")]
    removed = before - len(df)
    return df, removed

# pandas will call this for malformed CSV lines when using engine="python"
bad_lines = 0
def _skip_and_count(_bad_line):
    """
    Count and skip a malformed row. Returning None drops the row.
    """
    global bad_lines
    bad_lines += 1
    return None

# ---------- Streaming clean ----------------------------------------------------
first_write = True
rows_in = 0
rows_out = 0
retweet_rows = 0
last_progress_bucket = -1
fixed_header = None  # track first chunk's header to keep CSV consistent

for chunk in pd.read_csv(
    INPUT,
    chunksize=CHUNK,
    engine="python",               # tolerant parser for messy quotes
    on_bad_lines=_skip_and_count,  # count & skip malformed lines
    encoding="utf-8",
    encoding_errors="replace",     # replace bad bytes, don't crash
    dtype=str                      # read ALL columns as string → preserves raw data
):
    rows_in += len(chunk)

    # (0) Time-derived columns from DATE_COL — non-destructive (we do NOT overwrite 'date')
    # We compute once here to avoid doing it again later in EDA.
    if DATE_COL in chunk.columns:
        dt = pd.to_datetime(chunk[DATE_COL], utc=True, errors="coerce")

        if ADD_MONTH_FROM_DATE:
            chunk["month"] = dt.dt.strftime("%Y-%m")  # 'YYYY-MM' or NaN (we'll fill later)

        if ADD_UTC_BINS:
            # 0=Monday..6=Sunday, 0..23 hours in UTC
            chunk["weekday_utc"] = dt.dt.weekday.astype("Int64")  # keeps <NA> where unparsable
            chunk["hour_utc"]    = dt.dt.hour.astype("Int64")

        if ADD_LOCAL_BINS and HAVE_ZONEINFO:
            try:
                # Convert UTC → local tz; unparsable rows remain NaT and yield <NA>
                dt_local = dt.dt.tz_convert(ZoneInfo(LOCAL_TZ_NAME))
                chunk["weekday_local"] = dt_local.dt.weekday.astype("Int64")
                chunk["hour_local"]    = dt_local.dt.hour.astype("Int64")
            except Exception:
                # If conversion fails (e.g., all NaT), add empty columns to keep header stable
                chunk["weekday_local"] = pd.Series([pd.NA]*len(chunk), dtype="Int64")
                chunk["hour_local"]    = pd.Series([pd.NA]*len(chunk), dtype="Int64")
        elif ADD_LOCAL_BINS:
            # zoneinfo not available; add empty columns so downstream code sees consistent header
            chunk["weekday_local"] = pd.Series([pd.NA]*len(chunk), dtype="Int64")
            chunk["hour_local"]    = pd.Series([pd.NA]*len(chunk), dtype="Int64")

        # Friendly defaults for missing values
        if "month" in chunk.columns:
            chunk["month"] = chunk["month"].fillna("unknown")
    else:
        # No DATE_COL; still add placeholders if features are requested, to keep schema stable
        if ADD_MONTH_FROM_DATE:
            chunk["month"] = "unknown"
        if ADD_UTC_BINS:
            chunk["weekday_utc"] = pd.Series([pd.NA]*len(chunk), dtype="Int64")
            chunk["hour_utc"]    = pd.Series([pd.NA]*len(chunk), dtype="Int64")
        if ADD_LOCAL_BINS:
            chunk["weekday_local"] = pd.Series([pd.NA]*len(chunk), dtype="Int64")
            chunk["hour_local"]    = pd.Series([pd.NA]*len(chunk), dtype="Int64")

    # (1) Drop retweets (and count them)
    chunk, removed = drop_retweets(chunk, TEXT_COL)
    retweet_rows += removed

    # (2) Drop rows with missing text (but keep all other columns as-is)
    if TEXT_COL in chunk.columns:
        chunk = chunk[chunk[TEXT_COL].notna()]

    # (3) Ensure consistent header across chunks
    if first_write:
        fixed_header = list(chunk.columns)
    else:
        if list(chunk.columns) != fixed_header:
            print("⚠️  Detected column mismatch in a later chunk. "
                  "Aligning to the first chunk's header to keep the file consistent.")
            for col in fixed_header:
                if col not in chunk.columns:
                    chunk[col] = ""
            chunk = chunk[fixed_header]

    # (4) Write chunk (append after the first)
    rows_out += len(chunk)
    if first_write:
        chunk.to_csv(CLEAN_OUT, index=False, encoding="utf-8")
        first_write = False
    else:
        chunk.to_csv(CLEAN_OUT, index=False, mode="a", header=False, encoding="utf-8")

    # (5) Progress pings
    if PROGRESS_EVERY:
        bucket = rows_in // PROGRESS_EVERY
        if bucket > last_progress_bucket:
            print(f"…processed ~{rows_in:,} rows so far", flush=True)
            if bucket % 3 == 0 and bucket != 0:
                print(ASCII_KITTY, flush=True)
            last_progress_bucket = bucket

# ---------- Final summary ------------------------------------------------------
print("✅ Step 0 complete.")
print(f"Total rows parsed:        {rows_in:,}")
print(f"Total rows kept (clean):  {rows_out:,}")
print(f"Retweets removed:         {retweet_rows:,}")
print(f"Malformed lines skipped:  {bad_lines:,}")
print(f"Clean file saved to:      {CLEAN_OUT}")
if ADD_MONTH_FROM_DATE: print("• Added column: month (YYYY-MM)")
if ADD_UTC_BINS:        print("• Added columns: weekday_utc (0=Mon..6=Sun), hour_utc (0..23)")
if ADD_LOCAL_BINS and HAVE_ZONEINFO:
    print(f"• Added columns: weekday_local, hour_local ({LOCAL_TZ_NAME})")
elif ADD_LOCAL_BINS:
    print("• Skipped local bins (zoneinfo unavailable)")
