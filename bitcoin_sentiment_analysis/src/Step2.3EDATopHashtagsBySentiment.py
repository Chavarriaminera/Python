"""
Script: eda_top_hashtags_overall_and_by_sentiment_vader.py

Purpose:
Top N hashtags overall + top N hashtags by sentiment bucket (negative / neutral / positive)
using VADER compound scores. Uses a Bitcoin-themed dark palette, gradient bars, and log-scaled
x-axis to preserve visibility across large frequency ranges.

Notes:
- We use the dataset's existing 'hashtags' column (already extracted).
- We do NOT extract hashtags from 'clean_text' because preprocessing often removes '#'.
- Counts represent hashtag OCCURRENCES (one tweet can contribute multiple hashtags).
"""

# =============================================================================
# SECTION 0 — Imports
# =============================================================================

import ast  # safely parses strings like "['btc', 'crypto']" into Python lists
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path


# =============================================================================
# SECTION 1 — File Paths + Analysis Settings
# =============================================================================

INPUT_FILE = Path("C:/Users/chava/OneDrive/Documents/Data Science Projects/Bitcoin/Step2bitcoin_tweets_clean.csv")

TOP_N = 20
VADER_THRESHOLD = 0.05  # VADER neutral zone: [-0.05, +0.05]

# Optional: remove dominant tags so the chart shows more variety.
REMOVE_DOMINANT_TAGS = False
STOP_HASHTAGS = {"#bitcoin", "#btc", "#crypto"}


# =============================================================================
# SECTION 2 — Bitcoin Theme (Colors + Style)
# =============================================================================

sns.set_style("whitegrid")

BITCOIN_ORANGE = "#F7931A"
DARK_BACKGROUND = "#101820"
LIGHT_TEXT = "#E0E0E0"
GRID_COLOR = "#4A648C"

LIGHTER_ORANGE = "#FFC064"
DARKER_ORANGE = "#6E3300"

# Orange gradient colormap for bar fills (light -> dark)
cmap_bars = mcolors.LinearSegmentedColormap.from_list(
    "BitcoinBarGradient",
    [LIGHTER_ORANGE, BITCOIN_ORANGE, DARKER_ORANGE],
    N=256
)


# =============================================================================
# SECTION 3 — Load Data + Validate Columns
# =============================================================================

print(f"Loading: {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE)
print(f"✅ Rows loaded: {len(df):,}")

required_cols = {"compound_score", "hashtags"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"❌ Missing required columns: {missing}")

bitcoin = df[["compound_score", "hashtags"]].copy()
bitcoin = bitcoin.dropna(subset=["compound_score"])


# =============================================================================
# SECTION 4 — Create Sentiment Buckets (negative / neutral / positive)
# =============================================================================

def vader_bucket(score: float, threshold: float = VADER_THRESHOLD) -> str:
    """
    Converts continuous VADER compound score into discrete buckets.
    """
    if score < -threshold:
        return "negative"
    if score > threshold:
        return "positive"
    return "neutral"

bitcoin["sentiment_bucket"] = bitcoin["compound_score"].apply(vader_bucket)

print("\nSentiment bucket counts:")
print(bitcoin["sentiment_bucket"].value_counts(dropna=False))


# =============================================================================
# SECTION 5 — Parse + Normalize Hashtags from the 'hashtags' column
# =============================================================================

def parse_hashtags(x):
    """
    Handles cases where 'hashtags' is:
    - NaN
    - empty string
    - a string representation of a list: "['btc','crypto']"
    - already a list

    Returns:
    - Always returns a list of normalized hashtag strings with a SINGLE leading '#'
      e.g., ["#btc", "#crypto"]
    """
    if pd.isna(x):
        return []

    # If already a list, use it
    if isinstance(x, list):
        tags = x

    # If it's a string, try to parse it as a Python list
    elif isinstance(x, str):
        x = x.strip()
        if x == "":
            return []
        try:
            parsed = ast.literal_eval(x)  # safe parsing
            tags = parsed if isinstance(parsed, list) else []
        except Exception:
            return []
    else:
        return []

    # Normalize:
    # 1) lowercase/strip
    # 2) remove any existing leading '#'
    # 3) add exactly one '#'
    tags = [str(t).strip().lower() for t in tags if str(t).strip() != ""]
    tags = [t[1:] if t.startswith("#") else t for t in tags]  # remove leading '#'
    tags = [f"#{t}" for t in tags]

    return tags

bitcoin["hashtags_list"] = bitcoin["hashtags"].apply(parse_hashtags)

total_tags = bitcoin["hashtags_list"].apply(len).sum()
print(f"\n✅ Total hashtags extracted: {total_tags:,}")
print("DEBUG sample hashtags_list:", bitcoin["hashtags_list"].head(5).tolist())


# =============================================================================
# SECTION 6 — Explode to Long Format (one hashtag per row)
# =============================================================================

tags_long = bitcoin[["sentiment_bucket", "hashtags_list"]].explode("hashtags_list")
tags_long = tags_long.dropna(subset=["hashtags_list"])
tags_long = tags_long.rename(columns={"hashtags_list": "hashtag"})

print(f"\n✅ Total hashtag tokens after explode: {len(tags_long):,}")

if REMOVE_DOMINANT_TAGS:
    before = len(tags_long)
    tags_long = tags_long[~tags_long["hashtag"].isin(STOP_HASHTAGS)]
    after = len(tags_long)
    print(f"Removed dominant tags {STOP_HASHTAGS}. Rows: {before:,} -> {after:,}")


# =============================================================================
# SECTION 7 — Sanity Checks (optional but recommended)
# =============================================================================

print("\n--- Sanity checks ---")
print("Rows in tags_long:", len(tags_long))
print("Unique hashtags:", tags_long["hashtag"].nunique())

double_hash = tags_long["hashtag"].str.startswith("##", na=False).sum()
print("Double-hash tags (should be 0):", double_hash)

print("\nTop 10 hashtags overall (raw counts):")
print(tags_long["hashtag"].value_counts().head(10))


# =============================================================================
# SECTION 8 — Top N Hashtags Helpers (Count + Share %)
# =============================================================================

def top_hashtags_for_bucket(df_tags: pd.DataFrame, bucket: str, top_n: int = TOP_N) -> pd.DataFrame:
    """
    Top hashtags for one sentiment bucket.
    Share % is computed within that bucket (share of hashtag occurrences in that bucket).
    """
    subset = df_tags[df_tags["sentiment_bucket"] == bucket]
    if subset.empty:
        return pd.DataFrame(columns=["hashtag", "count", "share_pct"])

    counts = subset["hashtag"].value_counts()
    share_pct = (counts / counts.sum()) * 100

    out = pd.DataFrame({
        "hashtag": counts.index,
        "count": counts.values,
        "share_pct": share_pct.values
    }).head(top_n)

    return out

def top_hashtags_overall(df_tags: pd.DataFrame, top_n: int = TOP_N) -> pd.DataFrame:
    """
    Top hashtags overall (all sentiment buckets combined).
    Share % is computed across all hashtag occurrences.
    """
    counts = df_tags["hashtag"].value_counts()
    share_pct = (counts / counts.sum()) * 100

    out = pd.DataFrame({
        "hashtag": counts.index,
        "count": counts.values,
        "share_pct": share_pct.values
    }).head(top_n)

    return out


top_all = top_hashtags_overall(tags_long, TOP_N)
top_neg = top_hashtags_for_bucket(tags_long, "negative", TOP_N)
top_neu = top_hashtags_for_bucket(tags_long, "neutral", TOP_N)
top_pos = top_hashtags_for_bucket(tags_long, "positive", TOP_N)


# =============================================================================
# SECTION 9 — Plot Helpers (gradient + sorting)
# =============================================================================

def prep_plot_df(df_top: pd.DataFrame) -> pd.DataFrame:
    """
    Sort ascending so the largest bar appears at the bottom (more readable).
    """
    return df_top.sort_values("count", ascending=True).copy()

def gradient_colors_from_counts(counts: pd.Series):
    """
    Map counts -> gradient colors:
      lighter = less frequent, darker = more frequent
    """
    if counts.empty:
        return []

    if counts.max() == counts.min():
        return [BITCOIN_ORANGE] * len(counts)

    norm = (counts - counts.min()) / (counts.max() - counts.min())
    return [cmap_bars(v) for v in norm]


# =============================================================================
# SECTION 10 — Plot (4 panels, log scale, NO white frame box)
# =============================================================================

def plot_hashtags_4panel(top_all, top_neg, top_neu, top_pos, top_n=TOP_N, log_x=True):
    panels = [
        ("Overall",  top_all, LIGHT_TEXT),
        ("Negative", top_neg, GRID_COLOR),
        ("Neutral",  top_neu, LIGHT_TEXT),
        ("Positive", top_pos, BITCOIN_ORANGE),
    ]

    fig, axes = plt.subplots(ncols=4, figsize=(28, 9), facecolor=DARK_BACKGROUND)

    # Make sure the figure itself has no border
    fig.patch.set_edgecolor(DARK_BACKGROUND)
    fig.patch.set_linewidth(0)

    fig.suptitle(
        f"Top {top_n} Hashtags (Overall + by Sentiment Bucket, VADER)",
        fontsize=20, fontweight="bold", color=LIGHT_TEXT, y=0.98
    )

    fig.text(
        0.5, 0.94,
        "Gradient encodes hashtag frequency (darker = more frequent). X-axis uses log scale to preserve visibility across large frequency ranges.",
        ha="center", fontsize=12, color="gray"
    )

    for ax, (title, df_bucket, title_color) in zip(axes, panels):
        ax.set_facecolor(DARK_BACKGROUND)

        # Remove the subplot frame ("white box")
        for spine in ax.spines.values():
            spine.set_visible(False)

        # If no data, show a friendly message
        if df_bucket is None or df_bucket.empty:
            ax.set_title(title, fontsize=16, fontweight="bold", color=title_color, pad=10)
            ax.text(
                0.5, 0.5, "No hashtags found",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color="gray"
            )
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        plot_df = prep_plot_df(df_bucket)
        colors = gradient_colors_from_counts(plot_df["count"])

        bars = ax.barh(
            plot_df["hashtag"],
            plot_df["count"],
            color=colors,
            edgecolor=LIGHT_TEXT,
            linewidth=0.6,
            alpha=0.95
        )

        # Text + grid styling
        ax.tick_params(axis="x", colors=LIGHT_TEXT, labelsize=10)
        ax.tick_params(axis="y", colors=LIGHT_TEXT, labelsize=10)

        ax.grid(axis="x", color=GRID_COLOR, linestyle="--", alpha=0.35)
        ax.grid(axis="y", visible=False)

        ax.set_title(title, fontsize=16, fontweight="bold", color=title_color, pad=10)
        ax.set_xlabel("Hashtag Frequency (log scale)", fontsize=12, color=LIGHT_TEXT)

        # Log scale: only valid when counts > 0
        if log_x and plot_df["count"].min() > 0:
            ax.set_xscale("log")

        # Bar labels: count + share percent
        for b, pct in zip(bars, plot_df["share_pct"]):
            w = b.get_width()
            ax.text(
                w * 1.03,
                b.get_y() + b.get_height() / 2,
                f"{int(w):,} ({pct:.1f}%)",
                va="center", ha="left",
                fontsize=9, color=LIGHT_TEXT
            )

        # Safe x-limits to avoid ValueError: NaN/Inf
        max_count = plot_df["count"].max()
        if np.isfinite(max_count) and max_count > 0:
            # For log scale, left must be > 0
            ax.set_xlim(left=1, right=max(10, max_count * 3))

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.show()


# =============================================================================
# SECTION 11 — Run Plot
# =============================================================================

plot_hashtags_4panel(top_all, top_neg, top_neu, top_pos, top_n=TOP_N, log_x=True)
