# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 18:25:03 2025

What is this script for 
-------------------------------
Ths script performs exploratory data analysis 
- What are the most common words and phrases?
- Which emojis/hashtags show up a lot (if we still have raw text)?
- How does language change month to month (topic drift)?
- When are people posting (weekday × hour)?
- How long are tweets (token count)?
- If we have sentiment labels and/or VADER scores, do they look sane?

@author: chava
"""
import pandas as pd
import pyflakes as py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from pathlib import Path  # <-- CRITICAL: Ensures 'Path' is defined
from sklearn.feature_extraction.text import CountVectorizer
import math
import warnings

# Suppress harmless pandas warnings during chaining
# The line below is the fix for the AttributeError
warnings.filterwarnings('ignore', message='A value is trying to be set on a copy of a slice from a DataFrame')

# --- YOUR DEFINED FILE PATHS ---
INPUT_FILE  = Path("C:/Users/chava/OneDrive/Documents/Data Science Projects/Bitcoin/Step2bitcoin_tweets_clean.csv")
OUTPUT_FILE = Path("C:/Users/chava/OneDrive/Documents/Data Science Projects/Bitcoin/Step2bitcoin_tweets_EDA.csv")

# --- CONFIGURATION ---
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
ALPHA = 1.0  # Dirichlet Prior for Log Odds Smoothing

## --- DATA LOADING AND PREPARATION ---
print(f"Loading data from: {INPUT_FILE}")
try:
    df = pd.read_csv(INPUT_FILE)
    print(f"Successfully loaded {len(df):,} rows.")
except FileNotFoundError:
    print(f"ERROR: Input file not found at {INPUT_FILE}. Please check the path and file name.")
    exit() # Stop execution if the file isn't found


selected_columns=[
    'user_name',
    'clean_text',
    'compound_score',
    'quantile_sentiment'
    ]

df=df[selected_columns]



# ------------------------------------Descriptive Statistics -----------------------------


################## Section 1. 
# ----------------------------------------------------------------------
# Histogram 
# ----------------------------------------------------------------------

print("\n"+"="*50)

print("\n" + "="*50)
print("  PHASE 1: Histogram")
print("="*50)

# Temporarily set the display format to show 4 decimal places instead of scientific notation
original_float_format = pd.options.display.float_format
pd.options.display.float_format = '{:,.4f}'.format

print("\nDescriptive Stats for VADER Compound:")
# 2. Generates descriptive statistics (count, mean, standard deviation, min, max, quartiles) 
#    for the 'VADER_compound' sentiment score column.
#    This helps understand the central tendency, spread, and range of the sentiment scores.
print(df['compound_score'].describe())

#---
##  Distribution Histogram
# ----------------------------------------------------------------------
# This section visualizes the distribution of the continuous VADER compound scores.
# ----------------------------------------------------------------------
plt.figure(figsize=(10, 5))
# Creates a histogram (a bar chart showing frequency distribution) of the 'VADER_compound' scores.
# bins=100 provides high resolution to see the score spread, and kde=False turns off the density curve.
sns.histplot(df['compound_score'], bins=100, kde=False)
plt.title('Histogram of VADER Compound Scores')
plt.xlabel('VADER Compound Score')
plt.ylabel('Tweet Count')
plt.show()

################## Section 2 
# ----------------------------------------------------------------------
##                            Quantile Sentiment Bar Chart
# ----------------------------------------------------------------------
# This section visualizes the distribution of a categorical sentiment variable created 
# by binning the VADER compound scores into 7 quantile levels.
# ----------------------------------------------------------------------
plt.figure(figsize=(10, 5))
# 1. Calculates the frequency of each unique value in the 'quantile_sentiment' column.
#    normalize=True converts the counts to proportions, and mul(100) converts them to percentages.
sentiment_counts = df['quantile_sentiment'].value_counts(normalize=True).mul(100)
sentiment_order = [
    'extremely negative', 'very negative', 'negative', 'neutral',
    'positive', 'very positive', 'extremely positive'
]
# 2. Reindexes the sentiment counts to ensure the bar chart displays the categories in a logical, 
#    sequential order from most negative to most positive.
sentiment_counts = sentiment_counts.reindex(sentiment_order)

# 3. Creates a bar plot showing the percentage distribution of the 7 sentiment classes.
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='viridis')
plt.title('Distribution of 7-Level Quantile Sentiment')
plt.ylabel('Percentage of Tweets')
plt.xlabel('Sentiment Class')
# 4. Rotates the x-axis labels to prevent overlap for better readability.
plt.xticks(rotation=45, ha='right')
plt.show()



################## Section 3 
# ----------------------------------------------------------------------
####                            Top Word Frequency Analysis 
# ----------------------------------------------------------------------

N = 20
CHUNK_SIZE = 500000
MAX_FEATURES = 50000

# --- BITCOIN COLOR PALETTE ---
BITCOIN_ORANGE = '#F7931A'
# Updated colors for high contrast on dark background
DARK_BACKGROUND_COLOR = '#101820'
LIGHT_TEXT_COLOR = '#E0E0E0'
GRID_COLOR = '#4A648C' # Subtle blue for the tech grid

# --- SECTION START: TOP WORD FREQUENCY ANALYSIS ---

print("\n" + "="*50)
print("PHASE 2: TOPIC AND VOCABULARY ANALYSIS (Chunked)")
print("="*50)

# --- Step 1: Initialize Vectorizer and Define Total Rows (FIXED) ---
# NOTE: The variable 'df' must be defined in your execution environment for this code to run.

# Calculate total rows BEFORE the loop
total_rows = len(df['clean_text'].dropna())

# Initialize CountVectorizer
vectorizer = CountVectorizer(stop_words='english', max_features=MAX_FEATURES)

# Fit on the first chunk to establish vocabulary
first_chunk = df['clean_text'].dropna().astype(str).head(CHUNK_SIZE)
X_first = vectorizer.fit_transform(first_chunk)
total_count_sum = X_first.sum(axis=0)


# --- Step 2: Iterate and Aggregate in Chunks (Memory-Safe) ---

print(f"\nProcessing documents in chunks of {CHUNK_SIZE}...")

for i in range(CHUNK_SIZE, total_rows, CHUNK_SIZE):
    start = i
    end = min(i + CHUNK_SIZE, total_rows)
    print(f"Processing rows: {start} to {end}...")

    current_chunk = df['clean_text'].dropna().astype(str).iloc[start:end]
    X_chunk = vectorizer.transform(current_chunk)
    total_count_sum += X_chunk.sum(axis=0)

print("Aggregation complete.")

# --- Step 3: Final Data Preparation ---

word_counts = pd.DataFrame(
    total_count_sum.T,
    index=vectorizer.get_feature_names_out(),
    columns=['Count']
).sort_values(by='Count', ascending=False)

plot_data = word_counts.head(N)

# --- Step 4: Define Custom Formatter ---
def millions_formatter(x, pos):
    return f'{x / 1_000_000:.1f}M'

# --- Step 5: Visualization (Final Aesthetically Polished Style with Manual Glow) ---

# 1. Setup Dark Style
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(12, 7))

# Set figure and axis background colors darker than default 'dark_background'
fig.patch.set_facecolor(DARK_BACKGROUND_COLOR)
ax.set_facecolor(DARK_BACKGROUND_COLOR)

# --- MANUAL GLOW EFFECT ---
# We draw the bars multiple times to create a glow effect without mplcyberpunk.
for i in range(N):
    # 1. Wide, transparent bar (simulates darkest part of glow)
    ax.bar(plot_data.index[i], plot_data['Count'].iloc[i],
           color=BITCOIN_ORANGE,
           alpha=0.1,
           width=1.0)
    # 2. Slightly narrower, more visible bar (middle glow)
    ax.bar(plot_data.index[i], plot_data['Count'].iloc[i],
           color=BITCOIN_ORANGE,
           alpha=0.25,
           width=0.9)

# 3. Draw the main bar on top using seaborn
ax = sns.barplot(
    x=plot_data.index,
    y=plot_data['Count'],
    color=BITCOIN_ORANGE,
    saturation=1.0,
    linewidth=0,
    ax=ax # Ensure seaborn uses the existing axes
)

# --- Apply Enhancements ---
# 1. Axis Ticks & Alignment
ax.yaxis.set_major_formatter(ticker.FuncFormatter(millions_formatter))
# Use LIGHT_TEXT_COLOR for visibility on dark background
ax.tick_params(axis='both', labelsize=11, length=0, colors=LIGHT_TEXT_COLOR)
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

# 2. Subtler Gridlines
ax.grid(axis='y', linestyle='-', alpha=0.3, color=GRID_COLOR) # Tech grid
ax.grid(axis='x', visible=False)

# 3. Titles and Narrative
ax.set_title('', pad=0)

# Use LIGHT_TEXT_COLOR for titles
fig.text(
    x=0.08, y=0.95,
    s=f'Top {N} Crypto Terms: Frequency Analysis',
    fontsize=18,
    fontweight='bold',
    color=LIGHT_TEXT_COLOR
)

fig.text(
    x=0.08, y=0.91,
    s=f'Word frequency based on {total_rows:,} documents (excluding common stopwords)',
    fontsize=12,
    color='gray' # Gray still works well for secondary text
)

ax.set_xlabel(None)
ax.set_ylabel('Total Count of Mentions', fontsize=12, labelpad=15, color=LIGHT_TEXT_COLOR)

# --- Annotation for Visual Impact ---
for i, (count, word) in enumerate(zip(plot_data['Count'], plot_data.index)):
    if i < 4: # Annotate only the top 4 words
        ax.text(
            x=i,
            y=count + (ax.get_ylim()[1] * 0.01),
            s=millions_formatter(count, None),
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold',
            color=LIGHT_TEXT_COLOR
        )

# 4. Remove Spines (Borders)
sns.despine(ax=ax, top=True, right=True, left=True, bottom=True)

plt.tight_layout(rect=[0, 0, 1, 0.9])
plt.show()