"""
Script: eda_sentiment_distribution_vader.py

Purpose:
Exploratory visualization of VADER compound sentiment scores
for Bitcoin-related tweets. Uses log-scaled histogram and
intensity-based color encoding to surface rare extreme sentiment.
"""



import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import matplotlib.colors as mcolors 
import seaborn as sns
from pathlib import Path 
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
plt.rcParams['figure.figsize'] = (20, 9) # Increased default figsize

ALPHA = 1.0  # Dirichlet Prior for Log Odds Smoothing

## --- DATA LOADING AND PREPARATION ---
print(f"Loading data from: {INPUT_FILE}")
try:
    df = pd.read_csv(INPUT_FILE)
    top=(df.head(10))
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

bitcoin=df[selected_columns]


# ==============================================================================
# --- 1. CONFIGURATION & COLOR PALETTE ---
# ==============================================================================

# --- BASE BITCOIN COLOR PALETTE ---
BITCOIN_ORANGE = '#F7931A' # Core Orange
DARK_BACKGROUND_COLOR = '#101820'
LIGHT_TEXT_COLOR = '#E0E0E0'
GRID_COLOR = '#4A648C' # Subtle blue for the tech grid

# Histogram settings
N_BINS = 100 
VADER_THRESHOLD = 0.05 # Standard VADER neutral zone is -0.05 to +0.05

# --- CUSTOM GRADIENT STOPS (Sharper Distinction) ---
# Lightest point for the neutral/low magnitude center
LIGHTER_ORANGE = '#FFC064' 
# Darkest point for the extreme/high magnitude ends
DARKER_ORANGE = '#6E3300' 

# Define the Custom Sequential Colormap (Bar Face Color)
# N=10 creates a sharper transition between color stops for better distinction
cmap_custom = mcolors.LinearSegmentedColormap.from_list('BitcoinSeq_Final', [LIGHTER_ORANGE, BITCOIN_ORANGE, DARKER_ORANGE], N=8) 

# --- CUSTOM EDGE COLOR GRADIENT (High Contrast) ---
# Edge gradient goes from near-white (low magnitude) to near-black (high magnitude)
cmap_edge = mcolors.LinearSegmentedColormap.from_list('GrayEdge_Final', ['#FFFFFF', LIGHT_TEXT_COLOR, '#101010'], N=8) 


# ------------------------------------Descriptive Statistics -----------------------------


################## Section 1. 
# ----------------------------------------------------------------------
# Histogram 
# ----------------------------------------------------------------------

print("\n" + "="*70)
print("  PHASE 1: Histogram (Gradient & Log Scale)")
print("="*70)

# Temporarily set the display format to show 4 decimal places instead of scientific notation
original_float_format = pd.options.display.float_format
pd.options.display.float_format = '{:,.4f}'.format

# ... (Descriptive Stats section omitted for brevity) ...


# ==============================================================================
# --- 2. FIGURE & AXIS SETUP (Start of Plotting) ---
# ==============================================================================

# 1. Create figure and axis: Increased figsize for better visibility
fig, ax = plt.subplots(figsize=(20, 9), facecolor=DARK_BACKGROUND_COLOR)
ax.set_facecolor(DARK_BACKGROUND_COLOR) 
ax.set_yscale('log')
#ax.set_ylim(bottom=1)

# ==============================================================================
# --- 3. DATA PROCESSING & GRADIENT PLOTTING ---
# ==============================================================================

# Use ax.hist to plot the data, which allows access to the bar patches
# Use log=True to set the scale during the plot call
n, bins, patches = ax.hist(df['compound_score'], bins=N_BINS)


# Calculate the center of each bin for color assignment
bin_centers = 0.5 * (bins[:-1] + bins[1:])
# Color intensity is based on the ABSOLUTE score (stronger sentiment = darker color)
col = np.abs(bin_centers)
max_col = np.max(col) 

# Select the custom colormap
cmap = cmap_custom 

# Loop through patches (bars) and assign color based on bin intensity
for c, p in zip(col, patches):
    # Normalize the absolute score (c) to the colormap range (0 to 1)
    norm_c = c / max_col
    
    # 1. Set Face Color (Orange Gradient)
    face_color = cmap(norm_c)
    plt.setp(p, 'facecolor', face_color)
    
    # 2. Set Edge Color (High Contrast Gray Gradient)
    edge_color = cmap_edge(norm_c)
    plt.setp(p, 'edgecolor', edge_color) 
    
    # 3. Increase Linewidth for visual apparentness
    plt.setp(p, 'linewidth', 0.4) 
    
# ==============================================================================
# --- 4. AXIS FORMATTING & AESTHETICS ---
# ==============================================================================

# # Axis ticks and labels use LIGHT_TEXT_COLOR
ax.tick_params(axis='x', colors=LIGHT_TEXT_COLOR, labelsize=12)
ax.tick_params(axis='y', colors=LIGHT_TEXT_COLOR, labelsize=12)

# Add more X-axis ticks (e.g., every 0.20 units)
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.20)) 

# Add more Y-axis ticks (Logarithmic Scale) 
ax.yaxis.set_minor_locator(ticker.LogLocator(base=20.0, subs=(1,2,3,5,7), numticks=100))


# Gridlines use the subtle blue color
ax.grid(axis='y', linestyle='-', alpha=0.3, color=GRID_COLOR)
ax.grid(axis='x', visible=False)

# Format X-Axis Ticks (VADER Score: -1.0 to 1.0)
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
# Format Y-Axis Ticks (Tweet Count: Log Scale - no scientific notation)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:,.0f}'.format(y)))

# Add VADER Threshold Lines (Simple vertical lines to define zones)
ax.axvline(x=-VADER_THRESHOLD, color=GRID_COLOR, linestyle='-', linewidth=3, alpha=1.0) 
ax.axvline(x=VADER_THRESHOLD, color=BITCOIN_ORANGE, linestyle='-', linewidth=3, alpha=1.0)

# ==============================================================================
# --- 5. TITLES & CATEGORY ANNOTATIONS (No Overlap) ---
# ==============================================================================

# 1. Main Title (fig.suptitle)
fig.suptitle(
    'Tweet Sentiment Distribution (VADER Compound Score)', 
    fontsize=20, 
    fontweight='bold', 
    color=LIGHT_TEXT_COLOR,
    y=0.97 
)

# 2. Explanatory Subtitle (fig.text)
fig.text(
    0.5, 0.92, 
    "Gradient shows sentiment intensity (darker=more extreme). Y-axis is logarithmic to reveal rare emotional tweets.",
    fontsize=14, 
    color='gray', 
    ha='center'
)

# 3. Sentiment Category Labels (Fixed positions above the plot area)
ax_x_min, ax_x_max = ax.get_xlim()
plot_width = ax_x_max - ax_x_min

# Text for Negative zone label (Left side)
fig.text(0.2, 0.85, 
         f'Negative Sentiment ($< {-VADER_THRESHOLD}$)', 
         color=GRID_COLOR, 
         fontsize=14, 
         fontweight='bold', 
         ha='center')
         
# Text for Neutral zone label (Center)
fig.text(0.5, 0.85, 
         f'Neutral Sentiment ($-{VADER_THRESHOLD}$ to $+{VADER_THRESHOLD}$)', 
         color=LIGHT_TEXT_COLOR, 
         fontsize=14, 
         fontweight='bold', 
         ha='center')

# Text for Positive zone label (Right side)
fig.text(0.8, 0.85, 
         f'Positive Sentiment ($> +{VADER_THRESHOLD}$)', 
         color=BITCOIN_ORANGE, 
         fontsize=14, 
         fontweight='bold', 
         ha='center')


# 4. Axis Labels
ax.set_title('', pad=0)
ax.set_xlabel('VADER Compound Score', fontsize=14, color=LIGHT_TEXT_COLOR) 
ax.set_ylabel('Tweet Count (Log Scale)', fontsize=14, color=LIGHT_TEXT_COLOR) 

# Final clean-up to ensure everything fits (rect adjusted for new text height)
plt.tight_layout(rect=[0, 0, 1, 0.8])
plt.show()