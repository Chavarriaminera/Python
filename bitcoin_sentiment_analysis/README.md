# Bitcoin Sentiment Analysis

Exploratory analysis of sentiment dynamics in Bitcoin-related tweets
using VADER compound sentiment scores. This project focuses on
understanding the full sentiment distribution and identifying rare
but extreme emotional responses in social media data.

## Motivation
Social media sentiment is often highly imbalanced toward neutral
expressions. While most observations cluster near neutrality, rare
extreme sentiment events can be disproportionately important for
downstream modeling, risk analysis, and event-driven insights.
This project explores how visualization choices can surface those
signals without distortion.

## Approach
- Text preprocessing of Bitcoin-related tweets
- Sentiment scoring using VADER compound sentiment scores
- Exploratory data analysis (EDA) of sentiment distributions
- Log-scaled histogram visualization to preserve visibility of rare,
  high-magnitude sentiment values

## Example Output

![Sentiment Distribution](figures/sentiment_distribution_vader.png)

*Distribution of VADER compound sentiment scores for Bitcoin-related
tweets. A logarithmic y-axis is used to prevent the dominant neutral
mass from obscuring low-frequency but extreme sentiment events.*

## Interpretation
Most tweets cluster near neutral sentiment, creating a dominant
central peak. A logarithmic y-axis is used to prevent this mass from
flattening the tails of the distribution. The resulting visualization
reveals heavier positive sentiment tails relative to negative ones,
suggesting asymmetry in how sentiment is expressed around Bitcoin
in social media discourse.

## Project Structure
The analysis is organized as a stepwise pipeline:

- `0-Step 0.py` — Data ingestion and initial cleaning
- `1-Step 1.py` — Sentiment scoring using VADER
- `2-Step 2.py` — Core exploratory data analysis
- `2-Step 2 EDA.py` — Extended EDA and visualization experiments
- `Step2.2EDAHistogram.py` — Iterative refinement of sentiment
  distribution histograms

Scripts are ordered to reflect a typical exploratory-to-analytical
workflow used in applied NLP projects.

## Status
Work in progress. Planned next steps include top counts of hastags for both negative and positve sentiment.

