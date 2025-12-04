#!/usr/bin/env python3
"""
Task-04 Sentiment Analysis (pure Python)
Usage:
    python main.py --input dataset.csv --outdir outputs
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Ensure required NLTK data is available
nltk_data_needed = ["vader_lexicon"]
for pkg in nltk_data_needed:
    try:
        nltk.data.find(f"sentiment/{pkg}")
    except LookupError:
        print(f"Downloading NLTK data: {pkg} ...")
        nltk.download(pkg)

def find_text_column(df):
    """Try to guess the main text column."""
    candidates = ["text", "tweet", "content", "review", "message", "post"]
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: choose the first object (string) column
    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    if obj_cols:
        return obj_cols[0]
    raise ValueError("No text column found. Please ensure your CSV has a text column.")

def compute_sentiments(df, text_col):
    sia = SentimentIntensityAnalyzer()
    compounds = []
    sentiments = []
    for t in df[text_col].fillna("").astype(str):
        score = sia.polarity_scores(t)["compound"]
        compounds.append(score)
        if score >= 0.05:
            sentiments.append("Positive")
        elif score <= -0.05:
            sentiments.append("Negative")
        else:
            sentiments.append("Neutral")
    df["Polarity"] = compounds
    df["Sentiment"] = sentiments
    return df

def plot_and_save(df, outdir):
    os.makedirs(outdir, exist_ok=True)

    # Sentiment distribution countplot
    plt.figure(figsize=(6,4))
    sns.countplot(data=df, x="Sentiment", order=["Positive","Neutral","Negative"])
    plt.title("Sentiment Distribution")
    plt.tight_layout()
    p1 = os.path.join(outdir, "sentiment_dist.png")
    plt.savefig(p1, dpi=150)
    plt.close()
    print("Saved:", p1)

    # Polarity histogram
    plt.figure(figsize=(8,4))
    sns.histplot(df["Polarity"], kde=True)
    plt.title("Polarity (Compound) Score Distribution")
    plt.xlabel("Compound Polarity Score")
    plt.tight_layout()
    p2 = os.path.join(outdir, "polarity_hist.png")
    plt.savefig(p2, dpi=150)
    plt.close()
    print("Saved:", p2)

    # Sentiment percentage bar chart
    counts = df["Sentiment"].value_counts().reindex(["Positive","Neutral","Negative"]).fillna(0)
    perc = counts / counts.sum() * 100
    plt.figure(figsize=(6,4))
    perc.plot(kind="bar")
    plt.title("Sentiment Percentage")
    plt.ylabel("Percentage (%)")
    plt.tight_layout()
    p3 = os.path.join(outdir, "sentiment_percentage.png")
    plt.savefig(p3, dpi=150)
    plt.close()
    print("Saved:", p3)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", default="dataset.csv", help="Input CSV file")
    parser.add_argument("--outdir", "-o", default="outputs", help="Output directory")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    print("Loading dataset:", args.input)
    df = pd.read_csv(args.input)
    print("Dataset shape:", df.shape)
    print("Columns:", df.columns.tolist())

    text_col = find_text_column(df)
    print("Using text column:", text_col)

    df = compute_sentiments(df, text_col)

    # Save enriched CSV (only a few columns if available)
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    out_csv = os.path.join(outdir, "sentiment_summary.csv")
    df.to_csv(out_csv, index=False)
    print("Saved enriched CSV:", out_csv)

    # Create plots
    plot_and_save(df, outdir)

    # Print simple summary
    counts = df["Sentiment"].value_counts()
    print("\nSentiment counts:")
    print(counts)
    print("\nTop 5 positive samples (by polarity):")
    print(df.sort_values("Polarity", ascending=False).head(5)[[text_col, "Polarity", "Sentiment"]].to_string(index=False))
    print("\nTop 5 negative samples (by polarity):")
    print(df.sort_values("Polarity", ascending=True).head(5)[[text_col, "Polarity", "Sentiment"]].to_string(index=False))

if __name__ == "__main__":
    main()
