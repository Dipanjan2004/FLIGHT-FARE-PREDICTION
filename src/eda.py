"""Exploratory Data Analysis — generate plots that describe the flight-fare dataset.

Plots are saved as PNGs into the reports/ folder so they can be embedded in the
project report and the Streamlit sidebar.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import REPORTS_DIR
from src.data_loader import load_dataframe

sns.set_theme(style="whitegrid", palette="muted")


def _save(fig: plt.Figure, name: str) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / name
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.relative_to(REPORTS_DIR.parent)}")


def plot_price_distribution(df) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(df["price"], bins=80, color="steelblue", ax=ax)
    ax.set_title("Distribution of flight prices")
    ax.set_xlabel("Price (Rs.)")
    ax.set_ylabel("Number of flights")
    _save(fig, "01_price_distribution.png")


def plot_price_by_class(df) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=df, x="class", y="price", ax=ax)
    ax.set_title("Price by travel class")
    ax.set_xlabel("Class")
    ax.set_ylabel("Price (Rs.)")
    _save(fig, "02_price_by_class.png")


def plot_price_by_airline(df) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    order = df.groupby("airline")["price"].median().sort_values().index
    sns.boxplot(data=df, x="airline", y="price", order=order, ax=ax)
    ax.set_title("Price by airline (sorted by median)")
    ax.set_xlabel("Airline")
    ax.set_ylabel("Price (Rs.)")
    plt.xticks(rotation=20)
    _save(fig, "03_price_by_airline.png")


def plot_price_vs_days_left(df) -> None:
    avg = df.groupby("days_left")["price"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.lineplot(data=avg, x="days_left", y="price", marker="o", ax=ax)
    ax.set_title("Average price vs. days left until departure")
    ax.set_xlabel("Days left")
    ax.set_ylabel("Mean price (Rs.)")
    _save(fig, "04_price_vs_days_left.png")


def plot_price_by_stops(df) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=df, x="stops", y="price",
                order=["zero", "one", "two_or_more"], ax=ax)
    ax.set_title("Price by number of stops")
    ax.set_xlabel("Stops")
    ax.set_ylabel("Price (Rs.)")
    _save(fig, "05_price_by_stops.png")


def plot_duration_vs_price(df) -> None:
    sample = df.sample(min(len(df), 5000), random_state=42)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.scatterplot(data=sample, x="duration", y="price",
                    hue="class", alpha=0.4, s=18, ax=ax)
    ax.set_title("Flight duration vs. price (5000-row sample)")
    ax.set_xlabel("Duration (hours)")
    ax.set_ylabel("Price (Rs.)")
    _save(fig, "06_duration_vs_price.png")


def main() -> None:
    print("Loading dataset...")
    df = load_dataframe()
    print(f"Loaded {len(df):,} rows.")

    print("\nGenerating plots...")
    plot_price_distribution(df)
    plot_price_by_class(df)
    plot_price_by_airline(df)
    plot_price_vs_days_left(df)
    plot_price_by_stops(df)
    plot_duration_vs_price(df)

    print(f"\nDone. Plots saved to {REPORTS_DIR}/")


if __name__ == "__main__":
    main()
