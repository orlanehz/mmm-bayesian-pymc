"""Visualisations des diagnostics et résultats."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_roas(df: pd.DataFrame, save_path: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.errorbar(
        df["channel"],
        df["median"],
        yerr=[
            df["median"] - df["hdi_low"],
            df["hdi_high"] - df["median"],
        ],
        fmt="o",
        capsize=5,
    )
    plt.axhline(0, linestyle="--")
    plt.ylabel("ROAS")
    plt.title("Bayesian ROAS (90% credible interval)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_baseline_vs_media(
    decomp_df: pd.DataFrame,
    *,
    save_path: str,
    title: str = "Baseline vs Media decomposition (model scale)",
) -> None:
    """
    Expects decomp_df with columns: date, component, mean, hdi_low, hdi_high
    and components baseline/media/total.
    """
    df = decomp_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    baseline = df[df["component"] == "baseline"].sort_values("date")
    media = df[df["component"] == "media"].sort_values("date")
    total = df[df["component"] == "total"].sort_values("date")

    plt.figure(figsize=(10, 5))

    # Means
    plt.plot(baseline["date"], baseline["mean"], label="baseline (mean)")
    plt.plot(media["date"], media["mean"], label="media (mean)")
    plt.plot(total["date"], total["mean"], label="total (mean)")

    # Uncertainty band for total
    plt.fill_between(
        total["date"],
        total["hdi_low"],
        total["hdi_high"],
        alpha=0.2,
        label="total 90% HDI",
    )

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Target (model scale)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()