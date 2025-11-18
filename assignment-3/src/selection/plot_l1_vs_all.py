"""
Plot metrics vs. requested features for L1 subsets (plus all-features baseline).
Prerequisite: run compare_l1_vs_all.py to generate l1_vs_all.csv.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
SELECTIONS_DIR = DATA_DIR / "selections"
CSV_PATH = SELECTIONS_DIR / "l1_vs_all.csv"
OUTPUT_PATH = SELECTIONS_DIR / "l1_vs_all_plot.png"


def main():
    df = pd.read_csv(CSV_PATH)
    if "n_requested" not in df.columns:
        raise ValueError("Expected n_requested column in l1_vs_all.csv")

    # Keep rows with requested n <= 100 and sort descending by requested n
    df = df[df["n_requested"] <= 100].sort_values("n_requested", ascending=False)

    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))
    for metric in ["acc_mean", "auc_mean", "auprc_mean"]:
        if metric in df.columns:
            plt.plot(
                df["n_requested"],
                df[metric],
                marker="o",
                label=metric,
            )

    plt.xlabel("Number of features")
    plt.ylabel("Score")
    plt.title("L1 Feature Performance (n ≤ 100)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300)
    print(f"Saved plot to: {OUTPUT_PATH}")
    plt.close()


if __name__ == "__main__":
    main()
