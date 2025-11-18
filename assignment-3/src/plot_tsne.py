"""
Plot precomputed t-SNE embeddings.
Prerequisite: run feature_extraction.py to generate dataTSNE*.csv files.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TSNE_PATH = DATA_DIR / "dataTSNE.csv"


def plot_and_save(df: pd.DataFrame, output_path: Path, title: str):
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))
    ax = sns.scatterplot(
        data=df,
        x="tsne_1",
        y="tsne_2",
        hue="class",
        palette="deep",
        alpha=0.8,
        edgecolor="none",
    )
    ax.set_title(title)
    ax.set_xlabel("tsne_1")
    ax.set_ylabel("tsne_2")
    plt.legend(title="class")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved plot to {output_path}")


def main():
    df = pd.read_csv(TSNE_PATH)
    if not {"tsne_1", "tsne_2", "class"}.issubset(df.columns):
        raise ValueError("Expected columns tsne_1, tsne_2, and class in dataTSNE.csv")

    # Expecting precomputed t-SNE CSVs for each perplexity; adjust names if needed.
    for perplexity in (10, 30, 50):
        path = DATA_DIR / f"dataTSNE_p{perplexity}.csv"
        if path.exists():
            df_p = pd.read_csv(path)
        else:
            # Fallback to the default file if specific one is missing
            df_p = df
            path = TSNE_PATH
        plot_and_save(df_p, DATA_DIR / f"tsne_plot_p{perplexity}.png", f"t-SNE embedding (perplexity={perplexity})")


if __name__ == "__main__":
    main()
