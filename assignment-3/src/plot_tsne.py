from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TSNE_PATH = DATA_DIR / "dataTSNE.csv"


def main():
    df = pd.read_csv(TSNE_PATH)
    if not {"tsne_1", "tsne_2", "class"}.issubset(df.columns):
        raise ValueError("Expected columns tsne_1, tsne_2, and class in dataTSNE.csv")

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
    ax.set_title("t-SNE embedding")
    ax.set_xlabel("tsne_1")
    ax.set_ylabel("tsne_2")
    plt.legend(title="class")
    plt.tight_layout()

    # Save plot to the data directory
    output_path = DATA_DIR / "tsne_plot.png"
    plt.savefig(output_path, dpi=300)
    print(f"Saved plot to {output_path}")

    # Show the plot interactively (optional)
    plt.show()


if __name__ == "__main__":
    main()
