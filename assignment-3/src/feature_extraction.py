"""
Feature extraction: clean + standardize + t-SNE embeddings.
Prerequisite: data.csv present; sklearn installed. No other scripts required.
"""
import inspect
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.manifold import TSNE

from preprocessing import clean_data, standardize_data


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DATA_PATH = DATA_DIR / "data.csv"
TSNE_OUTPUT_PATH = DATA_DIR / "dataTSNE_p50.csv"


def tsne_embedding(
    df: pd.DataFrame,
    n_components: int = 2,
    perplexity: float = 50,
    learning_rate: float | str = "auto",
    n_iter: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c not in ["ID", "class"]]

    # Build params that are supported by the installed sklearn version.
    supported = inspect.signature(TSNE).parameters
    params = {
        "n_components": n_components,
        "perplexity": perplexity,
        "learning_rate": learning_rate,
        "init": "pca",
        "random_state": random_state,
    }
    if "n_iter" in supported:
        params["n_iter"] = n_iter
    elif "max_iter" in supported:
        params["max_iter"] = n_iter

    tsne = TSNE(**params)
    embedding = tsne.fit_transform(df[feature_cols])

    tsne_cols = [f"tsne_{i + 1}" for i in range(n_components)]
    tsne_df = pd.DataFrame(embedding, columns=tsne_cols, index=df.index)
    tsne_df["class"] = df["class"].values
    if "ID" in df.columns:
        tsne_df["ID"] = df["ID"].values
        tsne_df = tsne_df[["ID"] + tsne_cols + ["class"]]
    return tsne_df


def run_feature_extraction(
    data_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    tsne_kwargs: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Execute the feature extraction pipeline and return the t-SNE embedding.

    Args:
        data_path: Optional path to the input CSV. Defaults to assignment-3/data/data.csv.
        output_path: Optional path to save the t-SNE CSV. Defaults to assignment-3/data/dataTSNE.csv.
        tsne_kwargs: Optional dict with parameters forwarded to tsne_embedding (e.g., n_components, perplexity).

    Returns:
        DataFrame containing the t-SNE embedding plus ID/class columns.
    """
    data_path = Path(data_path) if data_path else RAW_DATA_PATH
    output_path = Path(output_path) if output_path else TSNE_OUTPUT_PATH
    tsne_kwargs = tsne_kwargs or {}

    df = pd.read_csv(data_path)
    if df.empty:
        raise ValueError(f"No rows found in input data: {data_path}")

    df = clean_data(df)
    df = standardize_data(df)
    tsne_df = tsne_embedding(df, **tsne_kwargs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tsne_df.to_csv(output_path, index=False)

    return tsne_df


if __name__ == "__main__":
    embedding = run_feature_extraction()
    print("t-SNE embedding created.")
    print(f"Rows: {len(embedding)}, Columns: {list(embedding.columns)}")
    print(f"Saved to: {TSNE_OUTPUT_PATH}")
