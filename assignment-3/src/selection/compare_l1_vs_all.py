"""
Compare performance of all features vs. L1-selected top-N features.

Runs stratified 5-fold CV with logistic regression on:
    - All features (baseline)
    - L1 top-N feature subsets for N from MAX_REQUEST down to 1

Outputs a CSV with metrics in assignment-3/data/selections/l1_vs_all.csv

Prerequisite: run feature_selection.py to generate feature_ranking_l1.csv.
"""
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

import sys

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from preprocessing import clean_data, standardize_data


DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
SELECTIONS_DIR = DATA_DIR / "selections"
RAW_DATA_PATH = DATA_DIR / "data.csv"
L1_RANKING_PATH = SELECTIONS_DIR / "feature_ranking_l1.csv"
OUTPUT_PATH = SELECTIONS_DIR / "l1_vs_all.csv"
MAX_REQUEST = 10  # highest N to evaluate (counts down to 1)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA_PATH)
    if df.empty:
        raise ValueError(f"No rows found in input data: {RAW_DATA_PATH}")
    df = clean_data(df)
    df = standardize_data(df)
    return df


def eval_feature_set(df: pd.DataFrame, features: List[str]) -> dict:
    X = df[features].values
    y = df["class"].values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, aucs, auprcs = [], [], []
    model = LogisticRegression(
        penalty="l2", solver="liblinear", max_iter=2000, random_state=42
    )
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        accs.append(accuracy_score(y_test, y_pred))
        aucs.append(roc_auc_score(y_test, y_prob))
        auprcs.append(average_precision_score(y_test, y_prob))
    return {
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "auprc_mean": float(np.mean(auprcs)),
        "auprc_std": float(np.std(auprcs)),
    }


def main():
    df = load_data()
    all_features = [c for c in df.columns if c not in ["ID", "class"]]

    if not L1_RANKING_PATH.exists():
        raise FileNotFoundError(f"Missing L1 ranking at {L1_RANKING_PATH}. Run feature_selection.py first.")

    l1_rank = pd.read_csv(L1_RANKING_PATH)["feature"].tolist()

    results = []

    # Baseline: all features
    metrics = eval_feature_set(df, all_features)
    metrics.update({"set": "all_features", "n_requested": len(all_features), "n_features": len(all_features)})
    results.append(metrics)

    # L1 subsets: N from MAX_REQUEST down to 1 (request), use available features if fewer.
    for n in range(MAX_REQUEST, 0, -1):
        used = min(n, len(l1_rank))
        topn = l1_rank[:used]
        metrics = eval_feature_set(df, topn)
        metrics.update({"set": f"l1_top{n}", "n_requested": n, "n_features": used})
        results.append(metrics)

    results_df = pd.DataFrame(results).sort_values(["n_requested"], ascending=False)
    SELECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print("Saved comparison to:", OUTPUT_PATH)
    print(results_df)

    # Report best n_requested by metric
    for metric in ["acc_mean", "auc_mean", "auprc_mean"]:
        best_row = results_df.loc[results_df[metric].idxmax()]
        print(
            f"Best {metric}: set={best_row['set']} "
            f"n_requested={int(best_row['n_requested'])} "
            f"n_features={int(best_row['n_features'])} "
            f"value={best_row[metric]:.4f}"
        )


if __name__ == "__main__":
    main()
