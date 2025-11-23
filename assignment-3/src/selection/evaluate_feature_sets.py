"""
Evaluate different feature sets using cross-validated models.

Loads the cleaned + standardized data, builds feature subsets from the
saved selection outputs, and scores them with a simple classifier.

Prerequisite: run feature_selection.py to generate the selection CSVs.

Usage:
    python assignment-3/src/selection/evaluate_feature_sets.py
"""
from pathlib import Path
from typing import Dict, List

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
TREE_RANKING_PATH = SELECTIONS_DIR / "feature_ranking_tree.csv"
FILTER_RANKING_PATH = SELECTIONS_DIR / "feature_ranking_filter.csv"
WRAPPER_SELECTED_PATH = SELECTIONS_DIR / "wrapper_selected_features.csv"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA_PATH)
    if df.empty:
        raise ValueError(f"No rows found in input data: {RAW_DATA_PATH}")
    df = clean_data(df)
    df = standardize_data(df)
    return df


def read_feature_list(path: Path, column: str = "feature") -> List[str]:
    if not path.exists():
        return []
    return pd.read_csv(path)[column].tolist()


def build_feature_sets(df: pd.DataFrame, top_n: int = 30) -> Dict[str, List[str]]:
    sets: Dict[str, List[str]] = {}

    # All features (excluding ID/class) as a baseline.
    all_feature_cols = [c for c in df.columns if c not in ["ID", "class"]]
    sets["all_features"] = all_feature_cols

    # Top-N from L1 ranking
    if L1_RANKING_PATH.exists():
        sets["l1_topN"] = pd.read_csv(L1_RANKING_PATH).head(top_n)["feature"].tolist()

    # Top-N from tree importances
    if TREE_RANKING_PATH.exists():
        sets["tree_topN"] = pd.read_csv(TREE_RANKING_PATH).head(top_n)["feature"].tolist()

    # Top-N from filter (mutual information)
    if FILTER_RANKING_PATH.exists():
        sets["filter_topN"] = pd.read_csv(FILTER_RANKING_PATH).head(top_n)["feature"].tolist()

    # Wrapper-selected set
    wrapper_feats = read_feature_list(WRAPPER_SELECTED_PATH)
    if wrapper_feats:
        sets["wrapper"] = wrapper_feats

    # Consensus: features that appear in at least two lists
    all_feats = []
    for feats in sets.values():
        all_feats.extend(feats)
    consensus = [f for f in set(all_feats) if all_feats.count(f) >= 2]
    if consensus:
        sets["consensus"] = consensus

    return sets


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
        "n_features": len(features),
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "auprc_mean": float(np.mean(auprcs)),
        "auprc_std": float(np.std(auprcs)),
    }


def main():
    df = load_data()
    feature_sets = build_feature_sets(df)
    if not feature_sets:
        raise ValueError("No feature sets found. Run feature_selection.py first.")

    results = []
    for name, feats in feature_sets.items():
        if not feats:
            continue
        metrics = eval_feature_set(df, feats)
        metrics["set"] = name
        results.append(metrics)

    results_df = pd.DataFrame(results).sort_values("auprc_mean", ascending=False)
    out_path = SELECTIONS_DIR / "feature_set_evaluation.csv"
    SELECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False)

    print("Feature set evaluation (sorted by AUPRC):")
    print(results_df)
    print(f"Saved metrics to: {out_path}")


if __name__ == "__main__":
    main()
