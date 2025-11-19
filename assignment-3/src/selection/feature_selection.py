"""
Feature selection helper.
Prerequisite: data.csv present and sklearn installed.

Steps:
1) Load raw data, clean and standardize using existing preprocessing.
2) Fit L1-regularized logistic regression to induce sparsity.
3) Compute permutation importances to validate feature signal.
4) Tree-based importances, filter (mutual info), and wrapper SFS.
5) Save ranked features and a selected-feature CSV.
"""
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SequentialFeatureSelector, mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split

import sys

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from preprocessing import clean_data, standardize_data


DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
RAW_DATA_PATH = DATA_DIR / "data.csv"
SELECTIONS_DIR = DATA_DIR / "selections"
SELECTED_FEATURES_PATH = SELECTIONS_DIR / "selected_features.csv"
L1_RANKING_PATH = SELECTIONS_DIR / "feature_ranking_l1.csv"
PERM_RANKING_PATH = SELECTIONS_DIR / "feature_ranking_permutation.csv"
TREE_RANKING_PATH = SELECTIONS_DIR / "feature_ranking_tree.csv"
FILTER_RANKING_PATH = SELECTIONS_DIR / "feature_ranking_filter.csv"
WRAPPER_SELECTED_PATH = SELECTIONS_DIR / "wrapper_selected_features.csv"


def train_l1_logreg(
    X: pd.DataFrame, y: pd.Series, C: float = 1.0, max_iter: int = 5000
) -> LogisticRegression:
    # L1 penalty yields sparse weights; saga supports multinomial/binary with L1.
    model = LogisticRegression(
        penalty="l1",
        C=C,
        solver="saga",
        max_iter=max_iter,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X, y)
    return model


def select_by_l1(model: LogisticRegression, feature_names: List[str]) -> pd.DataFrame:
    coefs = model.coef_[0]
    ranking = pd.DataFrame(
        {"feature": feature_names, "coef": coefs, "abs_coef": np.abs(coefs)}
    ).sort_values("abs_coef", ascending=False)
    return ranking


def run_feature_selection(top_n: int = 30) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(RAW_DATA_PATH)
    if df.empty:
        raise ValueError(f"No rows found in input data: {RAW_DATA_PATH}")

    df = clean_data(df)
    df = standardize_data(df)

    feature_cols = [c for c in df.columns if c not in ["ID", "class"]]
    X = df[feature_cols]
    y = df["class"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = train_l1_logreg(X_train, y_train)
    y_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, y_pred)

    l1_ranking = select_by_l1(model, feature_cols)
    selected = l1_ranking.head(top_n).copy()

    # Permutation importance on the validation set to cross-check signal.
    perm = permutation_importance(
        model, X_val, y_val, n_repeats=50, random_state=42, n_jobs=-1
    )
    perm_ranking = pd.DataFrame(
        {
            "feature": feature_cols,
            "mean_importance": perm.importances_mean,
            "std_importance": perm.importances_std,
        }
    ).sort_values("mean_importance", ascending=False)

    # Tree-based selection (Random Forest) using impurity importances.
    rf = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=1,
        class_weight="balanced",
    )
    rf.fit(X_train, y_train)
    tree_importances = pd.DataFrame(
        {"feature": feature_cols, "importance": rf.feature_importances_}
    ).sort_values("importance", ascending=False)

    # Filter method: mutual information ranking.
    mi = mutual_info_classif(X_train, y_train, random_state=42)
    filter_ranking = pd.DataFrame({"feature": feature_cols, "score": mi}).sort_values(
        "score", ascending=False
    )

    # Wrapper method: sequential forward selection with logistic regression.
    wrapper_model = LogisticRegression(
        penalty="l2", solver="liblinear", max_iter=2000, random_state=42
    )
    n_select = min(20, X_train.shape[1])
    sfs = SequentialFeatureSelector(
        wrapper_model,
        n_features_to_select=n_select,
        direction="forward",
        scoring="accuracy",
        n_jobs=-1,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    )
    sfs.fit(X, y)
    wrapper_features = [f for f, keep in zip(feature_cols, sfs.get_support()) if keep]
    wrapper_selected = pd.DataFrame({"feature": wrapper_features})

    # Persist outputs.
    SELECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    l1_ranking.to_csv(L1_RANKING_PATH, index=False)
    perm_ranking.to_csv(PERM_RANKING_PATH, index=False)
    selected.to_csv(SELECTED_FEATURES_PATH, index=False)
    tree_importances.to_csv(TREE_RANKING_PATH, index=False)
    filter_ranking.to_csv(FILTER_RANKING_PATH, index=False)
    wrapper_selected.to_csv(WRAPPER_SELECTED_PATH, index=False)

    print(f"Validation accuracy (L1 logistic): {val_acc:.3f}")
    print(f"Saved L1 ranking to: {L1_RANKING_PATH}")
    print(f"Saved permutation ranking to: {PERM_RANKING_PATH}")
    print(f"Saved tree-based ranking to: {TREE_RANKING_PATH}")
    print(f"Saved filter (mutual information) ranking to: {FILTER_RANKING_PATH}")
    print(f"Saved wrapper-selected features to: {WRAPPER_SELECTED_PATH}")
    print(f"Saved top {top_n} selected features to: {SELECTED_FEATURES_PATH}")

    return l1_ranking, perm_ranking, tree_importances, filter_ranking, wrapper_selected, selected


if __name__ == "__main__":
    run_feature_selection()
