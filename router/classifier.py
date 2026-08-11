"""Embeddings + classifier router: predicts P(large_model_required).

Originally plain LogisticRegression per the project spec. router.model_comparison
tested it against Random Forest, Gradient Boosting, SVM (RBF), and a small MLP
on the real labeled dataset: SVM (RBF) matched or beat LogisticRegression on
every axis simultaneously (system accuracy, latency, cost -- see
results/project_log.md), so it replaced LogisticRegression as the default
estimator here. LogisticRegression remains available via `make_estimator`.

Training requires a results file with 'query_text' and 'routing_label'
columns ('small' or 'large'), produced empirically per Section 6 after the
batch inference run (pipeline/batch_runner.py) and correctness scoring.
Rows with no routing_label (both tiers incorrect, per Section 6) are
excluded from training.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from router.features import embed

MODEL_PATH = Path(__file__).parent / "logreg_router.joblib"


def make_estimator():
    return SVC(kernel="rbf", C=1, gamma="scale", class_weight="balanced", probability=True, random_state=42)


def load_labeled_data(path: str) -> pd.DataFrame:
    p = Path(path)
    df = pd.read_json(p, lines=True) if p.suffix == ".jsonl" else pd.read_csv(p)
    return df.dropna(subset=["routing_label"])


def split_labeled_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Split df (and its embeddings/labels) into train/test, keeping all three
    aligned. Shared by train() and pipeline/baselines.py so the router and the
    baseline comparisons are always evaluated on the identical held-out rows."""
    X = embed(df["query_text"].tolist())
    y = (df["routing_label"] == "large").astype(int).to_numpy()
    df_train, df_test, X_train, X_test, y_train, y_test = train_test_split(
        df, X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return df_train, df_test, X_train, X_test, y_train, y_test


def train(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    _df_train, _df_test, X_train, X_test, y_train, y_test = split_labeled_data(df, test_size, random_state)

    clf = make_estimator()
    clf.fit(X_train, y_train)

    report = classification_report(y_test, clf.predict(X_test), output_dict=True)
    joblib.dump(clf, MODEL_PATH)
    return clf, report


def predict_proba(query_texts: list[str], clf=None) -> np.ndarray:
    """Return P(large_model_required) for each query."""
    if clf is None:
        clf = joblib.load(MODEL_PATH)
    X = embed(query_texts)
    return clf.predict_proba(X)[:, 1]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m router.classifier <path to labeled results file (.csv or .jsonl)>")
        print("File must contain 'query_text' and 'routing_label' columns,")
        print("populated after the batch inference + labeling steps (Sections 6/11).")
        sys.exit(0)
    labeled_df = load_labeled_data(sys.argv[1])
    trained_clf, eval_report = train(labeled_df)
    print(json.dumps(eval_report, indent=2))
