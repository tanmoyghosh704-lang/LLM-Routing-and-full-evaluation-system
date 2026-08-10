"""Embeddings + Logistic Regression router: predicts P(large_model_required).

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from router.features import embed

MODEL_PATH = Path(__file__).parent / "logreg_router.joblib"


def load_labeled_data(path: str) -> pd.DataFrame:
    p = Path(path)
    df = pd.read_json(p, lines=True) if p.suffix == ".jsonl" else pd.read_csv(p)
    return df.dropna(subset=["routing_label"])


def train(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    X = embed(df["query_text"].tolist())
    y = (df["routing_label"] == "large").astype(int).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    report = classification_report(y_test, clf.predict(X_test), output_dict=True)
    joblib.dump(clf, MODEL_PATH)
    return clf, report


def predict_proba(query_texts: list[str], clf: LogisticRegression | None = None) -> np.ndarray:
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
