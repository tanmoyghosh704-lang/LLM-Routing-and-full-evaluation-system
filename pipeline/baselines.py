"""Baseline systems (Section 7): A (always small), B (always large), plus the
rule-based and embeddings+SVM routers (System C variants).

All four are evaluated on the identical held-out test split used to train the
embeddings+SVM router (router.classifier.split_labeled_data), so none of
them have an unfair data-leakage advantage over the others. Accuracy and
latency come directly from the real per-tier answers/timings already recorded
in the Phase 6 batch run — no new model calls are made here.
"""

import json
from pathlib import Path

import pandas as pd

from router.classifier import split_labeled_data, train
from router.rule_based import route as rule_based_route


def load_full_run(path: str = "data/results/full_run.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def _summarize(df_test: pd.DataFrame, chosen_tier: pd.Series) -> dict:
    chosen_tier = pd.Series(chosen_tier, index=df_test.index)
    is_small = chosen_tier == "small"
    correct = df_test["small_model_correct"].where(is_small, df_test["large_model_correct"])
    latency = df_test["small_model_latency_seconds"].where(is_small, df_test["large_model_latency_seconds"])
    return {
        "n": len(df_test),
        "accuracy": float(correct.astype(bool).mean()),
        "avg_latency_seconds": float(latency.mean()),
        "pct_routed_small": float(is_small.mean()),
    }


def evaluate_baselines(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    labeled = df.dropna(subset=["routing_label"]).copy()
    _df_train, df_test, _X_train, X_test, _y_train, _y_test = split_labeled_data(
        labeled, test_size, random_state
    )
    clf, _report = train(labeled, test_size=test_size, random_state=random_state)

    results = {}

    results["A_always_small"] = _summarize(df_test, pd.Series("small", index=df_test.index))
    results["B_always_large"] = _summarize(df_test, pd.Series("large", index=df_test.index))

    rb_tier = df_test["query_text"].apply(rule_based_route)
    results["C_rule_based_router"] = _summarize(df_test, rb_tier)

    proba = clf.predict_proba(X_test)[:, 1]
    svm_tier = pd.Series(["large" if p >= 0.5 else "small" for p in proba], index=df_test.index)
    results["C_embeddings_svm_router"] = _summarize(df_test, svm_tier)

    return results, df_test, proba


if __name__ == "__main__":
    full_df = load_full_run()
    results, df_test, proba = evaluate_baselines(full_df)

    print(f"Held-out test set: {len(df_test)} queries\n")
    print(f"{'System':30s} {'Accuracy':>10s} {'AvgLatency':>12s} {'%Small':>8s}")
    for name, m in results.items():
        print(f"{name:30s} {m['accuracy']*100:9.1f}% {m['avg_latency_seconds']:11.2f}s {m['pct_routed_small']*100:7.1f}%")

    out_path = Path("data/results/baselines_comparison.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")
