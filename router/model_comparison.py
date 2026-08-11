"""Compare candidate classifiers for the embeddings-based router, on the same
held-out test split used throughout the project (router.classifier.split_labeled_data).

Note on latency: router inference itself (embed + classify) runs in
milliseconds regardless of which of these algorithms is used -- LLM
generation time dominates every latency number reported elsewhere in this
project. A better classifier here can only raise accuracy at a given routing
rate, which indirectly lets more traffic shift to the cheap tier at the same
accuracy (moving the whole trade-off frontier), not directly reduce latency.

With only ~265 training examples after the held-out split, more complex
models risk overfitting; hyperparameters are chosen via 5-fold CV on the
training set only, never touching the test set, to keep the comparison fair.
"""

import time

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from pipeline.baselines import load_full_run
from pipeline.threshold_sweep import _add_cost_columns, _summarize
from router.classifier import split_labeled_data
from router.rule_based import route as rule_based_route

CANDIDATES = {
    "logreg_default": (LogisticRegression(max_iter=1000), {}),
    "logreg_tuned": (
        LogisticRegression(max_iter=1000, class_weight="balanced"),
        {"C": [0.01, 0.1, 1, 10]},
    ),
    "random_forest": (
        RandomForestClassifier(random_state=42, class_weight="balanced"),
        {"n_estimators": [100, 300], "max_depth": [None, 5, 10]},
    ),
    "gradient_boosting": (
        GradientBoostingClassifier(random_state=42),
        {"n_estimators": [50, 100], "max_depth": [2, 3]},
    ),
    "svm_rbf": (
        SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42),
        {"C": [0.1, 1, 10], "gamma": ["scale", "auto"]},
    ),
    "mlp": (
        MLPClassifier(max_iter=2000, random_state=42, early_stopping=True),
        {"hidden_layer_sizes": [(32,), (64,)], "alpha": [1e-4, 1e-3, 1e-2]},
    ),
}


def compare_models(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42, cv_folds: int = 5):
    df = _add_cost_columns(df)
    labeled = df.dropna(subset=["routing_label"]).copy()
    _df_train, df_test, X_train, X_test, y_train, y_test = split_labeled_data(labeled, test_size, random_state)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    results = []

    for name, (estimator, param_grid) in CANDIDATES.items():
        t0 = time.perf_counter()
        if param_grid:
            search = GridSearchCV(estimator, param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            best_params = search.best_params_
            cv_accuracy = search.best_score_
        else:
            best_model = estimator.fit(X_train, y_train)
            best_params = {}
            cv_accuracy = None
        fit_time_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        test_pred = best_model.predict(X_test)
        predict_ms_per_query = (time.perf_counter() - t0) * 1000 / len(X_test)

        report = classification_report(y_test, test_pred, output_dict=True)

        # System/answer accuracy: does the *delivered answer* end up correct
        # once you actually route to whichever tier this model predicted --
        # the metric that matters in practice, not just tier-vs-routing_label
        # agreement (see module docstring for why these two can diverge).
        chosen_tier = pd.Series(["large" if p == 1 else "small" for p in test_pred], index=df_test.index)
        system_metrics = _summarize(df_test, chosen_tier)

        results.append(
            {
                "model": name,
                "best_params": best_params,
                "cv_accuracy": cv_accuracy,
                "classifier_test_accuracy": accuracy_score(y_test, test_pred),
                "test_f1_macro": report["macro avg"]["f1-score"],
                "system_accuracy": system_metrics["accuracy"],
                "avg_latency_s": system_metrics["avg_latency_seconds"],
                "avg_cost_usd": system_metrics["avg_cost_usd"],
                "pct_routed_small": system_metrics["pct_routed_small"],
                "fit_time_s": fit_time_s,
                "predict_ms_per_query": predict_ms_per_query,
            }
        )

    return pd.DataFrame(results).sort_values("system_accuracy", ascending=False).reset_index(drop=True), df_test


def reference_systems(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Baseline A/B and the rule-based router, for side-by-side comparison."""
    df = _add_cost_columns(df)
    labeled = df.dropna(subset=["routing_label"]).copy()
    _df_train, df_test, _X_train, _X_test, _y_train, _y_test = split_labeled_data(labeled, test_size, random_state)

    baseline_a = _summarize(df_test, pd.Series("small", index=df_test.index))
    baseline_b = _summarize(df_test, pd.Series("large", index=df_test.index))
    rb_tier = df_test["query_text"].apply(rule_based_route)
    rule_based = _summarize(df_test, rb_tier)
    return baseline_a, baseline_b, rule_based


if __name__ == "__main__":
    full_df = load_full_run()
    results_df, df_test = compare_models(full_df)
    baseline_a, baseline_b, rule_based = reference_systems(full_df)

    pd.set_option("display.width", 160)
    print(f"Held-out test set: {len(df_test)} queries\n")
    print("Reference systems (fixed, no threshold):")
    for name, m in [("A_always_small", baseline_a), ("B_always_large", baseline_b), ("rule_based", rule_based)]:
        print(
            f"  {name:20s} accuracy={m['accuracy']*100:5.1f}%  latency={m['avg_latency_seconds']:5.2f}s  "
            f"cost=${m['avg_cost_usd']:.5f}  %small={m['pct_routed_small']*100:5.1f}%"
        )
    print("\nCandidate router models (all at default 0.5 decision threshold):")
    print(results_df.to_string(index=False))
