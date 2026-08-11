"""Threshold sweep + trade-off curves (Section 10), including the simulated
cost methodology (Section 2).

Cost disclosure: Ollama inference is free and local, so "cost" here is
SIMULATED by mapping each tier's actual recorded token counts onto published
per-token pricing of a comparably-sized commercial API tier (a cheap-tier vs.
premium-tier split, modeled loosely on public pricing for small vs. flagship
hosted models). This is a stated methodology for producing a comparable cost
signal, not real spend, and is disclosed explicitly here and in the write-up.

Sweeps the trained embeddings+SVM router's P(large_model_required) across
the specified thresholds on the same held-out test split used throughout
Section 7, with Baseline A (always small), Baseline B (always large), and the
rule-based router as fixed reference points (they don't have a threshold).
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from pipeline.baselines import load_full_run
from router.classifier import split_labeled_data, train
from router.rule_based import route as rule_based_route

THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]

# Simulated per-million-token USD pricing, modeled on public cheap-tier vs.
# premium-tier commercial API pricing (e.g. a GPT-4o-mini/Claude Haiku class
# of pricing for the small tier, GPT-4o/Claude Sonnet class for the large
# tier) -- NOT the actual cost of running Ollama locally, which is free.
PRICING_PER_MILLION_TOKENS = {
    "small": {"input": 0.15, "output": 0.60},
    "large": {"input": 2.50, "output": 10.00},
}

PLOTS_DIR = Path("results/plots")

COLOR_ROUTER = "#2a78d6"       # categorical slot 1 (blue)
COLOR_BASELINE_B = "#1baf7a"   # categorical slot 2 (aqua)
COLOR_BASELINE_A = "#eda100"   # categorical slot 3 (yellow)
COLOR_RULE_BASED = "#008300"   # categorical slot 4 (green)
COLOR_GRID = "#e1e0d9"
COLOR_AXIS = "#898781"
COLOR_INK = "#0b0b0b"


def simulate_cost(tier: str, prompt_tokens: float, completion_tokens: float) -> float:
    p = PRICING_PER_MILLION_TOKENS[tier]
    return (prompt_tokens / 1e6) * p["input"] + (completion_tokens / 1e6) * p["output"]


def _add_cost_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["small_model_cost"] = df.apply(
        lambda r: simulate_cost("small", r["small_model_prompt_tokens"], r["small_model_completion_tokens"]), axis=1
    )
    df["large_model_cost"] = df.apply(
        lambda r: simulate_cost("large", r["large_model_prompt_tokens"], r["large_model_completion_tokens"]), axis=1
    )
    return df


def _summarize(df_test: pd.DataFrame, chosen_tier: pd.Series) -> dict:
    chosen_tier = pd.Series(chosen_tier, index=df_test.index)
    is_small = chosen_tier == "small"
    correct = df_test["small_model_correct"].where(is_small, df_test["large_model_correct"])
    latency = df_test["small_model_latency_seconds"].where(is_small, df_test["large_model_latency_seconds"])
    cost = df_test["small_model_cost"].where(is_small, df_test["large_model_cost"])
    return {
        "accuracy": float(correct.astype(bool).mean()),
        "avg_latency_seconds": float(latency.mean()),
        "avg_cost_usd": float(cost.mean()),
        "pct_routed_small": float(is_small.mean()),
    }


def run_sweep(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    df = _add_cost_columns(df)
    labeled = df.dropna(subset=["routing_label"]).copy()
    _df_train, df_test, _X_train, X_test, _y_train, _y_test = split_labeled_data(labeled, test_size, random_state)
    clf, _report = train(labeled, test_size=test_size, random_state=random_state)
    proba = clf.predict_proba(X_test)[:, 1]

    baseline_a = _summarize(df_test, pd.Series("small", index=df_test.index))
    baseline_b = _summarize(df_test, pd.Series("large", index=df_test.index))
    rb_tier = df_test["query_text"].apply(rule_based_route)
    rule_based = _summarize(df_test, rb_tier)

    rows = []
    for t in THRESHOLDS:
        chosen = pd.Series(["large" if p >= t else "small" for p in proba], index=df_test.index)
        m = _summarize(df_test, chosen)
        m["threshold"] = t
        m["latency_reduction_vs_B_pct"] = (
            (baseline_b["avg_latency_seconds"] - m["avg_latency_seconds"]) / baseline_b["avg_latency_seconds"] * 100
        )
        m["cost_reduction_vs_B_pct"] = (
            (baseline_b["avg_cost_usd"] - m["avg_cost_usd"]) / baseline_b["avg_cost_usd"] * 100
        )
        m["accuracy_loss_vs_B_pp"] = (baseline_b["accuracy"] - m["accuracy"]) * 100
        rows.append(m)

    sweep_df = pd.DataFrame(rows)[
        [
            "threshold",
            "accuracy",
            "pct_routed_small",
            "avg_latency_seconds",
            "latency_reduction_vs_B_pct",
            "avg_cost_usd",
            "cost_reduction_vs_B_pct",
            "accuracy_loss_vs_B_pp",
        ]
    ]
    return sweep_df, baseline_a, baseline_b, rule_based, df_test


def _style_axes(ax):
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, color=COLOR_GRID, linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(COLOR_AXIS)
    ax.tick_params(colors=COLOR_AXIS)
    ax.xaxis.label.set_color(COLOR_INK)
    ax.yaxis.label.set_color(COLOR_INK)
    ax.title.set_color(COLOR_INK)


def make_metric_plot(sweep_df, baseline_a, baseline_b, rule_based, metric_col, ylabel, title, filename, pct=False):
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    _style_axes(ax)

    y = sweep_df[metric_col] * 100 if pct else sweep_df[metric_col]
    ax.plot(
        sweep_df["threshold"], y, color=COLOR_ROUTER, linewidth=2, marker="o", markersize=8,
        label="Embeddings + SVM router", zorder=3,
    )

    bb_val = baseline_b[metric_col] * 100 if pct else baseline_b[metric_col]
    ba_val = baseline_a[metric_col] * 100 if pct else baseline_a[metric_col]
    rb_val = rule_based[metric_col] * 100 if pct else rule_based[metric_col]

    ax.axhline(bb_val, color=COLOR_BASELINE_B, linewidth=2, linestyle="--", label="Baseline B (always large)", zorder=2)
    ax.axhline(ba_val, color=COLOR_BASELINE_A, linewidth=2, linestyle="--", label="Baseline A (always small)", zorder=2)
    ax.axhline(rb_val, color=COLOR_RULE_BASED, linewidth=2, linestyle=":", label="Rule-based router", zorder=2)

    ax.set_xlabel("Routing threshold (P(large) >= t -> large)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, loc="best", fontsize=8.5)
    fig.tight_layout()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / filename)
    plt.close(fig)


def make_accuracy_vs_cost_savings_plot(sweep_df, baseline_a, baseline_b, rule_based, recommended_threshold):
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    _style_axes(ax)

    ax.plot(
        sweep_df["cost_reduction_vs_B_pct"], sweep_df["accuracy"] * 100,
        color=COLOR_ROUTER, linewidth=2, marker="o", markersize=8, zorder=3,
        label="Embeddings + SVM router (by threshold)",
    )
    for _, row in sweep_df.iterrows():
        if row["threshold"] == recommended_threshold:
            ax.plot(
                row["cost_reduction_vs_B_pct"], row["accuracy"] * 100,
                color=COLOR_ROUTER, marker="o", markersize=14, markeredgecolor=COLOR_INK,
                markeredgewidth=1.5, zorder=4,
            )
            ax.annotate(
                f"  chosen: t={recommended_threshold}",
                (row["cost_reduction_vs_B_pct"], row["accuracy"] * 100),
                fontsize=8.5, color=COLOR_INK,
            )

    ax.scatter([0], [baseline_b["accuracy"] * 100], color=COLOR_BASELINE_B, s=70, zorder=3, label="Baseline B (always large)")
    a_cost_reduction = (baseline_b["avg_cost_usd"] - baseline_a["avg_cost_usd"]) / baseline_b["avg_cost_usd"] * 100
    ax.scatter([a_cost_reduction], [baseline_a["accuracy"] * 100], color=COLOR_BASELINE_A, s=70, zorder=3, label="Baseline A (always small)")
    rb_cost_reduction = (baseline_b["avg_cost_usd"] - rule_based["avg_cost_usd"]) / baseline_b["avg_cost_usd"] * 100
    ax.scatter([rb_cost_reduction], [rule_based["accuracy"] * 100], color=COLOR_RULE_BASED, s=70, zorder=3, label="Rule-based router")

    ax.set_xlabel("Simulated cost reduction vs. Baseline B (%)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs. cost savings trade-off")
    ax.legend(frameon=False, loc="best", fontsize=8.5)
    fig.tight_layout()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / "accuracy_vs_cost_savings.png")
    plt.close(fig)


if __name__ == "__main__":
    full_df = load_full_run()
    sweep_df, baseline_a, baseline_b, rule_based, df_test = run_sweep(full_df)

    print(f"Held-out test set: {len(df_test)} queries\n")
    print(sweep_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    out_dir = Path("data/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    sweep_df.to_csv(out_dir / "threshold_sweep.csv", index=False)
    (out_dir / "threshold_sweep_baselines.json").write_text(
        json.dumps({"A_always_small": baseline_a, "B_always_large": baseline_b, "rule_based_router": rule_based}, indent=2)
    )
    print(f"\nWrote {out_dir / 'threshold_sweep.csv'} and threshold_sweep_baselines.json")

    make_metric_plot(sweep_df, baseline_a, baseline_b, rule_based, "accuracy", "Accuracy (%)", "Accuracy vs. routing threshold", "accuracy_vs_threshold.png", pct=True)
    make_metric_plot(sweep_df, baseline_a, baseline_b, rule_based, "avg_cost_usd", "Avg. simulated cost per query (USD)", "Simulated cost vs. routing threshold", "cost_vs_threshold.png")
    make_metric_plot(sweep_df, baseline_a, baseline_b, rule_based, "avg_latency_seconds", "Avg. latency per query (s)", "Latency vs. routing threshold", "latency_vs_threshold.png")

    # Recommended operating point: see results/project_log.md for the full
    # justification. t=0.65 is the largest threshold (most cost/latency
    # savings) at which the SVM router's accuracy still matches or beats the
    # rule-based router's flat 82.1% -- past this point accuracy drops below
    # that baseline and the curve's slope also visibly steepens.
    RECOMMENDED_THRESHOLD = 0.65
    make_accuracy_vs_cost_savings_plot(sweep_df, baseline_a, baseline_b, rule_based, RECOMMENDED_THRESHOLD)
    print(f"Wrote plots to {PLOTS_DIR}/")
