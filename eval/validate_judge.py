"""LLM-as-judge manual validation (Section 9).

Samples judge-scored (query, candidate answer) pairs from the full run,
re-runs the judge to capture its written justification (not persisted
elsewhere), and exports a CSV for a human to independently assess each one
before comparing against the judge's verdict. Do not scale the sample beyond
~40 examples (diminishing returns, protects time budget) and do not skip
this step -- it's what makes using an LLM-as-judge defensible at all.

Workflow:
  1. python -m eval.validate_judge sample   -> writes judge_validation_sample.csv
     with an empty 'human_verdict' column.
  2. A human fills in 'human_verdict' for every row ('correct' or 'incorrect'),
     based on their own independent read of the query/reference/candidate
     answer -- not anchored on the judge's verdict or justification.
  3. python -m eval.validate_judge score    -> reports the agreement rate.
"""

import random
import sys
from pathlib import Path

import pandas as pd

from eval.llm_judge import judge

SAMPLE_PATH = Path("data/results/judge_validation_sample.csv")
SAMPLE_SIZE = 35


def _expand_llm_judged_rows(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (query, tier) pair that was actually scored by the judge."""
    rows = []
    for _, row in df.iterrows():
        for tier in ("small", "large"):
            if row[f"{tier}_scoring_method"] == "llm_judge":
                rows.append(
                    {
                        "query_id": row["query_id"],
                        "difficulty_label": row["difficulty_label"],
                        "tier": tier,
                        "query_text": row["query_text"],
                        "reference_answer": row["reference_answer"],
                        "candidate_answer": row[f"{tier}_model_answer"],
                        "recorded_verdict": bool(row[f"{tier}_model_correct"]),
                    }
                )
    return pd.DataFrame(rows)


def build_sample(full_run_path: str = "data/results/full_run.csv", n: int = SAMPLE_SIZE, seed: int = 7) -> pd.DataFrame:
    df = pd.read_csv(full_run_path)
    judged = _expand_llm_judged_rows(df)

    rng = random.Random(seed)
    idx = list(judged.index)
    rng.shuffle(idx)
    sample = judged.loc[idx[:n]].reset_index(drop=True)

    justifications, verdicts = [], []
    for i, row in sample.iterrows():
        print(f"[{i+1}/{len(sample)}] re-judging {row['query_id']} ({row['tier']})...", flush=True)
        verdict, raw_text = judge(row["query_text"], row["reference_answer"], row["candidate_answer"])
        verdicts.append(verdict)
        justifications.append(raw_text)

    sample["judge_verdict"] = ["correct" if v else "incorrect" for v in verdicts]
    sample["judge_justification"] = justifications
    sample["human_verdict"] = ""  # to be filled in by hand: "correct" or "incorrect"

    # Sanity check: the freshly re-run verdict should almost always match what
    # was actually recorded during the batch run (same model, temperature=0).
    mismatches = (sample["judge_verdict"] == "correct") != sample["recorded_verdict"]
    if mismatches.any():
        print(f"NOTE: {mismatches.sum()} row(s) where re-judging gave a different verdict than the original run.")

    sample = sample[
        [
            "query_id", "difficulty_label", "tier", "query_text", "reference_answer",
            "candidate_answer", "human_verdict", "judge_verdict", "judge_justification", "recorded_verdict",
        ]
    ]
    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(SAMPLE_PATH, index=False)
    print(f"\nWrote {len(sample)} rows to {SAMPLE_PATH}")
    print("Fill in the 'human_verdict' column ('correct' / 'incorrect') for each row, then run:")
    print("  python -m eval.validate_judge score")
    return sample


def score_sample(path: str = str(SAMPLE_PATH)) -> dict:
    df = pd.read_csv(path)
    df["human_verdict"] = df["human_verdict"].astype(str).str.strip().str.lower()
    unfilled = df["human_verdict"].isin(["", "nan"])
    if unfilled.any():
        print(f"WARNING: {unfilled.sum()} row(s) still have an empty human_verdict -- excluding them from scoring.")
        df = df[~unfilled]

    agree = (df["human_verdict"] == df["judge_verdict"]).mean()
    print(f"Scored rows: {len(df)}")
    print(f"Agreement rate: {agree*100:.1f}%")

    disagreements = df[df["human_verdict"] != df["judge_verdict"]]
    if len(disagreements):
        print(f"\nDisagreements ({len(disagreements)}):")
        for _, row in disagreements.iterrows():
            print(f"  {row['query_id']} ({row['tier']}): judge={row['judge_verdict']}  human={row['human_verdict']}")

    return {"n": len(df), "agreement_rate": agree, "n_disagreements": len(disagreements)}


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "sample"
    if mode == "sample":
        build_sample()
    elif mode == "score":
        score_sample()
    else:
        print("Usage: python -m eval.validate_judge [sample|score]")
