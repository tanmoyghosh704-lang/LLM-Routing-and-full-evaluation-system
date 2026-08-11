"""Portable batch inference runner — identical code for local or Kaggle execution.

Runs each query through both Ollama tiers, logs latency and token counts,
scores correctness (exact-match where gradable, else LLM-as-judge), and
derives the empirical routing_label per Section 6:
  small correct & large correct   -> "small"
  small incorrect & large correct -> "large"
  small correct & large incorrect -> "small"
  both incorrect                  -> None (excluded from router training)

On Kaggle: install Ollama, start it as a background process, and pull both
models before invoking this script — no other environment-specific changes
are needed.
"""

import argparse
import random
from pathlib import Path

import pandas as pd

from eval.correctness import try_exact_match
from eval.llm_judge import judge, judge_pair
from models.ollama_client import ensure_models_pulled, generate, is_server_running


def load_queries(path: str) -> pd.DataFrame:
    p = Path(path)
    return pd.read_json(p, lines=True) if p.suffix == ".jsonl" else pd.read_csv(p)


def select_pilot_sample(df: pd.DataFrame, n_per_difficulty: int = 10, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    parts = []
    for _, group in df.groupby("difficulty_label"):
        idx = list(group.index)
        rng.shuffle(idx)
        parts.append(group.loc[idx[:n_per_difficulty]])
    return pd.concat(parts).sort_values("query_id").reset_index(drop=True)


def score_pair(
    query_text: str, reference_answer: str, small_answer: str, large_answer: str
) -> tuple[bool, str, bool, str]:
    """Score both candidate answers, using a single combined judge call when both
    fall through exact-match (the common case), to roughly halve judge latency."""
    small_result = try_exact_match(small_answer, reference_answer)
    large_result = try_exact_match(large_answer, reference_answer)

    if small_result is None and large_result is None:
        small_result, large_result, _ = judge_pair(query_text, reference_answer, small_answer, large_answer)
        return small_result, "llm_judge", large_result, "llm_judge"

    small_method = "exact_match"
    if small_result is None:
        small_result, _ = judge(query_text, reference_answer, small_answer)
        small_method = "llm_judge"

    large_method = "exact_match"
    if large_result is None:
        large_result, _ = judge(query_text, reference_answer, large_answer)
        large_method = "llm_judge"

    return small_result, small_method, large_result, large_method


def derive_routing_label(small_correct: bool, large_correct: bool) -> str | None:
    if small_correct:
        return "small"
    if large_correct:
        return "large"
    return None


def run_batch(
    input_path: str,
    output_path: str,
    limit: int | None = None,
    pilot: bool = False,
    checkpoint_every: int = 10,
) -> pd.DataFrame:
    if not is_server_running():
        raise RuntimeError("Ollama server is not running at http://localhost:11434")
    ensure_models_pulled()

    df = load_queries(input_path)
    if pilot:
        df = select_pilot_sample(df)
    elif limit:
        df = df.head(limit)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for i, row in df.iterrows():
        query_text = row["query_text"]
        reference_answer = row["reference_answer"]
        print(f"[{len(records)+1}/{len(df)}] {row['query_id']} ({row['difficulty_label']}): {query_text[:60]}", flush=True)

        small = generate(query_text, tier="small")
        large = generate(query_text, tier="large")

        small_correct, small_method, large_correct, large_method = score_pair(
            query_text, reference_answer, small.response_text, large.response_text
        )
        routing_label = derive_routing_label(small_correct, large_correct)

        record = dict(row)
        record.update(
            {
                "small_model_answer": small.response_text,
                "large_model_answer": large.response_text,
                "small_model_latency_seconds": small.latency_seconds,
                "large_model_latency_seconds": large.latency_seconds,
                "small_model_prompt_tokens": small.prompt_tokens,
                "small_model_completion_tokens": small.completion_tokens,
                "large_model_prompt_tokens": large.prompt_tokens,
                "large_model_completion_tokens": large.completion_tokens,
                "small_model_correct": small_correct,
                "large_model_correct": large_correct,
                "small_scoring_method": small_method,
                "large_scoring_method": large_method,
                "routing_label": routing_label,
            }
        )
        records.append(record)
        print(
            f"    small={'OK' if small_correct else 'X'} ({small_method}, {small.latency_seconds:.1f}s)  "
            f"large={'OK' if large_correct else 'X'} ({large_method}, {large.latency_seconds:.1f}s)  "
            f"routing_label={routing_label}",
            flush=True,
        )

        if checkpoint_every and len(records) % checkpoint_every == 0:
            pd.DataFrame(records).to_csv(out_path, index=False)
            print(f"    [checkpoint] saved {len(records)}/{len(df)} rows to {out_path}", flush=True)

    out_df = pd.DataFrame(records)
    out_df.to_csv(out_path, index=False)
    out_df.to_json(out_path.with_suffix(".jsonl"), orient="records", lines=True)
    print(f"\nWrote {len(out_df)} results to {out_path}")
    return out_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/queries_raw.jsonl")
    parser.add_argument("--output", default="data/results/pilot_run.csv")
    parser.add_argument("--pilot", action="store_true", help="Run the 30-query stratified pilot sample")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run_batch(args.input, args.output, limit=args.limit, pilot=args.pilot)
