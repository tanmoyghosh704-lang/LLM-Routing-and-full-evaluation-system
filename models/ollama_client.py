"""Thin wrapper around the local Ollama HTTP API for the two routing tiers,
plus a separate, more capable judge model used only for correctness grading
(never as a routing candidate) — see eval/llm_judge.py."""

from __future__ import annotations

import time
from dataclasses import dataclass

import requests

OLLAMA_BASE_URL = "http://localhost:11434"

TIER_MODELS = {
    # Originally qwen2.5:1.5b-instruct-q4_0; the pilot run showed it was
    # surprisingly capable (near-degenerate small/large split, see
    # results/writeup.md), so Tier 1 was moved down to 0.5b to restore a
    # genuine capability gap. Staying within the Qwen2.5 family keeps
    # instruction-tuning style constant and isolates parameter count as the
    # variable driving the capability difference.
    "small": "qwen2.5:0.5b-instruct-q4_0",
    "large": "qwen2.5:7b-instruct-q4_0",
}

# Deliberately distinct from both routing tiers: grading answers with the same
# model that produced one of them (Tier 2, in the original design) risks the
# judge being lenient toward its own reasoning style. qwen2.5:14b is only
# ever used for scoring, never as a routing candidate.
JUDGE_MODEL = "qwen2.5:14b"


@dataclass
class GenerationResult:
    tier: str | None
    model: str
    prompt: str
    response_text: str
    latency_seconds: float
    prompt_tokens: int
    completion_tokens: int


def _call_ollama(
    prompt: str,
    model: str,
    *,
    temperature: float = 0.0,
    timeout: float = 300.0,
    base_url: str = OLLAMA_BASE_URL,
    num_predict: int | None = None,
) -> tuple[dict, float]:
    options = {"temperature": temperature}
    if num_predict is not None:
        options["num_predict"] = num_predict
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": options,
    }
    start = time.perf_counter()
    resp = requests.post(f"{base_url}/api/generate", json=payload, timeout=timeout)
    resp.raise_for_status()
    latency = time.perf_counter() - start
    return resp.json(), latency


DEFAULT_MAX_TOKENS = 1024
"""Cap on tier-answer generation length. Without this, small instruction-tuned
models occasionally never emit a stop token and ramble until hitting the
model's context limit (observed: 0.5b generating 40960-token, ~230KB
responses on a handful of queries) — a real failure mode that would badly
skew latency/token-cost measurements if left uncapped. 1024 tokens is
generous relative to every reference answer in the dataset."""


def generate(
    prompt: str,
    tier: str,
    *,
    temperature: float = 0.0,
    timeout: float = 300.0,
    base_url: str = OLLAMA_BASE_URL,
    num_predict: int | None = DEFAULT_MAX_TOKENS,
) -> GenerationResult:
    """Call the given routing tier's model ('small' or 'large').

    Uses temperature=0 by default for reproducible eval runs.
    """
    model = TIER_MODELS[tier]
    data, latency = _call_ollama(
        prompt, model, temperature=temperature, timeout=timeout, base_url=base_url, num_predict=num_predict
    )
    return GenerationResult(
        tier=tier,
        model=model,
        prompt=prompt,
        response_text=data.get("response", ""),
        latency_seconds=latency,
        prompt_tokens=data.get("prompt_eval_count", 0),
        completion_tokens=data.get("eval_count", 0),
    )


def generate_with_model(
    prompt: str,
    model: str,
    *,
    temperature: float = 0.0,
    timeout: float = 300.0,
    base_url: str = OLLAMA_BASE_URL,
    num_predict: int | None = None,
) -> GenerationResult:
    """Call an arbitrary Ollama model by name (used for the judge model)."""
    data, latency = _call_ollama(
        prompt, model, temperature=temperature, timeout=timeout, base_url=base_url, num_predict=num_predict
    )
    return GenerationResult(
        tier=None,
        model=model,
        prompt=prompt,
        response_text=data.get("response", ""),
        latency_seconds=latency,
        prompt_tokens=data.get("prompt_eval_count", 0),
        completion_tokens=data.get("eval_count", 0),
    )


def is_server_running(base_url: str = OLLAMA_BASE_URL) -> bool:
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=5)
        return resp.ok
    except requests.RequestException:
        return False


def ensure_models_pulled(base_url: str = OLLAMA_BASE_URL, include_judge: bool = True) -> None:
    """Raise if a required model isn't present in `ollama list`."""
    resp = requests.get(f"{base_url}/api/tags", timeout=10)
    resp.raise_for_status()
    available = {m["name"] for m in resp.json().get("models", [])}
    required = list(TIER_MODELS.values()) + ([JUDGE_MODEL] if include_judge else [])
    missing = [m for m in required if m not in available]
    if missing:
        raise RuntimeError(
            f"Missing Ollama models: {missing}. Run `ollama pull <model>` first."
        )


if __name__ == "__main__":
    ensure_models_pulled()
    for tier in ("small", "large"):
        result = generate("What is the capital of France?", tier=tier)
        print(f"[{tier}] {result.model} ({result.latency_seconds:.2f}s): {result.response_text.strip()}")
