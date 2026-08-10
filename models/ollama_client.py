"""Thin wrapper around the local Ollama HTTP API for the two routing tiers."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import requests

OLLAMA_BASE_URL = "http://localhost:11434"

TIER_MODELS = {
    "small": "qwen2.5:1.5b-instruct-q4_0",
    "large": "qwen2.5:7b-instruct-q4_0",
}

Tier = Literal["small", "large"]


@dataclass
class GenerationResult:
    tier: Tier
    model: str
    prompt: str
    response_text: str
    latency_seconds: float
    prompt_tokens: int
    completion_tokens: int


def generate(
    prompt: str,
    tier: Tier,
    *,
    temperature: float = 0.0,
    timeout: float = 300.0,
    base_url: str = OLLAMA_BASE_URL,
) -> GenerationResult:
    """Call the given tier's model via Ollama's /api/generate endpoint.

    Uses temperature=0 by default for reproducible eval runs.
    """
    model = TIER_MODELS[tier]
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }

    start = time.perf_counter()
    resp = requests.post(f"{base_url}/api/generate", json=payload, timeout=timeout)
    resp.raise_for_status()
    latency = time.perf_counter() - start

    data = resp.json()
    return GenerationResult(
        tier=tier,
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


def ensure_models_pulled(base_url: str = OLLAMA_BASE_URL) -> None:
    """Raise if either tier's model isn't present in `ollama list`."""
    resp = requests.get(f"{base_url}/api/tags", timeout=10)
    resp.raise_for_status()
    available = {m["name"] for m in resp.json().get("models", [])}
    missing = [m for m in TIER_MODELS.values() if m not in available]
    if missing:
        raise RuntimeError(
            f"Missing Ollama models: {missing}. Run `ollama pull <model>` first."
        )


if __name__ == "__main__":
    ensure_models_pulled()
    for tier in ("small", "large"):
        result = generate("What is the capital of France?", tier=tier)
        print(f"[{tier}] {result.model} ({result.latency_seconds:.2f}s): {result.response_text.strip()}")
