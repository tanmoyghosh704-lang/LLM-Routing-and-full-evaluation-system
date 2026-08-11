"""Demo wrapper: live routing decisions over HTTP.

Run with:
    uvicorn demo.app:app --reload

Then either open http://127.0.0.1:8000/docs for interactive Swagger UI, or:
    curl -X POST http://127.0.0.1:8000/route -H "Content-Type: application/json" \
         -d '{"query": "What is the capital of France?"}'

For each query: embeds it, gets P(large_model_required) from the trained
router, applies the recommended threshold (pipeline.threshold_sweep.RECOMMENDED_THRESHOLD,
overridable per-request), calls the chosen Ollama tier, and returns the tier
used, routing probability, latency, simulated cost, and the answer -- the
full "query in -> tier used, latency, answer out" trail Section 12 asks for.
"""

from __future__ import annotations

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

from models.ollama_client import generate, is_server_running
from pipeline.threshold_sweep import RECOMMENDED_THRESHOLD, simulate_cost
from router.classifier import MODEL_PATH
from router.features import embed

app = FastAPI(
    title="Cost-Aware LLM Router — Demo",
    description="Routes a query to the small or large model tier based on a trained router's prediction.",
)

_router_model = None


def _get_router_model():
    global _router_model
    if _router_model is None:
        _router_model = joblib.load(MODEL_PATH)
    return _router_model


class RouteRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The query to route and answer.")
    threshold: float = Field(
        RECOMMENDED_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="P(large) >= threshold routes to the large tier. Defaults to the recommended operating point.",
    )


class RouteResponse(BaseModel):
    query: str
    tier_used: str
    p_large: float
    threshold: float
    latency_seconds: float
    simulated_cost_usd: float
    answer: str


@app.get("/")
def root():
    return {
        "message": "Cost-aware LLM routing demo. POST /route with {'query': '...'}. See /docs for details.",
        "recommended_threshold": RECOMMENDED_THRESHOLD,
    }


@app.get("/health")
def health():
    return {"ollama_running": is_server_running()}


@app.post("/route", response_model=RouteResponse)
def route(req: RouteRequest) -> RouteResponse:
    clf = _get_router_model()
    p_large = float(clf.predict_proba(embed([req.query]))[0, 1])
    tier = "large" if p_large >= req.threshold else "small"

    result = generate(req.query, tier=tier)
    cost = simulate_cost(tier, result.prompt_tokens, result.completion_tokens)

    return RouteResponse(
        query=req.query,
        tier_used=tier,
        p_large=p_large,
        threshold=req.threshold,
        latency_seconds=result.latency_seconds,
        simulated_cost_usd=cost,
        answer=result.response_text,
    )
