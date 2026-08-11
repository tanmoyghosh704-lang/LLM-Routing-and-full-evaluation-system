# Cost-Aware LLM Routing & Evaluation System

Routes each incoming query to the cheapest LLM tier capable of answering it
correctly, instead of always calling an expensive model. A trained router
(sentence embeddings + SVM) predicts `P(large_model_required)` per query;
sweeping that probability across a threshold trades off accuracy against
simulated cost and latency, with a specific operating point chosen and
justified against real measured data.

**At the recommended operating point:** ~82% accuracy retained vs. always
using the large model, while cutting simulated cost by ~34% and latency by
~22% — and genuinely beating a simple keyword-heuristic router on cost at a
matched accuracy level, not just trading one metric for the other.

Full methodology, results, disclosed limitations, and the reasoning behind
every design decision: **[results/writeup.md](results/writeup.md)**.
The complete build/debugging journal (what broke, how it was diagnosed and
fixed, in order): **[results/project_log.md](results/project_log.md)**.

## Repo structure

```
models/           Ollama HTTP client (small/large tiers + judge model)
router/           rule_based.py (heuristic), features.py (embeddings),
                   classifier.py (trained SVM router), model_comparison.py
eval/              correctness.py (exact-match), llm_judge.py (LLM-as-judge),
                   validate_judge.py (human validation sampling/scoring)
pipeline/          batch_runner.py, baselines.py, threshold_sweep.py
data/raw/          generated candidate dataset (350 queries)
data/results/      batch run outputs, baselines, threshold sweep, judge sample
notebooks/         kaggle_batch_run.ipynb (portable full-run notebook)
results/           plots/, writeup.md, project_log.md
demo/              app.py (FastAPI live-routing demo)
```

## Setup

```
python -m venv .venv
.venv/Scripts/activate      # Windows; use `source .venv/bin/activate` on macOS/Linux
pip install -r requirements.txt

ollama pull qwen2.5:0.5b-instruct-q4_0
ollama pull qwen2.5:7b-instruct-q4_0
ollama pull qwen2.5:14b
```

Ollama must be running locally (`ollama serve`, or the desktop app) before
running any of the commands below.

## Reproducing the results

```
# 1. Pilot run (30-query stratified sample) — sanity-check before the full batch
python -m pipeline.batch_runner --pilot --output data/results/pilot_run.csv

# 2. Full run (350 queries) — either locally, or via notebooks/kaggle_batch_run.ipynb
#    on a free Kaggle GPU (recommended: the full run takes hours on modest
#    local hardware; see results/project_log.md Phase 5/6 for why)
python -m pipeline.batch_runner --output data/results/full_run.csv

# 3. Train the router (embeddings + SVM) and see held-out performance
python -m router.classifier data/results/full_run.csv

# 4. Compare classifier algorithms (optional — this is how SVM was chosen
#    over the original LogisticRegression)
python -m router.model_comparison

# 5. Baseline A/B/C comparison
python -m pipeline.baselines

# 6. Threshold sweep, trade-off plots, and the recommended operating point
python -m pipeline.threshold_sweep

# 7. LLM-judge manual validation (Section 9) — samples judge-scored answers
#    for human review, then scores agreement once you've filled them in
python -m eval.validate_judge sample
#   ... fill in the human_verdict column in data/results/judge_validation_sample.csv ...
python -m eval.validate_judge score
```

## Demo

```
uvicorn demo.app:app --reload
```

Then open `http://127.0.0.1:8000/docs`, or:

```
curl -X POST http://127.0.0.1:8000/route \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the capital of France?"}'
```

Returns the tier used, the router's `P(large)`, latency, simulated cost, and
the answer.

## Key results summary

| System | Accuracy | Avg. latency | % routed to small |
|---|---|---|---|
| Always small | 44.8% | 2.58s | 100% |
| Always large | 100.0% | 7.98s | 0% |
| Rule-based router | 82.1% | 6.58s | 49.3% |
| **Trained router (SVM) @ threshold 0.65** | **82.1%** | **6.19s** | **53.7%** |

- Threshold 0.65 chosen because it's the largest threshold (most savings) at
  which the trained router's accuracy still matches or beats the rule-based
  baseline — full justification in the write-up.
- LLM-as-judge correctness scoring validated against independent human
  review: **97.1% agreement** (34/35), with the one disagreement's root
  cause documented rather than glossed over.
- Cost and latency figures are simulated/same-hardware proxies, explicitly
  disclosed in the write-up — not real API spend or production latency.

## Known limitations (see write-up for full detail)

- Held-out test set is small (67 queries) — real sampling noise.
- Dataset has not yet had its manual 20-25% human review pass.
- Simulated cost is based on assumed public API pricing tiers, not real spend.
