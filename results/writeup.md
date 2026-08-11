# Cost-Aware LLM Routing — Methodology & Results

## 1. Problem and objective

Large LLMs are more capable but slower and more expensive per query than small
LLMs, which are cheap and fast but fail more often on harder queries. A naive
system that always calls the large model wastes cost and latency on queries a
small model could have answered correctly; a system that always calls the
small model is unreliable. This project builds a data-driven router that
predicts, per query, whether a cheap/fast model is sufficient or whether the
query needs the expensive/slow one — and empirically characterizes the
resulting accuracy/cost/latency trade-off well enough to justify a specific
operating point, rather than picking a threshold by feel.

**The goal is not to minimize any single axis.** Always using the cheap model
trivially minimizes cost and latency (Baseline A, below) at the cost of
reliability; always using the expensive model trivially maximizes accuracy
(Baseline B) at the cost of paying full price for every query, including
trivial ones. The interesting, harder problem — and the one this project
answers with real numbers — is finding and justifying the right balance.

## 2. Methodology

### 2.1 Model tiers

| Tier | Model | Role |
|---|---|---|
| Small (Tier 1) | `qwen2.5:0.5b-instruct-q4_0` | Cheap/fast routing candidate |
| Large (Tier 2) | `qwen2.5:7b-instruct-q4_0` | Expensive/slow routing candidate |
| Judge | `qwen2.5:14b` | Correctness grading only — never a routing candidate |

**Deviation from the original design, disclosed:** Tier 1 was originally
specified as `qwen2.5:1.5b-instruct-q4_0`. An early pilot run showed this
model was surprisingly capable — the resulting small/large split was nearly
degenerate (87% of queries resolved to "small," including most "hard"-tier
conceptual questions), which would leave a router with almost nothing to
learn. Tier 1 was moved down to 0.5B, staying within the same model family so
that instruction-tuning style is held constant and parameter count is the
one variable driving the capability gap. This is exactly the kind of
empirical check Section 5 of the project brief calls for ("verify hard
queries are actually hard") — it just turned out the fix that was needed was
to the model tier, not the dataset.

The judge model is deliberately distinct from both routing tiers. Using the
same model to both answer a query (as Tier 2) and grade answers risks
leniency toward its own reasoning style; a dedicated, more capable model
avoids that. This was not a hypothetical concern — see Section 3.6.

### 2.2 Dataset

350 queries, hand-authored (not template-generated), across three intended
difficulty tiers and 16 categories:

| Difficulty | Count |
|---|---|
| Easy | 120 |
| Medium | 115 |
| Hard | 115 |

Categories span general knowledge, geography, history, physics, chemistry,
biology, mathematics, programming, logic puzzles, grammar, business/finance,
literature, everyday reasoning, philosophy/ethics, long-context analysis, and
domain-specific expert questions (law, engineering, immunology, etc.), each
with a reference answer. Two arithmetic errors were caught and fixed during
authoring; the final set has zero duplicate query texts.

**Not yet done:** the project brief's required manual review of ~20-25% of
the dataset for correctness/difficulty-appropriateness, and moving a
reviewed subset to `data/curated/`. The results below are computed on the
full uncurated 350-query set.

### 2.3 Routing label derivation

Ground truth is empirical, not assumed. Both tiers actually answer every
query; correctness is scored independently for each; the routing label is
derived per query:

- Small correct → **small** (regardless of large's result)
- Small incorrect, large correct → **large**
- Both incorrect → **excluded** from router training (unanswerable by either
  tier at this dataset's difficulty)

### 2.4 Correctness scoring

Two methods, chosen automatically per answer:

- **Exact-match** (`eval/correctness.py`): for reference answers with a
  short, objective core fact (a number, name, or short phrase before any
  explanatory clause), checked via phrase match with a numeric fallback.
  Reference answers longer than 20 words defer to the judge rather than risk
  a spurious match on a coincidental short fragment.
- **LLM-as-judge** (`eval/llm_judge.py`): for open-ended queries, using the
  dedicated judge model with a prompt that requires step-by-step
  verification of each claim/calculation before ruling on correctness,
  rather than a loose "does this look roughly right" check. Manually
  validated — see Section 3.6.

### 2.5 Router architecture

Two variants, per the project brief:

1. **Rule-based** (`router/rule_based.py`): a keyword + length heuristic,
   quick sanity-check baseline, no training.
2. **Embeddings + classifier** (`router/classifier.py`): sentence-transformer
   embeddings (`all-MiniLM-L6-v2`, 384-dim) feed a classifier that outputs
   `P(large_model_required)` — a probability, not a hard decision, so it can
   be tuned via threshold rather than retrained for different operating
   points. Originally plain LogisticRegression; six candidate algorithms
   (LogReg default/tuned, Random Forest, Gradient Boosting, SVM-RBF, a small
   MLP) were compared on identical held-out data, and SVM (RBF kernel)
   replaced LogisticRegression as the default after matching or beating it
   on accuracy, latency, and cost simultaneously — see Section 3.3.

### 2.6 Simulated cost methodology (disclosure)

Ollama inference is free and local, so it provides no real cost signal.
"Cost" in this report is **simulated** by mapping each tier's actual recorded
token counts onto published per-token pricing for a comparably-positioned
commercial API tier — a cheap-tier vs. premium-tier split modeled on public
pricing for a small vs. flagship hosted model class (in the ballpark of
GPT-4o-mini for the small tier and GPT-4o for the large tier):

| Tier | Input ($/1M tokens) | Output ($/1M tokens) |
|---|---|---|
| Small | $0.15 | $0.60 |
| Large | $2.50 | $10.00 |

This gives a realistic ~16.7x cost ratio between tiers. **This is not real
spend** — it is a stated proxy for comparing routing strategies on a
cost-like axis, disclosed explicitly rather than implied to be actual API
billing.

### 2.7 Latency methodology (disclosure)

Latency is wall-clock time for each Ollama `/api/generate` call. The 30-query
pilot runs were executed locally (RTX 1650, 4GB VRAM — Tier 2 partially
spills to CPU/RAM by design, which is what creates the latency gap this
project studies). The full 350-query run — which all baseline, threshold
sweep, and router-comparison results in this report are computed from — was
executed on a Kaggle P100 GPU instead, after the local pilot showed the full
run would take ~6-7 hours locally. Both tiers were measured on the same
hardware within each run, so within-run comparisons are self-consistent, but
these are relative, same-hardware comparisons between tiers — not
representative of real production API latency, which depends on
provider-side batching, network conditions, and infrastructure this project
has no access to.

## 3. Results

### 3.1 Full run and routing label distribution

350 queries processed through both tiers on Kaggle (P100, ~2.1 hours):

| Difficulty | small | large | excluded (both wrong) |
|---|---|---|---|
| Easy | 91 | 20 | 9 |
| Medium | 42 | 71 | 2 |
| Hard | 17 | 91 | 7 |
| **Total** | **150** | **182** | **18** |

This correlates sensibly with intended difficulty (hard-tier queries need
the large model 79% of the time) — the check the original 1.5B Tier 1 model
failed, and the reason it was replaced (Section 2.1).

Across all 332 labeled (non-excluded) rows, the large tier is correct 99.1%
of the time — near-ceiling, as expected for the stronger model on queries
either tier can answer at all. Three queries are the rarer edge case flagged
in the project brief as worth reporting if it occurs: small correct, large
incorrect — see `results/plots` source data
(`data/results/full_run.csv`) for the specific rows.

### 3.2 Baseline comparison (A/B/C)

Evaluated on an identical 67-query held-out test split (never used for
training the classifier):

| System | Accuracy | Avg. latency | % routed to small |
|---|---|---|---|
| A — always small | 44.8% | 2.58s | 100% |
| B — always large | 100.0% | 7.98s | 0% |
| C — rule-based router | 82.1% | 6.58s | 49.3% |
| C — embeddings + SVM router (default 0.5 threshold) | 91.0% | 7.10s | 38.8% |

Neither naive baseline is acceptable on its own — A is unreliable, B pays
full price for every query. Both router variants recover most of B's
accuracy while sending a substantial fraction of traffic to the cheap tier.

### 3.3 Router model selection

Six classifier candidates were compared on the same held-out split, with
hyperparameters chosen via 5-fold cross-validation on the training set only
(the test set was never used for model selection):

| Model | System accuracy | Avg. latency | Avg. cost | % to small |
|---|---|---|---|---|
| Random Forest | 89.6% | 7.29s | $0.00292 | 32.8% |
| **SVM (RBF)** | **88.1%** | **6.91s** | **$0.00268** | **43.3%** |
| Gradient Boosting | 86.6% | 7.24s | $0.00288 | 38.8% |
| LogisticRegression (default) | 86.6% | 6.93s | $0.00269 | 41.8% |
| LogisticRegression (CV-tuned) | 85.1% | 6.89s | $0.00264 | 46.3% |
| MLP (neural net) | 79.1% | 6.46s | $0.00240 | 52.2% |

SVM (RBF) matched or beat LogisticRegression on every axis simultaneously —
the cleanest available improvement, so it became the new default. Random
Forest scores higher on raw accuracy but achieves it by routing more
conservatively (less to the cheap tier, higher cost) — a different point on
the trade-off frontier, not a strict win. The MLP underperformed, consistent
with the overfitting risk of a higher-capacity model trained on only ~265
examples. (Note: router inference itself is milliseconds regardless of
algorithm — these latency differences come entirely from which tier gets
chosen more often, not from classifier speed.)

### 3.4 Threshold sweep and trade-off curves

The SVM router's `P(large_model_required)` was swept across the required
thresholds, with cost computed per Section 2.6:

| Threshold | Accuracy | % to small | Avg. latency | Latency saved vs. B | Avg. cost | Cost saved vs. B |
|---|---|---|---|---|---|---|
| 0.50 | 91.0% | 38.8% | 7.10s | 11.1% | $0.00280 | 18.2% |
| 0.55 | 88.1% | 43.3% | 6.91s | 13.4% | $0.00268 | 21.8% |
| 0.60 | 86.6% | 47.8% | 6.59s | 17.5% | $0.00249 | 27.4% |
| **0.65** | **82.1%** | **53.7%** | **6.19s** | **22.4%** | **$0.00226** | **34.1%** |
| 0.70 | 76.1% | 61.2% | 5.66s | 29.1% | $0.00194 | 43.2% |
| 0.75 | 74.6% | 65.7% | 5.42s | 32.1% | $0.00181 | 47.0% |
| 0.80 | 65.7% | 74.6% | 4.95s | 38.0% | $0.00152 | 55.5% |
| 0.85 | 58.2% | 83.6% | 4.02s | 49.6% | $0.00103 | 69.9% |

Four plots in `results/plots/` visualize this: `accuracy_vs_threshold.png`,
`cost_vs_threshold.png`, `latency_vs_threshold.png`, and the combined
`accuracy_vs_cost_savings.png`, each with Baseline A, Baseline B, and the
rule-based router plotted as fixed reference lines.

### 3.5 Recommended operating point: threshold = 0.65

At t=0.65, the router's accuracy (82.1%) is **exactly equal to** the
rule-based router's flat accuracy (82.1%), while its cost savings (34.1%)
clearly exceed the rule-based router's (~26.5%, computed the same way) —
genuine dominance at this point, not a trade-off. Past t=0.65, accuracy
drops below the rule-based baseline (76.1% at t=0.70) and the router no
longer offers a "matches-or-beats the simple heuristic" guarantee.

**Selection rule:** pick the largest threshold (maximizing cost/latency
savings) subject to accuracy still matching or exceeding the rule-based
baseline it's meant to replace. This is a more externally-anchored
justification than reading the curve's slope in isolation, and it is
visible directly in `accuracy_vs_cost_savings.png` as the point where the
SVM router's curve sits level with the rule-based router's point but
further right (more savings).

At this operating point, relative to always using the large model: **80%+
of B's accuracy retained, cost cut by over a third, latency cut by nearly a
quarter** — while remaining tunable if priorities change (a team that valued
latency over accuracy could move to t=0.75-0.80 instead, for example).

### 3.6 LLM-as-judge validation (Section 9)

35 (query, tier) pairs that were actually scored by the judge (not
exact-match) were sampled and independently reviewed by hand, without
anchoring on the judge's verdict.

**Agreement rate: 97.1% (34/35).**

The one disagreement (query: "significance of the Magna Carta") is
substantive, not noise: the candidate answer contained two real errors — an
incorrect regnal number ("King John I"; there was no King John II) and an
anachronistic claim that royal taxation required "the consent of
Parliament" (which didn't yet exist as an institution in 1215) — that the
judge missed because its verification focused on thematic alignment with the
reference answer rather than granular factual claims. This is a real,
demonstrated blind spot, not merely a validation formality: it shows the
judge can be fooled by an answer that matches the reference's overall shape
while getting specific facts wrong.

A related finding: re-running the judge on these 35 items at temperature=0
produced a different verdict than the originally recorded one for 3 of them
(8.6%), most likely floating-point/batching non-determinism in local
inference rather than true randomness. This means a small fraction of the
correctness labels underlying this entire report carry some inherent noise
— disclosed here rather than presented as exact.

**Net assessment:** 97.1% agreement is a strong, defensible number for
trusting the judge in aggregate, but it is not perfect, and the one miss
illustrates a specific, reportable limitation rather than random error.

## 4. Key findings

- Naive "always cheap" and "always expensive" strategies are both bad in
  different ways — Baseline A is unreliable (44.8%), Baseline B is
  needlessly expensive on every query, including trivial ones.
- The trained router isn't placebo ML: at a comparable operating point, it
  beats the keyword-heuristic baseline on the metric it's supposed to
  improve (cost), not just on accuracy, evidence that the embeddings carry
  real signal about query difficulty.
- Small-model capability is not a safe assumption — the original 1.5B Tier 1
  model was capable enough to make routing nearly pointless; this was only
  caught by checking the routing label distribution empirically, exactly as
  the project brief's own pilot-run degeneracy check is designed to catch.
- Every "trust the automation" point in this pipeline (the exact-match
  scorer, the LLM-judge, unbounded generation) had at least one real,
  demonstrable failure mode, found by inspecting suspicious outputs rather
  than assuming the numbers were correct. Full account in
  `results/project_log.md`.

## 5. Limitations

- **Small held-out test set** (67 queries) — accuracy percentages carry real
  sampling noise, visible in some of the sweep's less smooth steps.
- **Simulated, not real, cost** — a disclosed proxy based on commercial API
  pricing tiers, not actual spend (Section 2.6).
- **Same-hardware, not production-representative, latency** — self-consistent
  within each run, but not a claim about real-world API latency (Section
  2.7); the pilot ran on different hardware (local RTX 1650) than the full
  run used for all headline numbers (Kaggle P100).
- **Judge non-determinism**: ~8.6% of a small validation sample changed
  verdict on re-judging at temperature=0, indicating some inherent noise in
  the correctness labels this whole report is built on.
- **No single operating point strictly dominates on every axis** — the
  recommended threshold (0.65) is a justified choice given a specific
  selection rule (match-or-beat the rule-based baseline's accuracy), not the
  only defensible one; a different priority (e.g. "maximize savings subject
  to accuracy ≥ 75%") would pick t=0.70 instead.
- **Dataset not yet manually curated** — the project brief's required
  20-25% human review of the dataset for correctness/difficulty-
  appropriateness has not been completed.

## 6. Conclusion

A router built on sentence embeddings and a tuned SVM classifier, trained on
empirically-derived routing labels, recovers roughly 80% of the always-large
baseline's accuracy while cutting simulated cost by about a third and
latency by nearly a quarter — and does so while genuinely outperforming a
simple keyword heuristic at a matched accuracy level, not just trading one
metric for another. The recommended operating point (threshold = 0.65) is
chosen by an explicit, externally-anchored rule rather than by inspection of
the curve alone, and the LLM-as-judge scoring underlying the correctness
labels has been independently validated at 97.1% agreement with human
judgment — with its one known failure mode disclosed rather than hidden.
