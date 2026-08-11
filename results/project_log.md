# Project Log — Cost-Aware LLM Routing & Evaluation System

Running log of what was done, why, what broke, and how it was fixed. Updated as
the project progresses — use this to reconstruct the reasoning behind any
design decision without re-deriving it from scratch.

---

## Objective

Route incoming queries to the cheapest LLM tier capable of answering them
correctly, and produce rigorous, defensible evidence for where the
accuracy/cost/latency trade-off actually sits — not a keyword-based demo.

---

## Phase 1 — Environment setup

**What:** Scaffolded the repo structure, built `models/ollama_client.py` as a
thin wrapper around the local Ollama HTTP API, pulled the two initial tier
models (`qwen2.5:1.5b-instruct-q4_0`, `qwen2.5:7b-instruct-q4_0`).

**Why:** Everything downstream depends on being able to reliably call both
tiers and measure latency/tokens.

**Result:** Verified both tiers respond correctly. First smoke test showed
8.45s (small) vs 13.19s (large) for the same prompt — confirmed a genuine
latency gap exists on this hardware (RTX 1650, 4GB VRAM, Tier 2 spills to
CPU/RAM as expected).

---

## Phase 2 — Dataset generation

**What:** Authored 350 candidate queries directly (not template-generated)
across three difficulty tiers — 120 easy / 115 medium / 115 hard — spanning 16
categories (general knowledge, geography, history, physics, chemistry,
biology, math, programming, logic puzzles, grammar, business/finance,
literature, everyday reasoning, philosophy/ethics, long-context analysis,
domain-specific expert questions). Each has a `query_id`, `category`,
`difficulty_label`, and `reference_answer`.

**Why:** The workflow spec calls for 300-400 queries across three genuine
difficulty tiers, with "hard" queries verified to actually be hard rather than
just longer versions of medium ones.

**Quality control while authoring:** caught and fixed two math errors before
finalizing (a binomial-coefficient calculation and a conditional-probability
marble problem). Verified zero duplicate query texts across all 350.

**Output:** `data/raw/queries_raw.jsonl` / `.csv`.

**Still open:** the workflow's required manual review of ~20-25% of the
dataset for correctness/difficulty-appropriateness (Section 5) has not been
done yet, nor has the reviewed subset been moved to `data/curated/`.

---

## Phase 3 — Router scaffolding

**What:** Built `router/rule_based.py` (keyword + length heuristic, quick
sanity-check baseline) and the real router pipeline: `router/features.py`
(sentence-transformer embeddings, `all-MiniLM-L6-v2`, 384-dim, CPU) and
`router/classifier.py` (embeddings + LogisticRegression, code-complete but
untrained — there was no labeled data yet).

**Why:** Section 8 requires exactly these two router variants, with the
router outputting `P(large_model_required)` rather than a hard decision.

---

## Phase 4/5 — Batch inference pipeline + pilot run

This is where most of the real difficulty in the project showed up.

**What:** Built `eval/correctness.py` (exact-match heuristic, for reference
answers with a short, objective core fact) and `eval/llm_judge.py`
(LLM-as-judge, for open-ended queries where exact-match doesn't apply).
Built `pipeline/batch_runner.py` — the portable runner that calls both tiers,
scores correctness, and derives `routing_label` per Section 6's logic
(small correct → "small"; small wrong but large right → "large"; both wrong →
excluded from training). Ran a 30-query stratified pilot (10 per difficulty
tier) locally, per the workflow's decision rule to test timing before
committing to a full run.

### Difficulty 1 — near-degenerate routing split

First pilot run: 26/30 (86.7%) of queries routed to "small", only 1 to
"large" — right at the edge of the workflow doc's own "not >90% one-sided"
degeneracy check. Even hard-tier conceptual questions (photoelectric effect,
corporate-veil doctrine) were being graded "small model correct."

### Difficulty 2 — exact-match scorer bugs

Investigating the above surfaced two real bugs in `eval/correctness.py`:

- **Bug A (false negative):** core-answer extraction stopped before a
  LaTeX-formatted correct answer (e.g. a rectangle area/perimeter problem
  where the model wrote `\text{cm}^2`) — a genuinely correct small-model
  answer was marked wrong. Fixed by trying a literal phrase match first, then
  falling back to a numeric match regardless of the extracted core's word
  count.
- **Bug B (false positive, introduced by fixing A):** that same fix let a
  long, multi-step procedural reference answer (an 8-coin weighing puzzle)
  match on a coincidental short number ("3", from "divide into groups of 3")
  even when the full multi-step solution was wrong. Fixed with a length guard
  — reference answers over 20 words defer to LLM-judge instead of attempting
  number-matching at all.

### Difficulty 3 — LLM-judge leniency

Deep-dived one specific case: the small model's answer to the coin-puzzle was
internally self-contradictory (claimed "8 coins split into three groups of 3"
— which sums to 9, not 8 — and later tried to split 3 coins into two groups
of 2). Genuinely wrong. But the judge (at the time, the same 7B model used as
Tier 2) rated it **CORRECT**.

**Diagnosis:** the judge was pattern-matching surface similarity to the
reference's general approach ("groups of 3") rather than verifying
arithmetic/logical correctness — and using the same model as both an
answering tier and the judge risked leniency toward its own reasoning style.

**Fix:** introduced a dedicated `JUDGE_MODEL` (`qwen2.5:14b`, already
available locally) used only for grading, never as a routing candidate, plus
a stricter step-by-step verification prompt requiring the judge to check each
claim/calculation for internal consistency before ruling CORRECT.

**Verified the fix directly:** re-ran the same coin-puzzle case through the
new judge — it correctly caught the arithmetic contradiction and returned
INCORRECT. Re-scored the whole pilot with the new judge: split improved to
26 small / 3 large / 1 excluded (90% skew) — better, but the signal was still
thin, and notably 0/10 hard-tier pilot queries still needed the large model
even with the scoring fixes in place.

### Difficulty 4 — the real root cause was model capability, not scoring

User's insight: `qwen2.5:1.5b-instruct` might just be a genuinely strong
small model for this task mix (both general-knowledge recall and moderate
reasoning), not a scoring artifact. Confirmed by inspection — its answers to
several "hard" conceptual questions were substantively correct, not
judge-leniency false positives.

**Decision:** keep Tier 2 (7B) fixed, since that preserves the core routing
comparison the whole project studies. Move Tier 1 down to
`qwen2.5:0.5b-instruct-q4_0`, staying within the Qwen2.5 family so parameter
count — not model family, tokenizer, or training-data differences — is the
isolated variable driving the capability gap.

**Re-ran the 30-query pilot with 0.5B:** split became 14 small / 15 large /
1 excluded (46.7% / 50% / 3.3%) — healthy, and for the first time correlated
sensibly with difficulty tier: easy 9 small / 0 large, medium 4 small /
6 large, hard 1 small / 9 large.

### Difficulty 5 — efficiency: judge calls dominate wall-clock cost

The accuracy fix (14B judge + stricter, longer prompt) made the pipeline
meaningfully slower — measured ~62s for a single combined judge call on this
hardware. Extrapolated the full 350-query run to roughly 6-7 hours locally.

Tested whether the stricter *prompt* alone (without the model swap) run on
the faster 7B model would have caught the same coin-puzzle error — **it did
not** (still returned CORRECT), confirming the model upgrade itself, not just
the prompt wording, was the real fix. Speed could not be recovered by
reverting the judge to 7B without reintroducing the leniency bug.

**Optimization applied instead:** combined the small+large judge calls into a
single `judge_pair()` call (grades both candidates in one prompt) instead of
two separate calls, roughly halving judge-related latency for queries that
need it. Capped judge output length (`num_predict=150`) to bound worst-case
per-call latency.

### Decision — move Phase 6 (full run) to Kaggle

Per the workflow doc's own decision rule ("only move to Kaggle if the pilot
shows the full run would take unreasonably long locally"), and the realistic
~6-7 hour local estimate, moved the full run to Kaggle's free GPU (P100).

Built `notebooks/kaggle_batch_run.ipynb` and verified programmatically that
the embedded module files (`models/ollama_client.py`, `eval/correctness.py`,
`eval/llm_judge.py`, `pipeline/batch_runner.py`) are byte-identical to the
local versions — satisfying the doc's "same code, no separate versions"
requirement rather than a Kaggle-specific rewrite.

### Difficulty 6 — Kaggle environment issues

- **First failure:** `FileNotFoundError: 'ollama'` when trying to start the
  server. Root cause: the Ollama installer needs `zstd` to extract its
  archive, and Kaggle's base image doesn't have it. Fixed by installing
  `zstd` via `apt-get` before the Ollama install step, plus added a
  diagnostic cell that explicitly verifies the binary exists — with an
  actionable error message pointing at the notebook's Internet toggle —
  before attempting to start the server.
- **Second issue:** the run looked "stuck" for 4+ minutes with no new log
  output. Root cause: output buffering — `python -m pipeline.batch_runner`
  run via `subprocess.run()` wasn't attached to a real terminal, so its
  `print()` calls were block-buffered instead of flushed per line, and there
  was no partial-progress file to check against in the meantime. Fixed by
  (a) running the subprocess with `-u` (unbuffered) and streaming its stdout
  line-by-line into the notebook cell in real time, and (b) adding
  checkpointing to `pipeline/batch_runner.py` — results are now saved to the
  output CSV every 10 queries instead of only once at the very end.

### Phase 6 result

Full 350-query batch completed on Kaggle P100 in ~2.1 hours (309/350 done at
~111 minutes elapsed; extrapolated ETA at that point matched the actual
finish time closely).

Final distribution: **149 small (42.6%) / 183 large (52.3%) / 18 excluded as
both-incorrect (5.1%)**. By difficulty: easy 91 small / 20 large / 9 excluded;
medium 41 small / 72 large / 2 excluded; hard 17 small / **91 large** /
7 excluded — now correlates sensibly with intended difficulty (hard tier
needs the large model 79% of the time, vs. effectively 0% before the Tier 1
model swap). Validated on arrival: no duplicate or missing rows.

---

## Phase 7 (in progress) — Router training

**What:** Trained `router/classifier.py`'s embeddings
(`all-MiniLM-L6-v2`) + LogisticRegression classifier on the 332 labeled
examples (the 18 both-incorrect rows are excluded from training per Section
6), using an 80/20 stratified train/test split.

**Result — held-out test accuracy: 71.6%** (test set: 30 "small", 37
"large").

| class | precision | recall | f1 |
|---|---|---|---|
| small | 0.70 | 0.63 | 0.67 |
| large | 0.73 | 0.78 | 0.75 |

For context, always predicting the majority class ("large", 37/67 = 55.2% of
the test set) would give 55.2% accuracy — the trained router is +16.4 points
over that naive baseline, meaning it's learning real signal from the query
text, not just exploiting class imbalance.

**Refactor:** pulled the train/test split out of `train()` into a shared
`split_labeled_data()` function in `router/classifier.py`, so
`pipeline/baselines.py` (below) evaluates every system on the identical
held-out rows the router itself was scored on — no system gets an unfair
data-leakage advantage from a different split.

---

## Phase 7 continued — Baseline A/B/C comparison, and a real data-quality bug

**What:** Built `pipeline/baselines.py`, evaluating four systems on the same
67-row held-out test split: Baseline A (always small), Baseline B (always
large), the rule-based router (evaluated against the real dataset for the
first time — previously only smoke-tested on 4 made-up examples), and the
trained embeddings+LogReg router. Accuracy/latency come directly from the
real per-tier answers already recorded in `full_run.csv` — no new model calls
needed for this comparison itself.

### Difficulty 7 — unbounded small-tier generation

First run of the baselines surfaced a real bug, not just a modeling result.
Baseline B (always-large) scored a suspicious 100% on the test set, which led
to checking `large_model_correct` across all 332 labeled rows: **99.1%**
correct — so 100% on a 67-row sample isn't a fluke, it's expected given how
`routing_label` is constructed (rows are only excluded when *both* tiers
fail, and the 7B model is almost always the one that succeeds). This also
surfaced 3 genuine "small correct, large incorrect" edge cases — the rare
finding Section 6 explicitly asks to flag.

Separately, the latency distribution looked wrong: small-tier mean latency
(8.7s) was actually *higher* than large-tier mean (7.4s), despite small's
median being far lower (2.1s vs 6.9s) — a classic sign of extreme outliers
dragging the mean up. Investigation found 8 of 332 rows where
`small_model_completion_tokens` was exactly 40960 and latency exceeded 260s.
Reading one of the actual responses (the photoelectric-effect query)
confirmed it: a 229,787-character wall of rambling text that never emitted a
stop token, because tier-answer generation calls had no `num_predict` cap
(only the judge calls did).

**Fix:** added a `DEFAULT_MAX_TOKENS = 1024` cap to
`models.ollama_client.generate()` (generous relative to every reference
answer in the dataset, but bounds worst-case pathological runs). Rather than
re-running the full 350-query batch, patched just the 8 affected rows locally
— regenerated the small-tier answer, re-scored correctness, and re-derived
`routing_label` for each. Latencies came back realistic (3.8-8.0s, matching
the rest of the dataset), and one row (`q0166`) actually flipped from
incorrect to **correct** once the response was properly bounded — the
original 229KB of noise was likely burying the actual answer or confusing the
judge, so this was a genuine data-quality improvement, not just a latency
fix. Regenerated `notebooks/kaggle_batch_run.ipynb` afterward and re-verified
it's byte-identical to the patched local modules.

### Result (after the patch)

| System | Accuracy | Avg latency | % routed small |
|---|---|---|---|
| A — always small | 44.8% | 2.58s | 100% |
| B — always large | 100.0% | 7.98s | 0% |
| C — rule-based router | 82.1% | 6.58s | 49.3% |
| C — embeddings+LogReg router | 86.6% | 6.93s | 41.8% |

Both routers land well above Baseline A while routing 40-50% of queries to
the cheap tier — the core value proposition of the project — with the
trained LogReg router beating the keyword heuristic by ~4.5 points while
routing *more* conservatively (fewer to small, but more accurately).

Before this patch, the same comparison showed Baseline A at a misleadingly
high 6.34s average latency (inflated by the runaway-generation rows) — worth
remembering when reading anything computed before this fix.

---

## Phase 8 — Simulated cost methodology + threshold sweep

**What:** Built `pipeline/threshold_sweep.py`, covering both Section 2's
simulated-cost requirement and Section 10's threshold sweep in one pass.

**Cost methodology (disclosed per Section 2):** Ollama inference is free and
local, so "cost" is *simulated* by mapping each tier's actual recorded
prompt/completion token counts onto published per-token pricing of a
comparably-sized commercial API tier — a cheap-tier vs. premium-tier split
loosely modeled on public pricing for small vs. flagship hosted models
($0.15/$0.60 per million input/output tokens for the small tier, $2.50/$10.00
for the large tier). This is a stated methodology for producing a comparable
cost signal, not real spend.

**Sweep:** the trained embeddings+LogReg router's `P(large_model_required)`
was swept across the required thresholds (0.50-0.85) on the same held-out
test split used throughout Phase 7, with Baseline A, Baseline B, and the
rule-based router plotted as fixed reference lines (they don't have a
threshold to sweep). Four plots generated in `results/plots/`:
`accuracy_vs_threshold.png`, `cost_vs_threshold.png`,
`latency_vs_threshold.png`, `accuracy_vs_cost_savings.png`.

**Result table** (`data/results/threshold_sweep.csv`):

| threshold | accuracy | % to small | avg latency | latency saved vs B | avg cost | cost saved vs B |
|---|---|---|---|---|---|---|
| 0.50 | 86.6% | 41.8% | 6.93s | 13.2% | $0.00268 | 21.6% |
| 0.55 | 85.1% | 47.8% | 6.88s | 13.9% | $0.00264 | 23.1% |
| **0.60** | **80.6%** | **56.7%** | **6.09s** | **23.7%** | **$0.00221** | **35.5%** |
| 0.65 | 74.6% | 67.2% | 5.41s | 32.3% | $0.00180 | 47.5% |
| 0.70 | 62.7% | 80.6% | 4.50s | 43.6% | $0.00128 | 62.8% |
| 0.75 | 56.7% | 86.6% | 3.91s | 51.0% | $0.00093 | 72.5% |
| 0.80 | 50.8% | 94.0% | 3.20s | 60.0% | $0.00057 | 83.7% |
| 0.85 | 47.8% | 97.0% | 2.84s | 64.5% | $0.00036 | 89.3% |

(For reference: Baseline A = 44.8% accuracy / $0.00021 cost; Baseline B =
100% accuracy / $0.00342 cost — B's accuracy is close to the true large-tier
ceiling of ~99.1% measured across the full labeled set, not a fluke of this
particular 67-row test split; rule-based router = 82.1% accuracy / $0.00251
cost / 49.3% routed small.)

**Recommended operating point: threshold = 0.60.** Justification: computing
the marginal accuracy-lost-per-%-cost-saved ratio between each consecutive
pair of thresholds shows the trade stays favorable (roughly 0.4-0.5 points of
accuracy per point of cost savings) through the 0.55→0.65 range, then clearly
worsens at 0.65→0.70 (0.78) as accuracy craters from 74.6% down through the
50s. At t=0.60 specifically: accuracy (80.6%) sits within 1.5 points of the
rule-based baseline (82.1%) while delivering *more* cost savings than it
(35.5% vs. ~26.5%) — i.e., the trained router beats the simple heuristic on
the metric it's supposed to improve (cost), at a comparable accuracy, rather
than just trading one for the other. This is visible directly in
`accuracy_vs_cost_savings.png` as the point just past where the curve's slope
visibly steepens.

---

## Phase 8 follow-up — why does the trained router show *higher* latency than the rule-based router, and did this project actually work?

**The question:** `pipeline/baselines.py`'s comparison table shows the
trained router at 6.93s average latency, *higher* than the rule-based
router's 6.58s, even though the trained router has better accuracy (86.6% vs
82.1%). On its face this looks like a weak or confusing result to bring to an
interview.

**Root cause:** `pipeline/baselines.py` evaluates the trained router at its
untuned default classification threshold (0.5) — never swept, never chosen
for anything. At 0.5 the router sends only 41.8% of queries to the small
tier, vs. the rule-based router's 49.3%. Sending more traffic to the slower
large tier is *why* both its accuracy and its latency are higher — it's one
mechanical consequence of the routing mix, not two separate findings, and not
a flaw in the model.

**Is it a good result?** Only if presented with the full picture, not that
one table in isolation. In isolation, "the smarter system is slower" reads
as a weakness. In context — the threshold sweep already built in Phase 8 —
it's the opposite: the rule-based router is stuck at one fixed operating
point forever, while the trained router's probability output can be tuned
along the whole accuracy/cost/latency frontier. At the untuned default it
leans toward accuracy; at the recommended threshold (0.60) it flips to
beating rule-based on both cost (35.5% vs ~26.5% savings) and latency (6.09s
vs 6.58s), giving up only 1.5 points of accuracy to do it. There is no single
threshold in this sweep where the trained router strictly dominates
rule-based on all three metrics simultaneously — claiming otherwise would be
overclaiming. The defensible story is "tunable trade-off control vs. a fixed
heuristic," not "ours is just better across the board."

**Known inconsistency to fix:** `pipeline/baselines.py` (0.5, untuned) and
`pipeline/threshold_sweep.py` (recommends 0.60) currently report two
different numbers for "the router" with nothing reconciling them in the
output. Should be fixed before the write-up is finalized — either have the
baselines table use the recommended threshold, or explicitly label its 0.5
row as "default, untuned."

**Bottom line — did this project achieve something real?** Yes, with real
numbers behind it, and with honest limits stated:

- At the recommended threshold (0.60), the system recovers **80.6%** of
  Baseline B's accuracy (vs. Baseline A's 44.8%) while cutting simulated cost
  by **35.5%** and latency by **23.7%** relative to always using the large
  model. Framed the other way: going from "always cheap" to this system
  recovers +35.8 accuracy points over Baseline A while still capturing over
  a third of the possible cost savings.
- The *learned* router (embeddings + LogisticRegression) is not just
  placebo ML — at a comparable operating point it genuinely beats the
  keyword-heuristic baseline on the metric it's supposed to improve (cost),
  not just on accuracy. That's the evidence that the embedding features are
  carrying real signal about query difficulty, not merely mimicking what a
  human-written keyword rule would already catch.
- Honest caveats: the held-out test set is only 67 queries, so these
  percentages carry real sampling noise (visible in some of the sweep's
  jumpy, quantized-looking steps); "cost" is simulated from assumed API
  pricing ratios, not real spend; and the LLM-judge manual validation
  (Section 9) — which matters more here than usual, since we already caught
  one real judge-leniency bug — still hasn't been done, so some of the
  underlying correctness labels haven't been human-checked yet.

---

## Phase 7 revisited — trying more complex classifiers for the router

**Prompted by:** "can we use a more advanced ML algorithm for better
results, and get lower latency and higher accuracy?"

**Clarified first:** router inference itself (embed + classify) runs in
milliseconds regardless of algorithm — LLM generation time dominates every
latency number in this project. A better classifier can't directly reduce
latency; it can only raise accuracy at a given routing rate, which
*indirectly* lets more traffic shift to the cheap tier at the same accuracy,
moving the whole trade-off frontier outward rather than picking a different
point on the existing one.

**What:** built `router/model_comparison.py`, testing LogisticRegression
(default and CV-tuned), Random Forest, Gradient Boosting, SVM (RBF kernel),
and a small MLP — all on the identical held-out split used throughout the
project, with hyperparameters chosen via 5-fold CV on the training set only
(never touching the test set).

**A metric subtlety surfaced along the way:** there are two different
"accuracy" numbers in this project that can diverge — *classifier accuracy*
(predicted tier vs. ground-truth `routing_label`) vs. *system accuracy* (is
the actually-delivered answer correct, given whichever tier got chosen).
They diverge because misrouting isn't symmetric: since the large tier is
correct ~99.1% of the time on this dataset, routing a "should-be-small"
query to "large" anyway still usually produces a correct answer (wastes cost
but not correctness), while routing a "should-be-large" query to "small"
usually produces a wrong one. `model_comparison.py` reports both, but ranks
models by system accuracy since that's what the project actually optimizes
for.

**Result** (system accuracy, at each model's default 0.5 decision
threshold):

| model | system accuracy | avg latency | avg cost | % to small |
|---|---|---|---|---|
| Random Forest | 89.6% | 7.29s | $0.00292 | 32.8% |
| **SVM (RBF)** | **88.1%** | **6.91s** | **$0.00268** | 43.3% |
| Gradient Boosting | 86.6% | 7.24s | $0.00288 | 38.8% |
| LogisticRegression (previous default) | 86.6% | 6.93s | $0.00269 | 41.8% |
| LogisticRegression (CV-tuned) | 85.1% | 6.89s | $0.00264 | 46.3% |
| MLP (neural net) | 79.1% | 6.46s | $0.00240 | 52.2% |

SVM (RBF) matched or beat LogisticRegression on every axis simultaneously —
higher accuracy, marginally lower latency and cost, while routing *more*
traffic to the cheap tier. Random Forest scores even higher on raw accuracy
but gets there by being more conservative (routes less to small, costs
more) — a different frontier point, not a strict improvement. The MLP did
worse across the board, consistent with the overfitting risk expected from
only ~265 training examples for a model class that typically needs far more
data. 5-fold CV scores (computed on more data, less noisy than the 67-row
test set) tell a closer story — Random Forest and Gradient Boosting edge
slightly ahead of SVM there — so this ranking should be read directionally,
not as a precise ordering.

**Decision:** switched `router/classifier.py`'s default estimator from
LogisticRegression to SVM (RBF, C=1, gamma='scale', class_weight='balanced')
— the cleanest win with no shown downside. LogisticRegression wasn't
deleted, just no longer the default (`make_estimator()` is a single,
easy-to-change spot). Renamed references from "embeddings+LogReg router" to
"embeddings+SVM router" throughout `pipeline/baselines.py` and
`pipeline/threshold_sweep.py` to keep labels accurate.

**A real gotcha found while re-running the sweep:** `pipeline/baselines.py`
reported 91.0% for the new SVM router, not the 88.1% `model_comparison.py`
had measured. Root cause: SVM's `probability=True` uses Platt scaling
(sigmoid calibration via internal CV) to produce `predict_proba()` output,
and it's a documented scikit-learn quirk that thresholding `predict_proba()`
at 0.5 doesn't always exactly match `predict()`'s raw decision boundary.
`model_comparison.py` used `predict()` directly; `baselines.py` and the
threshold sweep necessarily use `predict_proba()`, since the whole point is
sweeping a continuous threshold. Checked that the sweep is still smooth and
monotonic across thresholds (it is) before trusting it — the calibration
quirk shifts the single default-threshold number but doesn't break the
sweep's overall shape. Worth remembering: this is one real (mild) cost of
choosing SVM over LogisticRegression, which has no such predict/predict_proba
mismatch since its probabilities come directly from its sigmoid link
function rather than a separate calibration step.

**Updated sweep with the SVM router** (`data/results/threshold_sweep.csv`):

| threshold | accuracy | % to small | avg latency | avg cost | cost saved vs B |
|---|---|---|---|---|---|
| 0.50 | 91.0% | 38.8% | 7.10s | $0.00280 | 18.2% |
| 0.55 | 88.1% | 43.3% | 6.91s | $0.00268 | 21.8% |
| 0.60 | 86.6% | 47.8% | 6.59s | $0.00248 | 27.4% |
| **0.65** | **82.1%** | **53.7%** | **6.19s** | **$0.00226** | **34.1%** |
| 0.70 | 76.1% | 61.2% | 5.66s | $0.00194 | 43.2% |
| 0.75 | 74.6% | 65.7% | 5.42s | $0.00181 | 47.0% |
| 0.80 | 65.7% | 74.6% | 4.95s | $0.00152 | 55.5% |
| 0.85 | 58.2% | 83.6% | 4.02s | $0.00103 | 69.9% |

**Recommended operating point updated: threshold = 0.65** (was 0.60 under
the old LogisticRegression router). Justification: at t=0.65 the SVM router's
accuracy (82.1%) is *exactly equal to* the rule-based router's flat accuracy
(82.1%), while its cost savings (34.1%) clearly beat the rule-based router's
(~26.5%) — genuine dominance, not a trade-off, at this specific point. Past
t=0.65, accuracy drops below the rule-based baseline (76.1% at t=0.70) while
the router no longer offers a "matches-or-beats the simple heuristic"
guarantee. The selection rule this generalizes to: pick the largest
threshold (most savings) subject to the constraint that accuracy still
matches or exceeds the rule-based baseline it's meant to replace — a cleaner,
more externally-anchored justification than reading the curve's slope alone.
All four plots in `results/plots/` regenerated to reflect the SVM router and
the new recommended point.

---

## Still open / not yet done

- Manual review of ~20-25% of the dataset for correctness/difficulty
  (Section 5) — not done.
- Moving the reviewed subset to `data/curated/` — not done.
- LLM-judge manual validation, 30-40 examples (Section 9) — not done. This
  matters more than usual here, since we already found and fixed one real
  judge-leniency failure during the pilot; the formal validation pass hasn't
  happened yet.
- `pipeline/baselines.py` still reports the SVM router at its untuned
  default (0.5) threshold, not the recommended 0.65 — same "two different
  unlabeled numbers for the router" gap as before, now against updated
  numbers. Should be labeled clearly in the write-up.
- Write-up (`results/writeup.md`), demo wrapper, README — not done.

## Done since the above sections were written

- Rule-based router evaluated against the real dataset — 82.1% accuracy on
  the held-out test split (see Phase 7 continued, above).
- Baseline A/B/C comparison on a held-out split (Section 7) — done, see table
  above.
- Simulated cost methodology (Section 2) and threshold sweep + trade-off
  plots + chosen operating point (Section 10) — done, see Phase 8 above.
