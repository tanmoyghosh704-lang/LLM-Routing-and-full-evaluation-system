"""LLM-as-judge correctness scoring for open-ended queries.

Uses a dedicated judge model (JUDGE_MODEL in models/ollama_client.py) that is
distinct from both routing tiers, so grading isn't done by the same model
that produced one of the candidate answers. Manual validation of judge
verdicts against human judgment (30-40 examples, Section 9) still happens
separately in eval/validate_judge.py — do not skip that step before fully
trusting these labels.

judge_pair() grades both the small- and large-tier candidate answers in a
single call (instead of two separate judge() calls) to roughly halve judge
latency for queries where neither answer is exact-match gradable.
"""

import re

from models.ollama_client import JUDGE_MODEL, generate_with_model

_VERIFICATION_INSTRUCTIONS = """Check each candidate answer step by step:
1. Identify each factual claim, calculation, or step in the candidate answer.
2. Verify each one is actually correct and internally consistent (e.g. if it says a
   quantity is divided into groups, the group sizes must actually sum to the total;
   if it does arithmetic, the arithmetic must be right).
3. Confirm the candidate's final conclusion matches the reference answer's key facts
   and conclusion. Minor differences in wording or extra detail are fine as long as
   the substance and every material step are correct — but any factual, logical, or
   arithmetic error, or a self-contradictory step, makes the answer INCORRECT, even
   if the surface-level approach resembles the reference."""

JUDGE_PROMPT_TEMPLATE = """You are carefully grading a candidate answer to a question, against a reference answer.

Question: {query}

Reference answer: {reference}

Candidate answer: {candidate}

{instructions}

Respond with exactly one word on the first line: CORRECT or INCORRECT. Follow with a
one-sentence justification on a second line, citing the specific step that is right
or wrong.
""".replace("{instructions}", _VERIFICATION_INSTRUCTIONS)

JUDGE_PAIR_PROMPT_TEMPLATE = """You are carefully grading two independent candidate answers to a question, against a reference answer.

Question: {query}

Reference answer: {reference}

Candidate A: {candidate_a}

Candidate B: {candidate_b}

{instructions}

Grade Candidate A and Candidate B independently of each other. Respond in exactly
two lines, in this format:
A: CORRECT or INCORRECT
B: CORRECT or INCORRECT
""".replace("{instructions}", _VERIFICATION_INSTRUCTIONS)

_JUDGE_NUM_PREDICT = 150


def _parse_verdict(text: str) -> bool:
    first_line = text.splitlines()[0].strip().upper() if text else ""
    if "INCORRECT" in first_line:
        return False
    if "CORRECT" in first_line:
        return True
    return False  # unparseable judge output; treat conservatively as incorrect


def _parse_pair_verdict(text: str, label: str) -> bool:
    match = re.search(rf"^\s*{label}\s*[:\-]\s*(CORRECT|INCORRECT)", text, re.IGNORECASE | re.MULTILINE)
    if not match:
        return False  # unparseable; treat conservatively as incorrect
    return match.group(1).upper() == "CORRECT"


def judge(query_text: str, reference_answer: str, candidate_answer: str) -> tuple[bool, str]:
    prompt = JUDGE_PROMPT_TEMPLATE.format(query=query_text, reference=reference_answer, candidate=candidate_answer)
    result = generate_with_model(prompt, JUDGE_MODEL, temperature=0.0, num_predict=_JUDGE_NUM_PREDICT)
    text = result.response_text.strip()
    return _parse_verdict(text), text


def judge_pair(
    query_text: str, reference_answer: str, candidate_a: str, candidate_b: str
) -> tuple[bool, bool, str]:
    """Grade two candidate answers in a single judge call. Returns (verdict_a, verdict_b, raw_text)."""
    prompt = JUDGE_PAIR_PROMPT_TEMPLATE.format(
        query=query_text, reference=reference_answer, candidate_a=candidate_a, candidate_b=candidate_b
    )
    result = generate_with_model(prompt, JUDGE_MODEL, temperature=0.0, num_predict=_JUDGE_NUM_PREDICT)
    text = result.response_text.strip()
    return _parse_pair_verdict(text, "A"), _parse_pair_verdict(text, "B"), text
