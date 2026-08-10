"""LLM-as-judge correctness scoring for open-ended queries.

Uses the large Ollama tier as judge. Manual validation of judge verdicts
against human judgment (30-40 examples, Section 9) happens separately in
eval/validate_judge.py — do not skip that step before trusting these labels.
"""

from models.ollama_client import generate

JUDGE_PROMPT_TEMPLATE = """You are grading a candidate answer to a question, against a reference answer.

Question: {query}

Reference answer: {reference}

Candidate answer: {candidate}

Does the candidate answer correctly and substantively address the question in a way \
that is consistent with the reference answer's key facts and conclusion? Minor \
differences in wording, extra detail, or a different but factually equivalent \
explanation should still count as correct. Respond with exactly one word on the \
first line: CORRECT or INCORRECT. Optionally follow with a one-sentence justification \
on a second line.
"""


def judge(query_text: str, reference_answer: str, candidate_answer: str) -> tuple[bool, str]:
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        query=query_text, reference=reference_answer, candidate=candidate_answer
    )
    result = generate(prompt, tier="large", temperature=0.0)
    text = result.response_text.strip()
    first_line = text.splitlines()[0].strip().upper() if text else ""

    if "INCORRECT" in first_line:
        verdict = False
    elif "CORRECT" in first_line:
        verdict = True
    else:
        verdict = False  # unparseable judge output; treat conservatively as incorrect

    return verdict, text
