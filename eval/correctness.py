"""Exact-match/rubric correctness scoring for objective, short-answer queries.

Only handles reference answers with a short "core" fact (a number, name, or
short phrase before any explanatory clause) — e.g. "Au", "1945", "150 miles
(60 x 2.5)" -> core "150 miles". Longer, open-ended reference answers (most
medium/hard queries) are not gradable this way; try_exact_match returns None
for those so the caller can fall back to eval/llm_judge.py.
"""

import re

_DELIM_RE = re.compile(r"[.,;(]|\bsince\b|\bbecause\b|\bwhich\b")
_NUM_RE = re.compile(r"-?\d+\.?\d*")


def _core_answer(reference_answer: str) -> str:
    match = _DELIM_RE.search(reference_answer)
    core = reference_answer[: match.start()] if match else reference_answer
    return core.strip()


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9.\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def try_exact_match(model_answer: str, reference_answer: str) -> bool | None:
    """Return True/False if confidently gradable by short-answer matching, else None."""
    # Guard on the *full* reference length, not just the extracted core: a
    # long multi-sentence procedural reference (e.g. a puzzle solution) can
    # have a short, coincidentally-numeric fragment before its first comma,
    # which would otherwise let a trivial number match (e.g. "3" from
    # "divide into groups of 3") falsely credit an unrelated or wrong answer.
    if len(_normalize(reference_answer).split()) > 20:
        return None

    core_norm = _normalize(_core_answer(reference_answer))
    words = core_norm.split()
    if not words or len(words) > 8:
        return None

    model_norm = _normalize(model_answer)

    # Try a literal phrase match first, but don't rely on it alone: model
    # answers often reformat numbers/units (e.g. LaTeX "\text{cm}^2" vs plain
    # "cm^2"), breaking a contiguous phrase match even when the underlying
    # answer is correct. Numeric answers get a fallback check regardless of
    # how many words the extracted core phrase has.
    pattern = r"\b" + re.escape(core_norm) + r"\b"
    if re.search(pattern, model_norm):
        return True

    core_nums = _NUM_RE.findall(core_norm)
    if core_nums and core_nums[0] in _NUM_RE.findall(model_norm):
        return True

    return False
