"""Heuristic router: quick sanity-check baseline, not the production router.

Scores a query by keyword presence and length as a rough proxy for
P(large_model_required). Section 8 in the workflow doc scopes this to
~1 hour of work; the embeddings + LogisticRegression router in
classifier.py is the real system.
"""

HARD_KEYWORDS = [
    "prove", "derive", "derivative", "integral", "eigenvalue", "algorithm",
    "write a function", "write a python", "write a sql", "debug", "time complexity",
    "big-o", "big o", "recursion", "explain why", "explain how", "analyze",
    "compare and contrast", "discuss", "trade-off", "tradeoff", "mechanism",
    "differential equation", "npv", "irr", "ethical", "philosoph", "derive an expression",
]

MEDIUM_KEYWORDS = ["why", "how", "explain", "calculate", "solve", "if", "suppose"]


def _keyword_score(text: str) -> float:
    lower = text.lower()
    score = sum(1.0 for kw in HARD_KEYWORDS if kw in lower)
    score += sum(0.3 for kw in MEDIUM_KEYWORDS if kw in lower)
    return score


def _length_score(text: str) -> float:
    n_words = len(text.split())
    return min(n_words / 60.0, 1.0)


def score(query_text: str) -> float:
    """Heuristic proxy for P(large_model_required), clamped to [0, 1]."""
    s = 0.5 * _keyword_score(query_text) + 0.5 * _length_score(query_text)
    return min(s, 1.0)


def route(query_text: str, threshold: float = 0.3) -> str:
    return "large" if score(query_text) >= threshold else "small"


if __name__ == "__main__":
    examples = [
        "What is the capital of France?",
        "Explain the difference between precision and recall.",
        "Derive the backpropagation equations for a two-layer neural network.",
        "Write a Python function to detect a cycle in a linked list using Floyd's algorithm.",
    ]
    for q in examples:
        print(f"P(large)={score(q):.2f}  route={route(q):5s}  {q}")
