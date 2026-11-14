"""Answer metrics (correctness, citations, abstention), paired tests and reliability estimators."""
from __future__ import annotations

import math
import re
import string
from collections import Counter
import unicodedata

from .schema import Prediction, Question

_PUNCT = set(string.punctuation)
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_NUMBER = re.compile(r"[-+]?(?:\d+(?:,\d{3})*(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


# ----------------------------------------------------------------------------- correctness
def normalize_answer(s: str) -> str:
    """SQuAD normalisation: lower case, drop punctuation, drop articles, collapse spaces."""
    s = unicodedata.normalize("NFKC", s).lower()     # PDF text: ligatures (ﬁ -> fi), full-width chars
    s = "".join(ch for ch in s                       # ASCII punctuation (SQuAD) + Unicode punctuation (’ – “ ”)
                if ch not in _PUNCT and not unicodedata.category(ch).startswith("P"))
    return " ".join(_ARTICLES.sub(" ", s).split())


def exact_match(pred: str, golds: list[str]) -> float:
    """1.0 if the normalised prediction equals one of the normalised gold aliases."""
    p = normalize_answer(pred)
    return float(any(p == normalize_answer(g) for g in golds))   # 0.0 if there is no alias


def token_f1(pred: str, gold: str) -> float:
    """SQuAD token F1 between normalised strings (a diagnostic, NOT a correctness criterion)."""
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    common = sum((Counter(p) & Counter(g)).values())     # multiset intersection
    if common == 0:                                      # covers the empty strings too
        return 0.0
    precision, recall = common / len(p), common / len(g)
    return 2 * precision * recall / (precision + recall)


def parse_number(s: str) -> float | None:
    """First number of s: handles 1,234.5 / -0.5 / .5 / 1.2e-3 / the unicode minus sign."""
    s = s.replace("\u2212", "-")
    m = _NUMBER.search(s)
    if m is None:
        return None
    text = m.group().replace(",", "")                    # thousands separators
    if text[0] in "+-" and m.start() > 0 and s[m.start() - 1].isalnum():
        text = text[1:]                                  # a hyphen, not a sign: ResNet-50, GPT-2
    return float(text)                                   # every match of _NUMBER is a valid float



def numeric_match(pred: str, gold: str, rel_tol: float = 1e-2, abs_tol: float = 1e-9) -> bool:
    a, b = parse_number(pred), parse_number(gold)
    return a is not None and b is not None and math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


def is_correct(pred: Prediction, q: Question) -> bool:
    """Unanswerable: correct iff abstain. Answerable: abstaining is wrong; numbers within
    rel_tol of an alias; text: exact match with an alias."""
    if not q.answerable:
        return pred.abstain
    if pred.abstain:
        return False
    if q.answer_type == "number":
        return any(numeric_match(pred.answer, g) for g in q.answers)
    return exact_match(pred.answer, q.answers) == 1.0



# ----------------------------------------------------------------------------- citations
def citation_precision(pred: Prediction, q: Question) -> float:
    """Fraction of the cited pages that are gold evidence (nan if nothing is cited)."""
    cited = set(pred.citations)
    return len(cited & set(q.evidence)) / len(cited) if cited else math.nan


def citation_recall(pred: Prediction, q: Question) -> float:
    """Fraction of the gold pages that are cited (nan if the question has no evidence)."""
    return len(set(pred.citations) & set(q.evidence)) / len(q.evidence) if q.evidence else math.nan


def evidence_consulted(pred: Prediction, q: Question) -> float:
    """Fraction of the gold pages the system SAW (retrieved, or returned by a tool)."""
    return len(set(pred.consulted) & set(q.evidence)) / len(q.evidence) if q.evidence else math.nan


def score(pred: Prediction, q: Question) -> dict:
    return {"qid": q.qid, "category": q.category, "answerable": q.answerable,
            "correct": float(is_correct(pred, q)), "abstain": float(pred.abstain),
            "em": exact_match(pred.answer, q.answers) if q.answers else math.nan,
            "f1": max((token_f1(pred.answer, g) for g in q.answers), default=math.nan),
            "cit_precision": citation_precision(pred, q), "cit_recall": citation_recall(pred, q),
            "evidence_consulted": evidence_consulted(pred, q),
            "n_llm_calls": pred.n_llm_calls, "tokens": pred.prompt_tokens + pred.completion_tokens,
            "latency_s": pred.latency_s}


# ----------------------------------------------------------------------------- tests
def mcnemar_exact(a: list[bool], b: list[bool]) -> tuple[int, int, float]:
    """Exact two-sided McNemar test on paired outcomes (same questions).

    Returns (n01, n10, p): n01 = a right & b wrong, n10 = a wrong & b right.
    Under H0, n01 | (n01 + n10) ~ Binomial(n01 + n10, 1/2).
    """
    if len(a) != len(b):
        raise ValueError("paired test: a and b must cover the same questions")
    n01 = sum(1 for x, y in zip(a, b) if x and not y)
    n10 = sum(1 for x, y in zip(a, b) if y and not x)
    n = n01 + n10                                        # only the discordant questions carry information
    if n == 0:
        return n01, n10, 1.0
    tail = sum(math.comb(n, i) for i in range(min(n01, n10) + 1)) / 2 ** n   # exact, in integers
    return n01, n10, min(1.0, 2 * tail)


def holm(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Holm-Bonferroni step-down: which hypotheses are rejected, family-wise error <= alpha."""
    m = len(pvalues)
    rejected = dict.fromkeys(pvalues, False)
    for i, name in enumerate(sorted(pvalues, key=pvalues.get)):  # smallest p-value first
        if pvalues[name] > alpha / (m - i):              # thresholds alpha/m, alpha/(m-1), ..., alpha
            break                                        # step-down: stop at the first failure
        rejected[name] = True
    return rejected


# ----------------------------------------------------------------------------- reliability
def pass_hat_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator of p^k (k independent runs ALL succeed), from c successes in n runs."""
    if not (0 < k <= n and 0 <= c <= n):
        raise ValueError("need 0 < k <= n and 0 <= c <= n")
    return math.comb(c, k) / math.comb(n, k)            # share of the k-subsets of runs that all succeed


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator of 1 - (1 - p)^k (AT LEAST one of k runs succeeds)."""
    if not (0 < k <= n and 0 <= c <= n):
        raise ValueError("need 0 < k <= n and 0 <= c <= n")
    total = math.comb(n, k)
    return (total - math.comb(n - c, k)) / total        # integer numerator: no cancellation in 1 - x
