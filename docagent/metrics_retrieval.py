"""Retrieval metrics on evidence pages, with bootstrap confidence intervals."""
from __future__ import annotations

import numpy as np

from .schema import PageRef, Question


def recall_at_k(ranked: list[PageRef], gold: set[PageRef], k: int) -> float:
    """Fraction of the gold pages found among the first k ranked pages."""
    # |top_k  inter  gold| / |gold|
    if not gold :
        return 0.0
    top_k = set(ranked[:k])
    return len(top_k & gold)/len(gold)


def reciprocal_rank(ranked: list[PageRef], gold: set[PageRef]) -> float:
    """1 / rank of the first gold page (0 if none is retrieved)."""
    for rank,page in enumerate(ranked , start = 1) :
        if page in gold :
            return 1/rank
    return 0.0
    


def bootstrap_ci(values, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0):
    """Mean and percentile-bootstrap (1 - alpha) confidence interval of the mean.

    Returns (mean, low, high). Resamples the n values with replacement n_boot times.
    """
    # draw an (n_boot, n) matrix of indices, average each row, take the
    # alpha/2 and 1 - alpha/2 quantiles
   
    values = np.asarray(values,dtype = float)
    n = len(values) 
    if n == 0:
        raise ValueError("values must not be empty")

    if n_boot <= 0:
        raise ValueError("n_boot must be positive")

    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    # random number generator 
    rng = np.random.default_rng(seed)

    # Generate bootstrap samples by sampling indices with replacement.
    indices = rng.integers(0,n ,size = (n_boot , n))

    # Build the bootstrap samples.
    samples = values[indices]    # retourne une matrice de taille de indices M et chaque M[ind] == values[ind]  

    # Compute the mean of each bootstrap sample.
    means = samples.mean(axis = 1)

    # Percentile confidence interval.
    low,high = np.quantile(means , [alpha/2 , 1 - alpha/2])

    return float(values.mean()) , float(low) , float(high)




def paired_bootstrap(a, b, **kw):
    """CI of mean(a - b) when a[i] and b[i] are measured on the SAME question i."""
    return bootstrap_ci(np.asarray(a, float) - np.asarray(b, float), **kw)


def evaluate_retrieval(retriever, questions: list[Question], modes, ks=(1, 5, 10)) -> dict:
    """Per-question metrics for each mode: {mode: {"recall@k": [...], "mrr": [...]}}.

    Unanswerable questions (no evidence) are skipped.
    """
    qs = [q for q in questions if q.evidence]
    out = {}
    for mode in modes:
        res = {f"recall@{k}": [] for k in ks}
        res["mrr"] = []
        for q in qs:
            ranked = retriever.search_pages(q.question, k_pages=max(ks), mode=mode)
            gold = set(q.evidence)
            for k in ks:
                res[f"recall@{k}"].append(recall_at_k(ranked, gold, k))
            res["mrr"].append(reciprocal_rank(ranked, gold))
        out[mode] = res
    return out


def format_table(results: dict) -> str:
    """Markdown table: mean [95% CI] for every mode and metric."""
    metrics = list(next(iter(results.values())).keys())
    lines = ["| mode | " + " | ".join(metrics) + " |", "|---" * (len(metrics) + 1) + "|"]
    for mode, res in results.items():
        cells = []
        for m in metrics:
            mean, lo, hi = bootstrap_ci(res[m])
            cells.append(f"{mean:.3f} [{lo:.3f}, {hi:.3f}]")
        lines.append(f"| {mode} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
