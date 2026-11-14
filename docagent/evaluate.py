"""The study: run the four configurations on the question set, score, summarise, compare."""
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from .metrics_answers import holm, mcnemar_exact, score
from .metrics_retrieval import bootstrap_ci
from .schema import Prediction, Question, append_jsonl, read_jsonl

CONFIGS = ("rag_dense", "rag_hybrid", "agent", "agent_verified")


def make_runner(name: str, cfg, pages, retriever, llm):
    """question (str) -> Prediction, for each configuration of the study."""
    from .agent import Agent
    from .rag import rag_answer
    from .tools import default_toolbox
    from .verifier import VerifiedAgent, Verifier

    if name == "rag_dense":
        return lambda q: rag_answer(q, retriever, llm, k=cfg.top_k, mode="dense")
    if name == "rag_hybrid":
        return lambda q: rag_answer(q, retriever, llm, k=cfg.top_k, mode="hybrid+rerank")
    agent = Agent(llm, default_toolbox(retriever, pages, cfg.page_part_chars), cfg.max_steps, cfg.max_errors)
    if name == "agent":
        return lambda q: agent.run(q).to_prediction()
    if name == "agent_verified":
        verifier = Verifier(llm, pages, max_chars=cfg.judge_chars, joint=cfg.joint_check)
        return VerifiedAgent(agent, verifier).run
    raise ValueError(f"unknown configuration {name!r}")


def run_config(name: str, runner, questions: list[Question], out_path) -> dict[str, Prediction]:
    """Resumable: questions already stored in out_path are not run again (a crash costs nothing)."""
    out_path = Path(out_path)
    done = {}
    if out_path.exists():
        done = {r["qid"]: Prediction.from_dict(r["prediction"]) for r in read_jsonl(out_path)}
    for q in questions:
        if q.qid in done:
            continue
        pred = runner(q.question)
        append_jsonl(out_path, [{"qid": q.qid, "config": name, "prediction": pred.to_dict()}])
        done[q.qid] = pred
    return done


def score_all(preds: dict[str, Prediction], questions: list[Question]) -> list[dict]:
    return [score(preds[q.qid], q) for q in questions if q.qid in preds]


def _mean(values) -> float:
    v = [x for x in values if not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(v)) if v else math.nan


def summarize(scores: list[dict]) -> list[dict]:
    """One row per category, plus 'all': accuracy [95% CI], citations, abstention, cost."""
    groups = defaultdict(list)
    for s in scores:
        groups[s["category"]].append(s)
        groups["all"].append(s)
    rows = []
    for cat, ss in groups.items():
        acc, lo, hi = bootstrap_ci([s["correct"] for s in ss])
        answerable = [s for s in ss if s["answerable"]]
        unanswerable = [s for s in ss if not s["answerable"]]
        rows.append({"category": cat, "n": len(ss), "accuracy": acc, "ci": (lo, hi),
                     "cit_precision": _mean(s["cit_precision"] for s in ss),
                     "evidence_consulted": _mean(s["evidence_consulted"] for s in ss),
                     "correct_abstention": _mean(s["abstain"] for s in unanswerable),
                     "false_abstention": _mean(s["abstain"] for s in answerable),
                     "calls": _mean(s["n_llm_calls"] for s in ss), "tokens": _mean(s["tokens"] for s in ss),
                     "latency_s": _mean(s["latency_s"] for s in ss)})
    return rows


def compare(scores_by_config: dict[str, list[dict]], pairs: list[tuple[str, str]], alpha: float = 0.05):
    """McNemar on every pair (all questions), Holm correction over the family of pairs."""
    tests = {}
    for a, b in pairs:
        ca = {s["qid"]: bool(s["correct"]) for s in scores_by_config[a]}
        cb = {s["qid"]: bool(s["correct"]) for s in scores_by_config[b]}
        common = sorted(set(ca) & set(cb))
        tests[f"{a} vs {b}"] = mcnemar_exact([ca[q] for q in common], [cb[q] for q in common])
    rejected = holm({k: v[2] for k, v in tests.items()}, alpha)
    return [{"pair": k, "n01": v[0], "n10": v[1], "p": v[2], "rejected": rejected[k]} for k, v in tests.items()]


def fmt(x, digits=2) -> str:
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{digits}f}"


def to_markdown(name: str, rows: list[dict]) -> str:
    head = ("| category | n | accuracy [95% CI] | cit. precision | evidence seen | correct abst. "
            "| false abst. | calls | tokens | latency (s) |")
    lines = [f"### {name}", "", head, "|" + "---|" * 10]
    for r in rows:
        lines.append(f"| {r['category']} | {r['n']} | {fmt(r['accuracy'])} [{fmt(r['ci'][0])}, {fmt(r['ci'][1])}] "
                     f"| {fmt(r['cit_precision'])} | {fmt(r['evidence_consulted'])} | {fmt(r['correct_abstention'])} "
                     f"| {fmt(r['false_abstention'])} | {fmt(r['calls'], 1)} | {fmt(r['tokens'], 0)} "
                     f"| {fmt(r['latency_s'], 1)} |")
    return "\n".join(lines)
