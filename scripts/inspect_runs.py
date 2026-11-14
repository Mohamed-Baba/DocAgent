"""Per-question view of the benchmark runs: what each configuration answered, and why.

    python scripts/inspect_runs.py              # every question: answers, evidence seen, verifier rejections
    python scripts/inspect_runs.py m03 s02      # some questions, plus every step of the two agents
"""
import json
import re
import sys

from docagent.config import Config
from docagent.evaluate import CONFIGS
from docagent.metrics_answers import is_correct
from docagent.schema import Prediction, load_questions, read_jsonl

PART = re.compile(r"\(part (\d+)/(\d+)\)")


def thought(reply: str) -> str:
    try:
        return str(json.loads(reply).get("thought", ""))
    except (json.JSONDecodeError, AttributeError):
        return ""


def show_steps(p: Prediction) -> None:
    for k, s in enumerate(p.trace):
        if k > 0 and s["index"] == 0:                      # VerifiedAgent: the steps of every attempt follow
            print("        --- new attempt, after the verifier's feedback")
        obs = s.get("observation") or ""
        notes = []
        if "[... truncated:" in obs:                       # the agent cut the observation (5000 characters)
            notes.append("OBSERVATION CUT")
        m = PART.search(obs[:300])
        if m:
            notes.append(f"part {m.group(1)}/{m.group(2)}")
        if s["error"]:
            notes.append(f"ERROR: {s['error']}")
        print(f"        {s['index']}. {s['action'] or 'invalid reply'} {s['args'] or ''}"
              + (f"  [{'; '.join(notes)}]" if notes else ""))
        if thought(s.get("reply") or ""):
            print(f"             thought: {thought(s['reply'])[:110]}")


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")               # a model thought with "≈" must not crash a Windows console
    cfg = Config()
    questions = {q.qid: q for q in load_questions(cfg.questions_path)}
    runs = {}
    for name in CONFIGS:
        path = cfg.runs_dir / f"{name}.jsonl"
        if path.exists():
            runs[name] = {r["qid"]: Prediction.from_dict(r["prediction"]) for r in read_jsonl(path)}
    selected = sys.argv[1:]
    if [qid for qid in selected if qid not in questions]:
        print(f"unknown qid: {[qid for qid in selected if qid not in questions]}")
    for qid, q in questions.items():
        if selected and qid not in selected:
            continue
        print(f"\n{qid} [{q.category}] {q.question}")
        print(f"    gold: {q.answers or 'abstain'} | evidence: {', '.join(map(str, q.evidence)) or '-'}")
        for name, preds in runs.items():
            p = preds.get(qid)
            if p is None:
                continue
            answer = "ABSTAIN" if p.abstain else p.answer[:30]
            seen = f"{len(set(q.evidence) & set(p.consulted))}/{len(q.evidence)}" if q.evidence else "-"
            cites = ", ".join(map(str, p.citations)) or "-"
            print(f"    {'ok ' if is_correct(p, q) else 'BAD'} {name:15s} {answer:30s} seen {seen:4s} cites: {cites}")
            for k, verdict in enumerate(p.meta.get("verdicts", []), start=1):
                if not verdict["rejected"] and not verdict["ok"]:
                    print(f"          attempt {k}: no citation")
                for ref, why in verdict["rejected"]:
                    print(f"          attempt {k}: {ref} rejected ({why})")
        if selected:                                       # the steps, to see where each agent went wrong
            for name in ("agent", "agent_verified"):
                p = runs.get(name, {}).get(qid)
                if p is not None:
                    print(f"    --- {name}: {p.n_llm_calls} calls, {p.latency_s:.0f} s")
                    show_steps(p)
