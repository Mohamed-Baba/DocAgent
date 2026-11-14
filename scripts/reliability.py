"""X7 (D7.3): reliability over repeated runs, pass^k for k = 1..runs, hybrid RAG against the agent.

    python scripts/reliability.py                  # 5 runs, 5 questions per category, temperature 0.7

Resumable like run_benchmark.py: the runs already stored in runs/reliability.jsonl are skipped.
"""
import argparse
from collections import defaultdict

from docagent.agent import Agent
from docagent.config import Config
from docagent.llm import OllamaClient
from docagent.metrics_answers import is_correct, pass_hat_k
from docagent.rag import rag_answer
from docagent.retrieval import build_retriever
from docagent.schema import append_jsonl, load_pages, load_questions, read_jsonl
from docagent.tools import default_toolbox


class Sampling:
    """One run of X7: same client, forced temperature, its own seed.

    OllamaClient always sends its seed (0 by default) and rag_answer always asks for temperature 0:
    without this wrapper, the n runs would be n identical copies.
    """

    def __init__(self, llm, temperature: float, seed: int):
        self.llm, self.temperature, self.seed = llm, temperature, seed

    def chat(self, messages, schema=None, temperature=0.0, max_tokens=None):
        self.llm.seed = self.seed
        return self.llm.chat(messages, schema=schema, temperature=self.temperature, max_tokens=max_tokens)


def pick_questions(questions, per_category: int):
    """The first `per_category` questions of each category (deterministic)."""
    by_cat = defaultdict(list)
    for q in questions:
        by_cat[q.category].append(q)
    return [q for qs in by_cat.values() for q in qs[:per_category]]


def run_reliability(client, retriever, pages, questions, cfg, runs: int, temperature: float, out_path) -> dict:
    """Run both configurations `runs` times on `questions`; return {config: [pass^1, ..., pass^runs]}."""
    done = {(r["config"], r["qid"], r["run"]) for r in read_jsonl(out_path)} if out_path.exists() else set()
    toolbox = default_toolbox(retriever, pages)
    for run in range(runs):
        llm = Sampling(client, temperature, seed=run)               # same seed for both configurations
        runners = {
            "rag_hybrid": lambda text: rag_answer(text, retriever, llm, k=cfg.top_k, mode="hybrid+rerank"),
            "agent": lambda text: Agent(llm, toolbox, cfg.max_steps, cfg.max_errors).run(text).to_prediction(),
        }
        for name, runner in runners.items():
            for q in questions:
                if (name, q.qid, run) in done:
                    continue
                pred = runner(q.question)
                append_jsonl(out_path, [{"config": name, "qid": q.qid, "run": run, "correct": is_correct(pred, q),
                                         "answer": pred.answer, "abstain": pred.abstain}])
    successes = defaultdict(int)
    for r in read_jsonl(out_path):
        if r["run"] < runs:
            successes[(r["config"], r["qid"])] += r["correct"]
    table = {}
    for name in ("rag_hybrid", "agent"):
        cs = [successes[(name, q.qid)] for q in questions]
        table[name] = [sum(pass_hat_k(runs, c, k) for c in cs) / len(cs) for k in range(1, runs + 1)]
    return table


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--per-category", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()

    cfg = Config()
    pages = load_pages(cfg.pages_path)
    retriever = build_retriever(cfg, pages)
    client = OllamaClient(cfg.llm_model, cfg.ollama_host, cfg.num_ctx)
    questions = pick_questions(load_questions(cfg.questions_path), args.per_category)
    table = run_reliability(client, retriever, pages, questions, cfg, args.runs, args.temperature,
                            cfg.runs_dir / "reliability.jsonl")
    print(f"{len(questions)} questions, {args.runs} runs at temperature {args.temperature}\n")
    print("| config | " + " | ".join(f"pass^{k}" for k in range(1, args.runs + 1)) + " |")
    print("|---" * (args.runs + 1) + "|")
    for name, values in table.items():
        print(f"| {name} | " + " | ".join(f"{v:.2f}" for v in values) + " |")