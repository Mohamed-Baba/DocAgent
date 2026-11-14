"""D7.3: run the four configurations on data/questions.jsonl and write runs/results.md.

    python scripts/run_benchmark.py                       # all configurations (resumable)
    python scripts/run_benchmark.py --configs agent       # one configuration
"""
import argparse
import time

from docagent.config import Config
from docagent.evaluate import CONFIGS, compare, make_runner, run_config, score_all, summarize, to_markdown
from docagent.llm import OllamaClient
from docagent.retrieval import build_retriever
from docagent.schema import load_pages, load_questions

PAIRS = [("rag_hybrid", "rag_dense"), ("agent", "rag_hybrid"), ("agent_verified", "agent"),
         ("agent_verified", "rag_hybrid")]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=CONFIGS)
    args = ap.parse_args()

    cfg = Config()
    pages = load_pages(cfg.pages_path)
    questions = load_questions(cfg.questions_path)
    retriever = build_retriever(cfg, pages)
    llm = OllamaClient(cfg.llm_model, cfg.ollama_host, cfg.num_ctx)

    scores, report = {}, [f"# Results ({time.strftime('%Y-%m-%d %H:%M')}, model {cfg.llm_model})", ""]
    for name in args.configs:
        preds = run_config(name, make_runner(name, cfg, pages, retriever, llm), questions,
                           cfg.runs_dir / f"{name}.jsonl")
        scores[name] = score_all(preds, questions)
        report += [to_markdown(name, summarize(scores[name])), ""]
    pairs = [p for p in PAIRS if p[0] in scores and p[1] in scores]
    if pairs:
        report += ["### Paired comparisons (exact McNemar, Holm-corrected at 0.05)", "",
                   "| pair | first right only | second right only | p | rejected |", "|---|---|---|---|---|"]
        for r in compare(scores, pairs):
            report.append(f"| {r['pair']} | {r['n01']} | {r['n10']} | {r['p']:.3f} | {r['rejected']} |")
    report += ["", f"VRAM used by the LLM: {llm.vram_bytes() / 1e9:.2f} GB"]
    out = cfg.runs_dir / "results.md"
    out.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
