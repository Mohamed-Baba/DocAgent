"""D2.1-D2.3: retrieval table (Recall@k, MRR with 95% bootstrap CIs) and paired comparisons.

    python scripts/eval_retrieval.py --modes bm25 dense hybrid hybrid+rerank
"""
import argparse

from docagent.config import Config
from docagent.metrics_retrieval import evaluate_retrieval, format_table, paired_bootstrap
from docagent.retrieval import MODES, build_retriever
from docagent.schema import load_pages, load_questions

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    ap.add_argument("--chunk-size", type=int, default=None)
    args = ap.parse_args()

    cfg = Config()
    if args.chunk_size:
        cfg.chunk_size, cfg.chunk_overlap = args.chunk_size, args.chunk_size // 4
    retriever = build_retriever(cfg, load_pages(cfg.pages_path), modes=args.modes)
    questions = load_questions(cfg.questions_path)
    results = evaluate_retrieval(retriever, questions, args.modes)
    n = len(next(iter(results.values()))["mrr"])
    print(f"{n} answerable questions, {len(retriever.chunks)} chunks\n")
    print(format_table(results))
    print("\nPaired differences on Recall@5 (mean [95% CI]):")
    for a, b in zip(args.modes, args.modes[1:]):
        mean, lo, hi = paired_bootstrap(results[b]["recall@5"], results[a]["recall@5"])
        print(f"  {b} - {a}: {mean:+.3f} [{lo:+.3f}, {hi:+.3f}]")
