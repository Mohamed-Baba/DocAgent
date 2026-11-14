"""D7.1: sanity checks of data/questions.jsonl before any measurement.

- every evidence page exists in data/pages.jsonl; unanswerable questions have no evidence
- counts per category (target: 15 each)
- lexical overlap: share of the question's content words that appear on its evidence pages.
  A high overlap favours BM25 (the question copies the page): rephrase those questions.

    python scripts/check_questions.py --max-overlap 0.6
"""
import argparse
from collections import Counter

from docagent.bm25 import tokenize
from docagent.config import Config
from docagent.schema import load_pages, load_questions

STOP = set("a an the of to in on for and or is are was were be by with from as at that this which what "
           "how why when where who does do did it its their these those than then into".split())

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-overlap", type=float, default=0.6)
    args = ap.parse_args()
    cfg = Config()
    pages = {p.ref: p for p in load_pages(cfg.pages_path)}
    questions = load_questions(cfg.questions_path)
    print("per category:", dict(Counter(q.category for q in questions)))
    problems = 0
    for q in questions:
        if q.answerable != bool(q.evidence) or q.answerable != bool(q.answers):
            print(f"[{q.qid}] answerable questions need answers AND evidence; unanswerable ones neither")
            problems += 1
        missing = [str(e) for e in q.evidence if e not in pages]
        if missing:
            print(f"[{q.qid}] unknown evidence pages: {missing}")
            problems += 1
        words = {w for w in tokenize(q.question) if w not in STOP}
        if q.evidence and words:
            page_words = set().union(*(set(tokenize(pages[e].text)) for e in q.evidence if e in pages))
            overlap = len(words & page_words) / len(words)
            if overlap > args.max_overlap:
                print(f"[{q.qid}] lexical overlap {overlap:.2f} > {args.max_overlap}: rephrase? {q.question!r}")
    print(f"{len(questions)} questions, {problems} blocking problem(s)")
