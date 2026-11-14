"""D1: ingest data/pdfs/*.pdf into data/pages.jsonl (+ page images). Re-running skips cached PDFs.

    python scripts/ingest_corpus.py
"""
import json
import logging

from docagent.config import Config
from docagent.ingest import ingest_corpus

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cfg = Config()
    stats = ingest_corpus(cfg.pdf_dir, cfg.pages_path, cfg.images_dir)
    print(json.dumps(stats, indent=2))
