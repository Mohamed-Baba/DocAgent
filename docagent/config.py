"""Central configuration (plumbing). Edit paths and model names here, nowhere else."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Run every script from the repository root, or set DOCAGENT_ROOT.
ROOT = Path(os.environ.get("DOCAGENT_ROOT", Path.cwd()))


@dataclass
class Config:
    # data
    pdf_dir: Path = field(default_factory=lambda: ROOT / "data" / "pdfs")
    pages_path: Path = field(default_factory=lambda: ROOT / "data" / "pages.jsonl")
    images_dir: Path = field(default_factory=lambda: ROOT / "data" / "page_images")
    questions_path: Path = field(default_factory=lambda: ROOT / "data" / "questions.jsonl")
    cache_dir: Path = field(default_factory=lambda: ROOT / "cache")
    runs_dir: Path = field(default_factory=lambda: ROOT / "runs")
    # models
    llm_model: str = "qwen3-vl:8b-instruct"   # the Ollama tag validated in D0.2 (`ollama list`)
    ollama_host: str = "http://localhost:11434"
    num_ctx: int = 16384                      # set it explicitly: never rely on the server default
    embed_model: str = "Qwen/Qwen3-Embedding-0.6B"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    # retrieval
    chunk_size: int = 200        # words per chunk
    chunk_overlap: int = 50      # words shared by two consecutive chunks
    n_candidates: int = 50       # candidates per retriever, sent to fusion and reranking
    top_k: int = 5               # chunks shown to the model
    # agent
    max_steps: int = 8
    max_errors: int = 3
    page_part_chars: int = 4800  # inspect_page shows a long page in parts that fit in one observation (5000)
    # verification
    judge_chars: int = 12000     # the judge sees the whole page (methods pages reach ~9,000 characters)
    joint_check: bool = True     # pages that fail alone are judged together once (computations across pages)
