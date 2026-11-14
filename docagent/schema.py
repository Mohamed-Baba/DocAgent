"""Shared data types and JSONL helpers (plumbing, used by every stage)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator


@dataclass(frozen=True, order=True)
class PageRef:
    """A page of the corpus: the unit of evidence (pages are numbered from 1)."""
    doc_id: str
    page: int

    def __str__(self) -> str:
        return f"{self.doc_id} p.{self.page}"


@dataclass
class Page:
    doc_id: str
    page: int
    text: str                                        # Markdown of the page (Docling)
    tables: list[str] = field(default_factory=list)  # tables of the page, as Markdown
    image_path: str | None = None                    # PNG rendering of the page

    @property
    def ref(self) -> PageRef:
        return PageRef(self.doc_id, self.page)


@dataclass
class Chunk:
    chunk_id: int
    doc_id: str
    page: int
    text: str
    start: int  # index of the first word of the chunk inside its page

    @property
    def ref(self) -> PageRef:
        return PageRef(self.doc_id, self.page)


@dataclass
class Hit:
    chunk: Chunk
    score: float


@dataclass
class Question:
    qid: str
    category: str              # simple | multistep | multidoc | unanswerable
    question: str
    answers: list[str]         # accepted short answers (aliases); empty if unanswerable
    evidence: list[PageRef]    # pages that contain the answer; empty if unanswerable
    answer_type: str = "text"  # text | number

    @property
    def answerable(self) -> bool:
        return self.category != "unanswerable"


@dataclass
class Prediction:
    """What every configuration (RAG or agent) returns for one question."""
    answer: str
    abstain: bool
    citations: list[PageRef]
    consulted: list[PageRef]   # pages shown to the model (retrieved or returned by tools)
    n_llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    trace: list[dict] = field(default_factory=list)
    meta: dict = field(default_factory=dict)   # e.g. stop_reason, verifier verdicts

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Prediction":
        d = dict(d)
        d["citations"] = [PageRef(**c) for c in d.get("citations", [])]
        d["consulted"] = [PageRef(**c) for c in d.get("consulted", [])]
        return Prediction(**d)


# --------------------------------------------------------------------------- JSONL
def write_jsonl(path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:   # always utf-8 (Windows default is cp1252)
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path) -> Iterator[dict]:
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def save_pages(path, pages: Iterable[Page]) -> None:
    write_jsonl(path, (asdict(p) for p in pages))


def load_pages(path) -> list[Page]:
    return [Page(**r) for r in read_jsonl(path)]


def load_questions(path) -> list[Question]:
    questions = []
    for r in read_jsonl(path):
        questions.append(Question(
            qid=r["qid"], category=r["category"], question=r["question"],
            answers=list(r.get("answers", [])),
            evidence=[PageRef(e["doc_id"], int(e["page"])) for e in r.get("evidence", [])],
            answer_type=r.get("answer_type", "text"),
        ))
    return questions
