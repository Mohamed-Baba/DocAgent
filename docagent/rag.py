"""One-shot RAG: retrieve once, generate once (the baselines of the study)."""
from __future__ import annotations

import time

from .schema import Hit, PageRef, Prediction

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "citations": {
            "type": "array",
            "items": {"type": "object",
                      "properties": {"doc_id": {"type": "string"}, "page": {"type": "integer"}},
                      "required": ["doc_id", "page"]},
        },
        "abstain": {"type": "boolean"},
    },
    "required": ["answer", "citations", "abstain"],
}

RAG_SYSTEM = (
    "You answer questions about scientific papers using ONLY the numbered sources below.\n"
    "Reply with a JSON object {\"answer\": ..., \"citations\": [{\"doc_id\": ..., \"page\": ...}], "
    "\"abstain\": ...}.\n"
    "- answer: a short answer (a few words or a number), no explanation.\n"
    "- citations: the pages (doc_id and page exactly as shown) that support the answer.\n"
    "- abstain: true, with an empty answer and no citation, if the sources do not contain the answer."
)


def format_sources(hits: list[Hit], max_chars: int = 1500) -> str:
    return "\n\n".join(f"[S{i}] doc_id={h.chunk.doc_id} page={h.chunk.page}\n{h.chunk.text[:max_chars]}"
                       for i, h in enumerate(hits, start=1))


def parse_citations(raw) -> list[PageRef]:
    """Keep the well-formed citations only."""
    out = []
    for c in raw or []:
        if isinstance(c, dict) and isinstance(c.get("doc_id"), str) and type(c.get("page")) is int:
            out.append(PageRef(c["doc_id"], c["page"]))
    return out


def rag_answer(question: str, retriever, llm, k: int = 5, mode: str = "hybrid+rerank") -> Prediction:
    """Retrieve k chunks, make ONE constrained call, return a Prediction."""
    t0 = time.perf_counter()
    # search; messages = [system RAG_SYSTEM, user "Sources: ... Question:..."];
    # llm.chat with ANSWER_SCHEMA; parse (invalid JSON -> abstain);
    # consulted= retrieved pages (unique, in order)
    hits = retriever.search(question, k=k, mode=mode)
    messages = [{"role": "system", "content": RAG_SYSTEM},
                {"role": "user", "content": f"Sources:\n{format_sources(hits)}\n\nQuestion: {question}"}]
    reply = llm.chat(messages, schema=ANSWER_SCHEMA, max_tokens=300)     # the cap stops runaway generations
    try:
        out = reply.json()
    except ValueError:
        out = None
    if not isinstance(out, dict):                    # invalid JSON, or valid JSON that is not an object
        out = {"answer": "", "citations": [], "abstain": True}
    answer = str(out.get("answer", "")).strip()
    abstain = bool(out.get("abstain", False)) or not answer     # empty answer = abstention, as in Agent._finish
    pred = Prediction(
        answer="" if abstain else answer,
        abstain=abstain,
        citations=[] if abstain else parse_citations(out.get("citations")),
        consulted=list(dict.fromkeys(h.chunk.ref for h in hits)),
        n_llm_calls=1, prompt_tokens=reply.prompt_tokens,
        completion_tokens=reply.completion_tokens,
    )
    pred.latency_s = time.perf_counter() - t0
    return pred