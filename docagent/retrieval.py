"""The retriever: BM25, dense, hybrid (RRF) and hybrid + cross-encoder reranking."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from .bm25 import BM25
from .schema import Chunk, Hit, PageRef

MODES = ("bm25", "dense", "hybrid", "hybrid+rerank")


def rrf(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion. rankings: lists of item ids, best first.

    Returns [(id, score)] sorted by score (desc), ties broken by id (asc) for determinism.
    """
    # score(d) = sum over the rankings that contain d of 1 / (k + rank),
    # rank starting at 1
    scores = defaultdict(float)

    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] += 1.0 / (k + rank)

    return [
        (int(item_id), float(score))
        for item_id, score in sorted(
            scores.items(),
            key=lambda x: (-x[1], x[0])
        )
    ]


def rank_pages(hits: list[Hit]) -> list[PageRef]:
    """MaxP: a page is as relevant as its best chunk (hits are sorted best first)."""
    return list(dict.fromkeys(h.chunk.ref for h in hits))


class Retriever:
    def __init__(self, chunks: list[Chunk], bm25: BM25 | None = None, encoder=None, index=None,
                 reranker=None, n_candidates: int = 50):
        self.chunks = chunks
        self.bm25, self.encoder, self.index, self.reranker = bm25, encoder, index, reranker
        self.n_candidates = n_candidates
        self.doc_of = np.array([c.doc_id for c in chunks])

    def _mask(self, doc_ids) -> np.ndarray | None:
        return None if not doc_ids else np.isin(self.doc_of, list(doc_ids))

    def _bm25_ranking(self, query: str, mask) -> list[tuple[int, float]]:
        s = self.bm25.scores(query)
        if mask is not None:
            s = np.where(mask, s, -np.inf)
        order = np.argsort(-s, kind="stable")[: self.n_candidates]
        return [(int(i), float(s[i])) for i in order if s[i] > 0]   # must share a term

    def _dense_ranking(self, query: str, mask) -> list[tuple[int, float]]:
        q = self.encoder.encode([query], is_query=True)
        m = None if mask is None else torch.from_numpy(mask)
        vals, idx = self.index.search(q, self.n_candidates, mask=m)
        return [(int(i), float(v)) for v, i in zip(vals[0], idx[0]) if np.isfinite(float(v))]

    def search(self, query: str, k: int = 5, mode: str = "hybrid+rerank",
               doc_ids: set[str] | None = None) -> list[Hit]:
        """Top-k chunks for `query`; doc_ids restricts the search to some papers."""
        mask = self._mask(doc_ids)
        # route on mode; hybrid = rrf of the two candidate id lists (keep
        # n_candidates); hybrid+rerank = rescore the fused candidates with the
        # cross-encoder and sort; unknown mode -> ValueError

        if mode == "bm25":
            ranked = self._bm25_ranking(query, mask)

        elif mode == "dense":
            ranked = self._dense_ranking(query, mask)

        elif mode in ("hybrid", "hybrid+rerank"):
            bm25_ranked = self._bm25_ranking(query, mask)
            dense_ranked = self._dense_ranking(query, mask)

            fused = rrf([
                [idx for idx, score in bm25_ranked],
                [idx for idx, score in dense_ranked],
            ])

            fused = fused[:self.n_candidates]

            if mode == "hybrid":
                ranked = fused

            else:
                if self.reranker is None:
                    raise ValueError("Reranker is required for hybrid+rerank")

                candidates = [self.chunks[idx].text for idx, score in fused]

                scores = self.reranker.score(query, candidates)

                reranked = [
                    (idx, float(score))
                    for (idx, _), score in zip(fused, scores)
                ]

                ranked = sorted(reranked,key=lambda x: (-x[1], x[0]))

        else:
            raise ValueError(f"Unknown retrieval mode: {mode}")

        return [Hit(self.chunks[i], float(s)) for i, s in ranked[:k]]

            

    def search_pages(self, query: str, k_pages: int = 10, mode: str = "hybrid+rerank",
                     doc_ids: set[str] | None = None) -> list[PageRef]:
        hits = self.search(query, k=self.n_candidates, mode=mode, doc_ids=doc_ids)
        return rank_pages(hits)[:k_pages]


def build_retriever(cfg, pages, modes=MODES) -> Retriever:
    """Plumbing: chunk, fit BM25, load the models only if a mode needs them, cache embeddings."""
    from .chunking import chunk_pages
    from .dense import DenseEncoder, ExactIndex, encode_chunks_cached
    from .rerank import CrossEncoderReranker

    chunks = chunk_pages(pages, cfg.chunk_size, cfg.chunk_overlap)
    bm25 = BM25().fit([c.text for c in chunks])
    encoder = index = reranker = None
    if any(m != "bm25" for m in modes):
        encoder = DenseEncoder(cfg.embed_model)
        emb = encode_chunks_cached(encoder, [c.text for c in chunks], cfg.cache_dir)
        index = ExactIndex(emb)
    if "hybrid+rerank" in modes:
        reranker = CrossEncoderReranker(cfg.rerank_model)
    return Retriever(chunks, bm25, encoder, index, reranker, cfg.n_candidates)
