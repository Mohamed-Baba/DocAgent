"""Stage 1 tests: chunking (P4), BM25 (P5-P6), pooling and dense search (P7-P9),
fusion and reranking (P10-P12), metrics (P13-P14), one-shot RAG (P15)."""
import math
import random
from collections import Counter

import numpy as np
import pytest
import torch

from conftest import (BowEncoder, OverlapReranker, make_tokenizer, tiny_cross_encoder,
                      tiny_decoder)
from docagent.bm25 import BM25, tokenize
from docagent.chunking import chunk_pages
from docagent.dense import DenseEncoder, ExactIndex, last_token_pool, mean_pool
from docagent.llm import ScriptedLLM
from docagent.metrics_retrieval import (bootstrap_ci, paired_bootstrap, recall_at_k,
                                        reciprocal_rank)
from docagent.rag import ANSWER_SCHEMA, rag_answer
from docagent.rerank import CrossEncoderReranker
from docagent.retrieval import Retriever, rank_pages, rrf
from docagent.schema import Page, PageRef


# ----------------------------------------------------------------------------- P4
def test_chunking_counts_and_coverage():
    rng = random.Random(0)
    for _ in range(200):
        size = rng.randint(2, 30)
        overlap = rng.randint(0, size - 1)
        n = rng.randint(1, 120)
        words = [f"w{i}" for i in range(n)]
        chunks = chunk_pages([Page("d", 1, " ".join(words))], size, overlap)
        stride = size - overlap
        expected = 1 if n <= overlap else math.ceil((n - overlap) / stride)
        assert len(chunks) == expected
        assert chunks[-1].start + len(chunks[-1].text.split()) == n        # reaches the end
        spans = [set(range(c.start, c.start + len(c.text.split()))) for c in chunks]
        for a in range(n):                                                  # every span of
            for ell in range(1, min(overlap + 1, n - a) + 1):               # <= overlap + 1 words
                assert any(set(range(a, a + ell)) <= s for s in spans)      # is inside a chunk


def test_chunking_edge_cases():
    pages = [Page("d", 1, ""), Page("d", 2, "a b c")]
    chunks = chunk_pages(pages, 4, 1)
    assert [(c.chunk_id, c.page, c.text) for c in chunks] == [(0, 2, "a b c")]
    with pytest.raises(ValueError):
        chunk_pages(pages, 4, 4)


# ----------------------------------------------------------------------------- P5-P6
def _naive_bm25(docs, query, k1=1.2, b=0.75):
    toks = [tokenize(d) for d in docs]
    N, avgdl = len(docs), sum(map(len, toks)) / len(docs)
    out = []
    for t in toks:
        tf, s = Counter(t), 0.0
        for term in set(tokenize(query)):
            n = sum(term in set(u) for u in toks)
            if tf[term] == 0:
                continue
            idf = math.log(1 + (N - n + 0.5) / (n + 0.5))
            s += idf * tf[term] * (k1 + 1) / (tf[term] + k1 * (1 - b + b * len(t) / avgdl))
        out.append(s)
    return np.array(out)


def test_bm25_matches_naive_formula(toy_pages):
    docs = [p.text for p in toy_pages] + ["sinkhorn sinkhorn sinkhorn transport", "unrelated words only"]
    bm25 = BM25().fit(docs)
    for query in ["sinkhorn transport", "flow matching gaussian", "Sinkhorn, SINKHORN!", "absent"]:
        np.testing.assert_allclose(bm25.scores(query), _naive_bm25(docs, query), rtol=1e-5, atol=1e-6)
    assert (bm25.idf > 0).all()
    assert bm25.tf.shape == (len(docs), len(bm25.vocab))


# ----------------------------------------------------------------------------- P7-P9
def test_pooling_left_and_right_padding():
    h = torch.arange(2 * 4 * 3, dtype=torch.float32).view(2, 4, 3)
    left = torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1]])
    right = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
    assert torch.equal(last_token_pool(h, left), h[[0, 1], [3, 3]])
    assert torch.equal(last_token_pool(h, right), h[[0, 1], [1, 2]])
    torch.testing.assert_close(mean_pool(h, right)[0], h[0, :2].mean(0))


@pytest.mark.parametrize("eos_template", [False, True])
def test_dense_encoder_unit_norm_and_padding_invariance(eos_template):
    texts = ["optimal transport", "flow matching trains continuous normalizing flows quickly", "sinkhorn"]
    tok = make_tokenizer(texts, eos_template=eos_template)
    enc = DenseEncoder(model=tiny_decoder(len(tok)), tokenizer=tok, device="cpu", batch_size=8)
    assert enc.add_eos is (not eos_template)
    together = enc.encode(texts)
    alone = torch.cat([enc.encode([t]) for t in texts])
    torch.testing.assert_close(together.norm(dim=1), torch.ones(3))
    torch.testing.assert_close(together, alone, atol=1e-5, rtol=1e-4)   # left padding changes nothing
    assert not torch.allclose(enc.encode(["sinkhorn"], is_query=True), enc.encode(["sinkhorn"]))


class _Recorder(torch.nn.Module):
    """Wraps the tiny decoder and records the input_ids it receives."""

    def __init__(self, inner):
        super().__init__()
        self.inner, self.seen = inner, []

    def forward(self, input_ids=None, attention_mask=None):
        self.seen.append((input_ids.clone(), attention_mask.clone()))
        return self.inner(input_ids=input_ids, attention_mask=attention_mask)


def test_dense_encoder_keeps_eos_after_truncation():
    tok = make_tokenizer(["a b c d e f g h"])
    model = _Recorder(tiny_decoder(len(tok)))
    enc = DenseEncoder(model=model, tokenizer=tok, device="cpu", max_length=4)
    out = enc.encode(["a b c d e f g h", "a b"])
    ids, mask = model.seen[-1]
    assert out.shape == (2, 32) and ids.shape[1] <= 4               # truncated to max_length
    assert (ids[:, -1] == tok.eos_token_id).all()                   # EOS survives truncation
    assert mask[1].tolist() == [0, 1, 1, 1]                         # left padding


def test_exact_index_matches_brute_force():
    g = torch.Generator().manual_seed(0)
    E = torch.nn.functional.normalize(torch.randn(50, 16, generator=g), dim=1)
    Q = torch.nn.functional.normalize(torch.randn(3, 16, generator=g), dim=1)
    index = ExactIndex(E, device="cpu")
    vals, idx = index.search(Q, 5)
    ref = torch.topk(Q @ E.T, 5, dim=1)
    assert torch.equal(idx, ref.indices)
    mask = torch.zeros(50, dtype=torch.bool)
    mask[[3, 7]] = True
    vals, idx = index.search(Q, 5, mask=mask)
    assert idx.shape == (3, 2) and set(idx[0].tolist()) == {3, 7}


# ----------------------------------------------------------------------------- P10-P12
def test_rrf():
    fused = rrf([[1, 2, 3], [3, 1]], k=60)
    assert [i for i, _ in fused] == [1, 3, 2]
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 62)
    assert [i for i, _ in rrf([[5], [4]])] == [4, 5]           # tie -> smaller id first


def test_cross_encoder_batch_invariance():
    q = "what is sinkhorn"
    passages = ["sinkhorn scales rows and columns " * 30, "flow matching", "optimal transport plan"]
    tok = make_tokenizer([q] + passages, pair_template=True, padding_side="right")
    rr = CrossEncoderReranker(model=tiny_cross_encoder(len(tok)), tokenizer=tok, device="cpu",
                              max_length=32, batch_size=1)
    s1 = rr.score(q, passages)
    rr.batch_size = 3
    s3 = rr.score(q, passages)
    assert s1.shape == (3,) and np.isfinite(s1).all()
    np.testing.assert_allclose(s1, s3, rtol=1e-4, atol=1e-5)


def _toy_retriever(pages):
    chunks = chunk_pages(pages, 50, 10)
    texts = [c.text for c in chunks]
    enc = BowEncoder(texts)
    return Retriever(chunks, BM25().fit(texts), enc, ExactIndex(enc.encode(texts), device="cpu"),
                     OverlapReranker(), n_candidates=10)


def test_retriever_modes_and_filter(toy_pages):
    r = _toy_retriever(toy_pages)
    for mode in ("bm25", "dense", "hybrid", "hybrid+rerank"):
        hits = r.search("sinkhorn algorithm scalings", k=2, mode=mode)
        assert hits[0].chunk.ref == PageRef("cuturi2013", 2), mode
    hits = r.search("sinkhorn flow matching", k=5, mode="hybrid+rerank", doc_ids={"lipman2023"})
    assert hits and all(h.chunk.doc_id == "lipman2023" for h in hits)
    assert r.search_pages("flow matching", k_pages=3, mode="bm25")[0].doc_id in {"lipman2023", "tong2023"}
    with pytest.raises(ValueError):
        r.search("x", mode="magic")


def test_rank_pages_is_maxp(toy_pages):
    r = _toy_retriever(toy_pages)
    hits = r.search("flow matching trajectories", k=10, mode="bm25")
    pages = rank_pages(hits)
    assert len(pages) == len(set(pages))


# ----------------------------------------------------------------------------- P13-P14
def test_recall_and_rr():
    ranked = [PageRef("a", 1), PageRef("b", 2), PageRef("c", 3)]
    gold = {PageRef("b", 2), PageRef("z", 9)}
    assert recall_at_k(ranked, gold, 1) == 0.0 and recall_at_k(ranked, gold, 2) == 0.5
    assert reciprocal_rank(ranked, gold) == 0.5 and reciprocal_rank(ranked, {PageRef("z", 9)}) == 0.0


def test_bootstrap():
    mean, lo, hi = bootstrap_ci([1, 0, 1, 1, 0, 1, 1, 1], n_boot=2000)
    assert lo <= mean <= hi and 0 <= lo and hi <= 1
    assert bootstrap_ci([0.3] * 10) == pytest.approx((0.3, 0.3, 0.3))
    d, lo, hi = paired_bootstrap([1, 1, 1, 0], [0, 1, 0, 0])
    assert d == pytest.approx(0.5) and lo <= d <= hi


# ----------------------------------------------------------------------------- P15
def test_rag_answer(toy_pages):
    r = _toy_retriever(toy_pages)
    llm = ScriptedLLM([{"answer": "row and column scalings", "abstain": False,
                        "citations": [{"doc_id": "cuturi2013", "page": 2}, {"doc_id": "x", "page": "2"}]}])
    pred = rag_answer("what does the sinkhorn algorithm alternate", r, llm, k=3, mode="hybrid+rerank")
    assert pred.answer == "row and column scalings" and not pred.abstain
    assert pred.citations == [PageRef("cuturi2013", 2)]               # malformed citation dropped
    assert PageRef("cuturi2013", 2) in pred.consulted and pred.n_llm_calls == 1
    assert llm.calls[0]["schema"] == ANSWER_SCHEMA
    assert "Question: what does the sinkhorn" in llm.calls[0]["messages"][1]["content"]
    bad = rag_answer("q", r, ScriptedLLM(["not json"]), k=2, mode="bm25")
    assert bad.abstain and bad.answer == "" and bad.citations == []
