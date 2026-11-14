"""Okapi BM25 from scratch on a sparse term-frequency matrix."""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
import scipy.sparse as sp

_TOKEN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25:
    """score(q, d) = sum_{t in q} idf(t) * tf (k1 + 1) / (tf + k1 (1 - b + b |d| / avgdl))."""

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b

    def fit(self, docs: list[str]) -> "BM25":
        # 1. Tokenize every document.
        tokenized_docs = [tokenize(doc) for doc in docs]

        # 2. Build the vocabulary (term -> column index).
        terms = sorted({
            term
            for tokens in tokenized_docs
            for term in tokens
        })

        self.vocab = {
            term: idx
            for idx, term in enumerate(terms)
        }

        # 3. Build the sparse term-frequency matrix.
        rows, cols, data = [], [], []

        for doc_idx, tokens in enumerate(tokenized_docs):
            counts = Counter(tokens)

            for term, count in counts.items():
                rows.append(doc_idx)
                cols.append(self.vocab[term])
                data.append(count)

        N = len(docs)
        V = len(self.vocab)

        self.tf = sp.coo_matrix(
            (data, (rows, cols)),
            shape=(N, V),
            dtype=np.float32
        ).tocsc()

        # 4. Compute document lengths.
        self.doc_len = np.array(
            [len(tokens) for tokens in tokenized_docs],
            dtype=np.float32
        )

        self.avgdl = float(self.doc_len.mean()) if N > 0 else 0.0

        # 5. Compute document frequency (df).
        df = np.asarray(
            (self.tf > 0).sum(axis=0)
        ).ravel()

        # 6. Compute inverse document frequency (IDF).
        self.idf = np.log(
            1 + (N - df + 0.5) / (df + 0.5)
        ).astype(np.float32)

        return self
        

    def scores(self, query: str) -> np.ndarray:
        """BM25 score of every document for `query` (distinct query terms)."""

        # One score per document.
        scores = np.zeros(self.tf.shape[0], dtype=np.float32)

        # Process every distinct query term.
        for term in set(tokenize(query)):

            # Find the vocabulary index.
            j = self.vocab.get(term)

            if j is None:
                continue

            # Locate the non-zero entries of column j in CSC format.
            start = self.tf.indptr[j]
            end = self.tf.indptr[j + 1]

            rows = self.tf.indices[start:end]
            tf = self.tf.data[start:end]

            # IDF of this term.
            idf = self.idf[j]

            if self.avgdl == 0:
                continue

            # Length normalization.
            norm = self.k1 * (
                1 - self.b + self.b * self.doc_len[rows] / self.avgdl
            )

            # Add the BM25 contribution to the corresponding documents.
            scores[rows] += idf * (
                tf * (self.k1 + 1) / (tf + norm)
            )

        return scores
