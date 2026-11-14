"""Cross-encoder reranking: score each (query, passage) pair jointly."""
from __future__ import annotations

import numpy as np
import torch


class CrossEncoderReranker:
    """Default: BAAI/bge-reranker-v2-m3 (XLM-RoBERTa encoder + one-logit head)."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str | None = None,
                 max_length: int = 512, batch_size: int = 16, model=None, tokenizer=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length, self.batch_size = max_length, batch_size
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        if model is None:
            from transformers import AutoModelForSequenceClassification
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            if self.device.startswith("cuda"):
                model = model.to(dtype=torch.float16)
        self.tokenizer = tokenizer
        self.model = model.to(self.device).eval()

    
    @torch.no_grad()
    def score(self, query: str, passages: list[str]) -> np.ndarray:
        """Relevance logits s(q, p), higher = more relevant."""
        out = []

        for i in range(0, len(passages), self.batch_size):
            batch = passages[i:i + self.batch_size]

            # 1. Construire les paires (query, passage)
            queries = [query] * len(batch)

            # 2. Tokeniser les paires simultanément
            inputs = self.tokenizer(
                queries,
                batch,
                padding=True,
                truncation="only_second",
                max_length=self.max_length,
                return_tensors="pt",
            )

            # 3. Envoyer les tenseurs sur le device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # 4. Forward pass
            logits = self.model(**inputs).logits

            # 5. Récupérer un score par paire
            scores = logits.view(-1)

            # 6. Convertir en float32 et CPU
            scores = scores.float().cpu()

            out.append(scores)

        return torch.cat(out).numpy() if out else np.zeros(0, dtype=np.float32)
