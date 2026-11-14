"""Dense retrieval: a bi-encoder (text -> unit vector) and exact search (one matrix product)."""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F


def last_token_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """hidden (B, L, d), mask (B, L) with 1 on real tokens -> (B, d): state of the last real token.

    Must work for left AND right padding.
    """
    # position of the last 1 of each mask row = L - 1 - argmax(flipped
    # row); then gather hidden[b, last[b]]
    L = mask.shape[1]
    flipped = mask.flip(dims = [1]).float()
    # trouver la position du premier mask inversé ( on fait ça car on veut la dernière position en réalité)
    # Cependant on ne peut qu'obtenir que la 1ere position donc on inverse puis le premier sera le dernier et on revient
    last_inv = flipped.argmax(dim = 1)
    last_ind = L - 1 - last_inv
    # Récupérer le vecteur du dernier token réel
    batch_ind = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[batch_ind,last_ind]



def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average of the real tokens (the pooling of encoder models such as E5)."""
    # masked sum divided by the number of real tokens (clamp the denominator)
    masked_hidden = hidden * mask.unsqueeze(-1)
    summed = masked_hidden.sum(dim=1)
    count = mask.sum(dim=1, keepdim=True).clamp(min=1)
    return summed / count
    


def _default_device(device: str | None) -> str:
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


class DenseEncoder:
    """Bi-encoder. Default: Qwen3-Embedding-0.6B, a decoder-only model with last-token pooling.

    Queries get a one-sentence instruction; passages are encoded as they are.
    """

    TASK = "Given a question about scientific papers, retrieve the passages that answer the question"

    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B", device: str | None = None,
                 max_length: int = 512, batch_size: int = 16, model=None, tokenizer=None):
        self.model_name = model_name
        self.device = _default_device(device)
        self.max_length, self.batch_size = max_length, batch_size
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        if model is None:
            from transformers import AutoModel
            model = AutoModel.from_pretrained(model_name)
            if self.device.startswith("cuda"):
                model = model.to(dtype=torch.bfloat16)
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = model.to(self.device).eval()
        # Does the tokenizer already end every sequence with EOS? (depends on its version)
        self.add_eos = self.tokenizer("probe")["input_ids"][-1] != self.tokenizer.eos_token_id

    @torch.no_grad()
    def encode(self, texts: list[str], is_query: bool = False) -> torch.Tensor:
        """(len(texts), d) float32 CPU tensor of unit vectors."""
        if is_query:
            texts = [f"Instruct: {self.TASK}\nQuery:{t}" for t in texts]
        out = []
        for i in range(0, len(texts), self.batch_size):
            # tokenize without padding, truncating to max_length - add_eos;
            # append EOS if needed; pad with tokenizer.pad (left); forward;
            # last_token_pool; L2-normalise in float32; move to CPU
            batch = texts[i:i + self.batch_size]

            encoded = self.tokenizer(
                batch,
                truncation=True,
                max_length=self.max_length - int(self.add_eos),
                padding=False,
                return_tensors=None,
            )

            if self.add_eos:
                encoded["input_ids"] = [
                    ids + [self.tokenizer.eos_token_id]
                    for ids in encoded["input_ids"]
                ]

            features = [
                {"input_ids": ids}
                for ids in encoded["input_ids"]
            ]

            batch_inputs = self.tokenizer.pad(
                features,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            hidden = self.model(**batch_inputs).last_hidden_state

            pooled = last_token_pool(
                hidden,
                batch_inputs["attention_mask"],
            )

            pooled = F.normalize(pooled.float(), p=2, dim=1)

            out.append(pooled.cpu())

        return torch.cat(out) if out else torch.empty(0, 0)


class ExactIndex:
    """Exact maximum inner product search: no vector database, one matmul + top-k."""

    def __init__(self, embeddings: torch.Tensor, device: str | None = None):
        self.device = _default_device(device)
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.E = embeddings.to(self.device, dtype)                    # (N, d), unit rows

    @torch.no_grad()
    def search(self, queries: torch.Tensor, k: int, mask: torch.Tensor | None = None):
        """queries (B, d) unit vectors; mask (N,) bool, False = excluded.

        Returns (scores, indices), CPU tensors of shape (B, k') with k' = min(k, #allowed rows).
        """
        # scores = Q E^T in float32; excluded columns -> -inf; k' = min(k,allowed); torch.topk
        Q = queries.to(self.device, dtype=torch.float32)
        E = self.E.to(dtype=torch.float32)

        scores = Q @ E.T

        if mask is not None:
            mask = mask.to(self.device, dtype=torch.bool)
            scores[:, ~mask] = float("-inf")
            allowed = int(mask.sum().item())
        else:
            allowed = scores.shape[1]

        k_eff = min(k, allowed)
        vals, idx = torch.topk(scores, k=k_eff, dim=1)

        return vals.cpu(), idx.cpu()


def encode_chunks_cached(encoder: DenseEncoder, texts: list[str], cache_dir) -> torch.Tensor:
    """Encode the chunk texts once; reload from disk when model and texts are unchanged."""
    key = hashlib.sha1((encoder.model_name + "\x00" + "\x00".join(texts)).encode("utf-8")).hexdigest()[:16]
    path = Path(cache_dir) / f"emb_{key}.pt"
    if path.exists():
        return torch.load(path)
    emb = encoder.encode(texts, is_query=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(emb, path)
    return emb
