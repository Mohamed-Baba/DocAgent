"""Shared test helpers: tiny offline tokenizers and models, a minimal PDF writer, fake encoders."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import docagent without installing

from docagent.bm25 import tokenize  # noqa: E402
from docagent.schema import Page  # noqa: E402

SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[EOS]"]


def make_tokenizer(texts, eos_template: bool = False, pair_template: bool = False,
                   padding_side: str = "left"):
    """Word-level tokenizer built offline from `texts` (no download)."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    from transformers import PreTrainedTokenizerFast

    words = sorted({w for t in texts for w in t.split()} | {"Instruct:", "Query:", "probe"})
    vocab = {tok: i for i, tok in enumerate(SPECIALS + words)}
    tok = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    if eos_template:
        tok.post_processor = TemplateProcessing(single="$A [EOS]", special_tokens=[("[EOS]", vocab["[EOS]"])])
    if pair_template:
        tok.post_processor = TemplateProcessing(
            single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])])
    return PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="[PAD]", unk_token="[UNK]",
                                   eos_token="[EOS]", cls_token="[CLS]", sep_token="[SEP]",
                                   padding_side=padding_side)


def tiny_decoder(vocab_size: int):
    """A 2-layer Qwen3 decoder (RoPE, causal attention), randomly initialised."""
    from transformers import Qwen3Config, Qwen3Model
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=vocab_size, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                      max_position_embeddings=256, pad_token_id=0)
    return Qwen3Model(cfg).eval()


def tiny_cross_encoder(vocab_size: int):
    from transformers import BertConfig, BertForSequenceClassification
    torch.manual_seed(0)
    cfg = BertConfig(vocab_size=vocab_size, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                     intermediate_size=64, max_position_embeddings=256, num_labels=1)
    return BertForSequenceClassification(cfg).eval()


def make_pdf(page_texts: list[str]) -> bytes:
    """A minimal valid PDF (Helvetica, one text line per page): enough for pdfium."""
    n = len(page_texts)
    objs = ["<< /Type /Catalog /Pages 2 0 R >>",
            "<< /Type /Pages /Kids [" + " ".join(f"{4 + 2 * i} 0 R" for i in range(n)) + f"] /Count {n} >>",
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    for i, text in enumerate(page_texts):
        stream = f"BT /F1 12 Tf 72 770 Td ({text}) Tj ET".encode("latin-1")
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
                    f"/Resources << /Font << /F1 3 0 R >> >> /Contents {5 + 2 * i} 0 R >>")
        objs.append(f"<< /Length {len(stream)} >>\nstream\n" + stream.decode("latin-1") + "\nendstream")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for k, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{k} 0 obj\n{body}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


class BowEncoder:
    """Deterministic stand-in for DenseEncoder: normalised bag of words."""
    model_name = "bow"

    def __init__(self, texts):
        self.vocab = {w: i for i, w in enumerate(sorted({w for t in texts for w in tokenize(t)}))}

    def encode(self, texts, is_query=False):
        X = torch.full((len(texts), len(self.vocab)), 1e-6)
        for r, t in enumerate(texts):
            for w in tokenize(t):
                if w in self.vocab:
                    X[r, self.vocab[w]] += 1.0
        return F.normalize(X, dim=-1)


class OverlapReranker:
    """Stand-in for the cross-encoder: number of query words present in the passage."""

    def score(self, query, passages):
        import numpy as np
        q = set(tokenize(query))
        return np.array([float(len(q & set(tokenize(p)))) for p in passages], dtype=np.float32)


@pytest.fixture
def toy_pages():
    return [
        Page("cuturi2013", 1, "sinkhorn distances lightspeed computation of optimal transport entropic regularization"),
        Page("cuturi2013", 2, "the sinkhorn algorithm alternates row and column scalings of the kernel matrix"),
        Page("lipman2023", 1, "flow matching trains continuous normalizing flows by regressing vector fields"),
        Page("lipman2023", 2, "conditional flow matching uses gaussian probability paths between noise and data"),
        Page("tong2023", 3, "minibatch optimal transport couplings straighten the flow matching trajectories"),
    ]
