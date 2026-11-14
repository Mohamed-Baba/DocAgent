"""Stage 1 tests: Ollama client (P1), Docling pages (P2), ingestion with fallback and cache (P3)."""
import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

import pytest
from PIL import Image

from conftest import make_pdf
from docagent.ingest import doc_to_pages, ingest_corpus
from docagent.llm import OllamaClient, ScriptedLLM
from docagent.schema import load_pages


# ----------------------------------------------------------------------------- P1
class _FakeOllama(BaseHTTPRequestHandler):
    received: list = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _FakeOllama.received.append(body)
        out = {"message": {"role": "assistant", "content": json.dumps({"answer": "42"})},
               "prompt_eval_count": 123, "eval_count": 7}
        self._send(out)

    def do_GET(self):
        self._send({"models": [{"name": "m:tag", "model": "m:tag", "size_vram": 5_000_000_000}]})

    def _send(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture
def fake_server():
    _FakeOllama.received = []
    server = HTTPServer(("127.0.0.1", 0), _FakeOllama)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def test_ollama_payload_and_reply(fake_server, tmp_path):
    img = tmp_path / "p.png"
    img.write_bytes(b"\x89PNG fake bytes")
    client = OllamaClient("m:tag", host=fake_server, num_ctx=4096, seed=3)
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}
    history = [{"role": "system", "content": "sys"},
               {"role": "user", "content": "look", "images": [str(img)]},
               {"role": "assistant", "content": "ok", "images": []}]
    reply = client.chat(history, schema=schema, temperature=0.2, max_tokens=50)
    body = _FakeOllama.received[-1]
    assert body["model"] == "m:tag" and body["stream"] is False
    assert body["format"] == schema
    assert body["options"] == {"temperature": 0.2, "num_ctx": 4096, "seed": 3, "num_predict": 50}
    assert base64.b64decode(body["messages"][1]["images"][0]) == img.read_bytes()
    assert "images" not in body["messages"][2]            # empty image lists are dropped
    assert history[1]["images"] == [str(img)]            # the caller's history is not mutated
    assert reply.json() == {"answer": "42"}
    assert (reply.prompt_tokens, reply.completion_tokens) == (123, 7)
    assert reply.latency_s >= 0
    assert client.vram_bytes() == 5_000_000_000


def test_scripted_llm_records_calls():
    llm = ScriptedLLM([{"a": 1}, "plain"])
    assert llm.chat([{"role": "user", "content": "x"}]).json() == {"a": 1}
    assert llm.chat([{"role": "user", "content": "y"}], schema={"type": "object"}).text == "plain"
    assert len(llm.calls) == 2 and llm.calls[1]["schema"] == {"type": "object"}


# ----------------------------------------------------------------------------- P2
class _FakeDoc:
    """Mimics the parts of DoclingDocument used by doc_to_pages."""

    def __init__(self):
        img = SimpleNamespace(pil_image=Image.new("RGB", (20, 30), "white"))
        self.pages = {2: SimpleNamespace(image=None), 1: SimpleNamespace(image=img)}
        table = SimpleNamespace(prov=[SimpleNamespace(page_no=2)],
                                export_to_markdown=lambda doc=None: "| a | b |\n|---|---|\n| 1 | 2 |")
        orphan = SimpleNamespace(prov=[], export_to_markdown=lambda doc=None: "lost")
        self.tables = [table, orphan]

    def export_to_markdown(self, page_no=None):
        return f"## Section\ntext of page {page_no}"


def test_doc_to_pages(tmp_path):
    pages = doc_to_pages(_FakeDoc(), "paper", tmp_path / "img")
    assert [p.page for p in pages] == [1, 2]                      # sorted, numbered from 1
    assert pages[0].text.endswith("page 1") and pages[1].text.endswith("page 2")
    assert pages[0].tables == [] and pages[1].tables[0].startswith("| a | b |")
    assert pages[0].image_path.endswith("paper_p001.png")
    assert Image.open(pages[0].image_path).size == (20, 30)
    assert pages[1].image_path is None


# ----------------------------------------------------------------------------- P3
class _FakeConverter:
    def convert(self, path):
        if "bad" in path.name:
            raise RuntimeError("layout model crashed")
        return SimpleNamespace(document=_FakeDoc())


def test_ingest_corpus_fallback_failure_and_cache(tmp_path):
    pdfs = tmp_path / "pdfs"
    pdfs.mkdir()
    (pdfs / "good.pdf").write_bytes(make_pdf(["unused"]))
    (pdfs / "bad.pdf").write_bytes(make_pdf(["fallback page one words", "fallback page two"]))
    (pdfs / "bad_corrupt.pdf").write_bytes(b"this is not a pdf")
    out = tmp_path / "pages.jsonl"
    stats = ingest_corpus(pdfs, out, tmp_path / "img", converter=_FakeConverter())
    assert stats["docs"] == 2 and stats["pages"] == 4
    assert stats["fallback"] == ["bad"] and stats["failed"] == ["bad_corrupt"]
    pages = {(p.doc_id, p.page): p for p in load_pages(out)}
    assert "fallback page two" in pages[("bad", 2)].text
    assert pages[("good", 2)].tables
    again = ingest_corpus(pdfs, out, tmp_path / "img", converter=_FakeConverter())
    assert again["docs"] == 0 and again["failed"] == ["bad_corrupt"]   # cached docs are skipped
