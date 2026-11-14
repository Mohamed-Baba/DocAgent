"""X1 (D0.2): how often is the output valid JSON, with and without a schema, across temperatures?

    python scripts/json_validity.py --n 20
"""
import argparse
import json

from docagent.config import Config
from docagent.llm import OllamaClient
from docagent.schema import load_pages

SCHEMA = {"type": "object",
          "properties": {"topic": {"type": "string"}, "n_equations": {"type": "integer"},
                         "keywords": {"type": "array", "items": {"type": "string"}}},
          "required": ["topic", "n_equations", "keywords"]}
PROMPT = ("Read this page of a paper and reply ONLY with a JSON object with keys "
          "topic (string), n_equations (integer), keywords (list of strings).\n\n{page}")


def valid(text: str) -> bool:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return False
    return (isinstance(obj, dict) and isinstance(obj.get("topic"), str)
            and type(obj.get("n_equations")) is int and isinstance(obj.get("keywords"), list))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()
    cfg = Config()
    llm = OllamaClient(cfg.llm_model, cfg.ollama_host, num_ctx=8192)
    pages = [p for p in load_pages(cfg.pages_path) if len(p.text.split()) > 100][: args.n]
    print("| temperature | schema | valid | mean latency (s) |\n|---|---|---|---|")
    for temperature in (0.0, 0.7, 1.2):
        for use_schema in (False, True):
            ok, lat = 0, 0.0
            for p in pages:
                msg = [{"role": "user", "content": PROMPT.format(page=p.text[:3000])}]
                r = llm.chat(msg, schema=SCHEMA if use_schema else None, temperature=temperature)
                ok += valid(r.text)
                lat += r.latency_s
            print(f"| {temperature} | {use_schema} | {ok}/{len(pages)} | {lat / len(pages):.2f} |")
    print(f"\nVRAM used by the model: {llm.vram_bytes() / 1e9:.2f} GB")
