"""D4: ask one question to the agent and print a readable trace.

    python scripts/ask.py "Which coupling makes the flow matching paths straighter?"
"""
import argparse

from docagent.agent import Agent
from docagent.config import Config
from docagent.llm import OllamaClient
from docagent.retrieval import build_retriever
from docagent.schema import load_pages
from docagent.tools import default_toolbox

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args()

    cfg = Config()
    pages = load_pages(cfg.pages_path)
    retriever = build_retriever(cfg, pages)
    llm = OllamaClient(cfg.llm_model, cfg.ollama_host, cfg.num_ctx)
    agent = Agent(llm, default_toolbox(retriever, pages),
                  max_steps=args.max_steps or cfg.max_steps, max_errors=cfg.max_errors)
    res = agent.run(args.question)
    for s in res.steps:
        print(f"--- step {s.index + 1}  [{s.prompt_tokens} prompt tok | {s.completion_tokens} gen tok | {s.latency_s:.1f} s]")
        print(s.reply)
        if s.error:
            print("ERROR:", s.error)
        if s.observation:
            print(">>>", s.observation[:800].replace("\n", "\n    "))
    print("=== stop reason:", res.stop_reason)
    print("answer:", "(abstain)" if res.abstain else res.answer)
    print("citations:", ", ".join(map(str, res.citations)) or "-")
    print("consulted:", ", ".join(map(str, res.consulted)) or "-")
