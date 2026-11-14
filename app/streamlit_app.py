"""D6.1: minimal demo (plumbing).  streamlit run app/streamlit_app.py"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docagent.agent import Agent  # noqa: E402
from docagent.config import Config  # noqa: E402
from docagent.llm import OllamaClient  # noqa: E402
from docagent.retrieval import build_retriever  # noqa: E402
from docagent.schema import load_pages  # noqa: E402
from docagent.tools import default_toolbox  # noqa: E402
from docagent.verifier import VerifiedAgent, Verifier  # noqa: E402


@st.cache_resource
def load_system():
    cfg = Config()
    pages = load_pages(cfg.pages_path)
    retriever = build_retriever(cfg, pages)
    llm = OllamaClient(cfg.llm_model, cfg.ollama_host, cfg.num_ctx)
    agent = Agent(llm, default_toolbox(retriever, pages), cfg.max_steps, cfg.max_errors)
    return VerifiedAgent(agent, Verifier(llm, pages)), {p.ref: p for p in pages}


st.set_page_config(page_title="docagent", layout="wide")
st.title("Research assistant over scientific papers")
system, pages = load_system()
question = st.text_input("Question", placeholder="Which coupling makes flow matching paths straighter?")
if st.button("Ask", type="primary") and question:
    with st.spinner("Searching, reading, verifying..."):
        pred = system.run(question)
    if pred.abstain:
        st.warning("Abstention: the corpus does not support an answer.")
    else:
        st.success(pred.answer)
        cols = st.columns(max(1, len(pred.citations)))
        for col, ref in zip(cols, pred.citations):
            with col:
                st.caption(str(ref))
                page = pages.get(ref)
                if page and page.image_path:
                    st.image(page.image_path)
    st.caption(f"{pred.n_llm_calls} model calls | {pred.prompt_tokens + pred.completion_tokens} tokens | "
               f"{pred.latency_s:.1f} s")
    with st.expander("Trace"):
        for step in pred.trace:
            st.markdown(f"**step {step['index'] + 1}: {step['action'] or 'invalid reply'}**")
            st.code(step["reply"], language="json")
            if step["observation"]:
                st.text(step["observation"][:1500])
        st.json(pred.meta)
