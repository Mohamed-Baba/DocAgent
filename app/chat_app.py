"""Chat with the corpus: a ChatGPT-like page, with your notes and example questions in the sidebar.

    streamlit run app/chat_app.py          (from the root of the repository, with Ollama running)

Each question is answered on its own, without memory of the previous ones, by the configuration chosen
in the sidebar, built exactly as in the benchmark (evaluate.make_runner). The notes stay on your machine
in app/notes.md: they are never sent to the model.
"""
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from docagent.config import Config  # noqa: E402
from docagent.evaluate import make_runner  # noqa: E402
from docagent.llm import OllamaClient  # noqa: E402
from docagent.retrieval import build_retriever  # noqa: E402
from docagent.schema import Prediction, load_pages, load_questions  # noqa: E402

NOTES = ROOT / "app" / "notes.md"
CONFIGS = {"Agent + verifier": "agent_verified", "Agent": "agent", "RAG (hybrid)": "rag_hybrid"}


@st.cache_resource(show_spinner="Loading the corpus and the models...")
def load_system():
    cfg = Config()
    pages = load_pages(cfg.pages_path)
    retriever = build_retriever(cfg, pages)
    llm = OllamaClient(cfg.llm_model, cfg.ollama_host, cfg.num_ctx)
    runners = {name: make_runner(name, cfg, pages, retriever, llm) for name in CONFIGS.values()}
    examples = [q.question for q in load_questions(cfg.questions_path)] if cfg.questions_path.exists() else []
    return runners, {p.ref: p for p in pages}, examples


def save_notes() -> None:
    NOTES.write_text(st.session_state.notes, encoding="utf-8")


def show_answer(msg: dict, pages: dict) -> None:
    if "error" in msg:
        st.error(msg["error"])
        return
    pred = Prediction.from_dict(msg["pred"])
    if pred.abstain:
        st.warning("No answer: the corpus does not support one (abstention).")
    else:
        st.markdown(f"**{pred.answer}**")
        st.caption("Sources: " + ", ".join(map(str, pred.citations)))
    st.caption(f"{msg['config']} · {pred.n_llm_calls} model calls · "
               f"{pred.prompt_tokens + pred.completion_tokens} tokens · {pred.latency_s:.0f} s")
    if pred.citations:
        with st.expander("Cited pages"):
            for ref in pred.citations:
                page = pages.get(ref)
                if page is not None and page.image_path and Path(page.image_path).exists():
                    st.image(page.image_path, caption=str(ref))
                else:
                    st.caption(f"{ref}: no page image")
    with st.expander("How the answer was found"):
        if not pred.trace:                                   # RAG: one call on the retrieved passages
            st.caption("Passages retrieved from: " + (", ".join(map(str, pred.consulted)) or "-"))
        for k, step in enumerate(pred.trace):
            if k > 0 and step["index"] == 0:                 # the verifier asked for a new attempt
                st.markdown("*New attempt, after the verifier's feedback*")
            st.markdown(f"**{step['index'] + 1}. {step['action'] or 'invalid reply'}** `{step['args'] or ''}`")
            if step["error"]:
                st.caption(f"error: {step['error']}")
            elif step["observation"]:
                st.text(step["observation"][:800])
        for k, verdict in enumerate(pred.meta.get("verdicts", []), start=1):
            for ref, why in verdict["rejected"]:
                st.caption(f"Verifier, attempt {k}: {ref} rejected ({why})")


st.set_page_config(page_title="docagent", layout="wide")
runners, pages, examples = load_system()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "notes" not in st.session_state:
    st.session_state.notes = NOTES.read_text(encoding="utf-8") if NOTES.exists() else ""

picked = None
with st.sidebar:
    config = st.radio("Configuration", list(CONFIGS))
    st.text_area("Notes (saved in app/notes.md, never sent to the model)", key="notes", height=240,
                 on_change=save_notes)
    if examples:
        with st.expander("Example questions"):
            for i, q in enumerate(examples):
                if st.button(q, key=f"example_{i}"):
                    picked = q
    if st.button("New conversation"):
        st.session_state.messages = []

st.title("docagent")
st.caption("Questions about the papers of the corpus. Every answer cites its pages, or the system abstains. "
           "Each question is answered on its own.")
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            show_answer(msg, pages)

question = st.chat_input("Ask a question about the corpus") or picked
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching, reading, checking..."):
            try:
                msg = {"role": "assistant", "config": config, "pred": runners[CONFIGS[config]](question).to_dict()}
            except Exception as e:                           # Ollama stopped, a model missing...
                msg = {"role": "assistant", "error": f"{type(e).__name__}: {e}. Is Ollama running?"}
        show_answer(msg, pages)
    st.session_state.messages.append(msg)
