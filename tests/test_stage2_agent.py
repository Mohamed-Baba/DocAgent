"""Stage 2 tests: validation (P16), calculator (P17), corpus tools (P18-P19),
JSON extraction and action parsing (P20-P21), the agent loop (P22)."""
import json

import pytest

from conftest import BowEncoder, OverlapReranker
from docagent.agent import (Agent, ActionError, FINAL_ANSWER_SCHEMA, extract_first_json,
                            parse_action)
from docagent.bm25 import BM25
from docagent.chunking import chunk_pages
from docagent.dense import ExactIndex
from docagent.llm import ScriptedLLM
from docagent.retrieval import Retriever
from docagent.schema import Page, PageRef
from docagent.tools import (CalculatorTool, ExtractTableTool, InspectPageTool, SearchTool, ToolBox,
                            safe_eval, validate_args)

SCHEMA = {"type": "object",
          "properties": {"query": {"type": "string"}, "page": {"type": "integer", "minimum": 1},
                         "mode": {"type": "string", "enum": ["a", "b"]}, "flag": {"type": "boolean"}},
          "required": ["query"]}


# ----------------------------------------------------------------------------- P16
def test_validate_args():
    assert validate_args({"query": "x", "page": 2, "mode": "a", "flag": False}, SCHEMA) == []
    assert validate_args("x", SCHEMA) == ["args must be a JSON object"]
    errs = validate_args({"page": 0, "mode": "c", "other": 1}, SCHEMA)
    assert any("missing required argument 'query'" in e for e in errs)
    assert any("unexpected argument 'other'" in e for e in errs)
    assert any("'mode' must be one of" in e for e in errs)
    assert any("'page' must be >= 1" in e for e in errs)
    assert validate_args({"query": "x", "page": True}, SCHEMA)          # bool is not an integer
    assert validate_args({"query": "x", "page": 2.0}, SCHEMA)           # 2.0 is not an integer
    assert validate_args({"query": 3}, SCHEMA)


# ----------------------------------------------------------------------------- P17
def test_safe_eval_computes():
    assert safe_eval("2 + 3 * 4") == 14
    assert safe_eval("-(2 ** 10) // 3") == -342
    assert safe_eval("sqrt(16) + log(e) + max(1, 2, 3)") == pytest.approx(8.0)
    assert safe_eval("round(pi, 2)") == 3.14


@pytest.mark.parametrize("expr", ["__import__('os').system('echo hi')", "().__class__", "x + 1",
                                  "open('f')", "lambda: 1", "[1, 2]", "True + 1", "9 ** 9 ** 9",
                                  "sqrt(x=4)"])
def test_safe_eval_refuses(expr):
    with pytest.raises(ValueError):
        safe_eval(expr)


def test_calculator_tool_reports_errors():
    calc = CalculatorTool()
    assert calc(expression="1/4").text == "1/4 = 0.25"
    for bad in ["1/0", "sqrt(-1)", "10.0 ** 400", "2 +"]:
        r = calc(expression=bad)
        assert r.error and r.text.startswith("Error:")


# ----------------------------------------------------------------------------- P18-P19
@pytest.fixture
def corpus(toy_pages, tmp_path):
    img = tmp_path / "cuturi2013_p002.png"
    img.write_bytes(b"png")
    pages = list(toy_pages)
    pages[1] = Page("cuturi2013", 2, pages[1].text, tables=["| eps | time |\n|---|---|\n| 0.1 | 2s |"],
                    image_path=str(img))
    chunks = chunk_pages(pages, 50, 10)
    texts = [c.text for c in chunks]
    enc = BowEncoder(texts)
    retriever = Retriever(chunks, BM25().fit(texts), enc, ExactIndex(enc.encode(texts), device="cpu"),
                          OverlapReranker(), n_candidates=10)
    return pages, retriever


def test_search_tool(corpus):
    pages, retriever = corpus
    tool = SearchTool(retriever, {p.doc_id for p in pages}, k=3)
    r = tool(query="sinkhorn algorithm scalings")
    assert r.text.startswith("[1] cuturi2013 p.2") and r.pages[0] == PageRef("cuturi2013", 2)
    assert len(r.pages) == len(set(r.pages))
    r = tool(query="flow matching", doc_id="tong2023")
    assert {p.doc_id for p in r.pages} == {"tong2023"}
    r = tool(query="x", doc_id="cuturi2031")
    assert r.error and "cuturi2013" in r.text


def test_page_tools(corpus):
    pages, _ = corpus
    inspect, tables = InspectPageTool(pages), ExtractTableTool(pages)
    r = inspect(doc_id="cuturi2013", page=2)
    assert r.text.startswith("cuturi2013 p.2") and r.images and r.pages == [PageRef("cuturi2013", 2)]
    assert inspect(doc_id="cuturi2013", page=9).text == "Error: cuturi2013 has pages 1 to 2."
    assert inspect(doc_id="nope", page=1).error
    assert InspectPageTool(pages, with_image=False)(doc_id="cuturi2013", page=2).images == []
    assert "| eps | time |" in tables(doc_id="cuturi2013", page=2).text
    assert tables(doc_id="lipman2023", page=1).text.startswith("No table")


def test_toolbox(corpus):
    pages, retriever = corpus
    box = ToolBox([SearchTool(retriever, {p.doc_id for p in pages}), CalculatorTool()])
    assert "- search(query: string, doc_id: string (optional)):" in box.describe()
    assert box.action_schema()["properties"]["action"]["enum"] == ["search", "calculator", "final_answer"]
    assert box.action_schema(final_only=True)["properties"]["action"]["enum"] == ["final_answer"]
    r = box.call("calculator", {"wrong": "1"})           # TypeError inside the tool -> observation
    assert r.error and "TypeError" in r.text


# ----------------------------------------------------------------------------- P20-P21
def test_extract_first_json():
    assert extract_first_json('Sure! ```json\n{"a": {"b": 1}}\n``` done') == {"a": {"b": 1}}
    assert extract_first_json('{"s": "a } brace and \\" quote"} tail') == {"s": 'a } brace and " quote'}
    assert extract_first_json('{broken {"ok": true}') == {"ok": True}
    assert extract_first_json("[1, 2] no object") is None


def test_parse_action(corpus):
    pages, retriever = corpus
    box = ToolBox([SearchTool(retriever, {p.doc_id for p in pages}), CalculatorTool()])
    a = parse_action('{"thought": "t", "action": "search", "args": {"query": "q"}}', box)
    assert (a.action, a.args) == ("search", {"query": "q"})
    a = parse_action('blah {"thought": "", "action": "final_answer", "args": {"answer": "1", '
                     '"citations": [{"doc_id": "d", "page": 1}], "abstain": false}}', box)
    assert a.action == "final_answer"
    for bad, msg in [("no json", "not a JSON object"),
                     ('{"action": "fly", "args": {}}', "unknown action"),
                     ('{"action": "search", "args": {"q": "x"}}', "invalid args for search"),
                     ('{"action": "final_answer", "args": {"answer": "1", "citations": [{"doc_id": "d", '
                      '"page": "1"}], "abstain": false}}', "each citation")]:
        with pytest.raises(ActionError, match=msg):
            parse_action(bad, box)


# ----------------------------------------------------------------------------- P22
def _agent(corpus, replies, **kw):
    pages, retriever = corpus
    llm = ScriptedLLM(replies)
    box = ToolBox([SearchTool(retriever, {p.doc_id for p in pages}), InspectPageTool(pages),
                   ExtractTableTool(pages), CalculatorTool()])
    return Agent(llm, box, **kw), llm


def _act(action, **args):
    return {"thought": f"use {action}", "action": action, "args": args}


FINAL = _act("final_answer", answer="row and column scalings",
             citations=[{"doc_id": "cuturi2013", "page": 2}], abstain=False)


def test_agent_happy_path(corpus):
    agent, llm = _agent(corpus, [_act("search", query="sinkhorn algorithm"),
                                 _act("inspect_page", doc_id="cuturi2013", page=2), FINAL])
    res = agent.run("What does the Sinkhorn algorithm alternate?")
    assert res.stop_reason == "final_answer" and res.answer == "row and column scalings"
    assert res.citations == [PageRef("cuturi2013", 2)] and PageRef("cuturi2013", 2) in res.consulted
    assert [s.action for s in res.steps] == ["search", "inspect_page", "final_answer"]
    last = llm.calls[-1]["messages"]
    assert last[0]["role"] == "system" and "inspect_page(doc_id: string, page: integer)" in last[0]["content"]
    assert last[-1]["role"] == "user" and last[-1]["content"].startswith("Observation 2:")
    assert last[-1]["images"]                                         # the page image is shown
    assert llm.calls[0]["schema"]["properties"]["action"]["enum"][-1] == "final_answer"
    pred = res.to_prediction()
    assert pred.n_llm_calls == 3 and pred.meta["stop_reason"] == "final_answer"
    json.dumps(pred.to_dict())                                        # serialisable trace


def test_agent_recovers_then_stops_on_errors(corpus):
    agent, llm = _agent(corpus, ["I think the answer is 42", FINAL])
    res = agent.run("q")
    assert res.stop_reason == "final_answer" and res.steps[0].error
    assert "Error: the reply is not a JSON object" in llm.calls[1]["messages"][-1]["content"]
    agent, _ = _agent(corpus, ["bad"] * 3, max_errors=3)
    res = agent.run("q")
    assert res.stop_reason == "max_errors" and res.abstain and len(res.steps) == 3


def test_agent_budget_forces_final_answer(corpus):
    agent, llm = _agent(corpus, [_act("search", query="sinkhorn"), _act("search", query="sinkhorn again"),
                                 FINAL], max_steps=2)
    res = agent.run("q")
    assert res.stop_reason == "max_steps" and res.answer == "row and column scalings"
    assert llm.calls[-1]["schema"]["properties"]["action"]["enum"] == ["final_answer"]
    assert "budget exhausted" in llm.calls[-1]["messages"][-1]["content"]


def test_agent_keeps_only_newest_image_and_uses_feedback(corpus):
    agent, llm = _agent(corpus, [_act("inspect_page", doc_id="cuturi2013", page=2),
                                 _act("inspect_page", doc_id="cuturi2013", page=2),
                                 _act("final_answer", answer="", citations=[], abstain=True)])
    res = agent.run("q", feedback="your citation was wrong")
    msgs = llm.calls[-1]["messages"]
    assert "Feedback on your previous attempt: your citation was wrong" in msgs[1]["content"]
    with_images = [m for m in msgs if m.get("images")]
    assert len(with_images) == 1 and with_images[0] is msgs[-1]
    assert "[page image removed from the context]" in msgs[3]["content"]
    assert res.abstain and res.citations == []
    assert FINAL_ANSWER_SCHEMA["required"] == ["answer", "citations", "abstain"]
