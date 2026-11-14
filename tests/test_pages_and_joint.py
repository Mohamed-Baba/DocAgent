"""Long pages read in parts (PagedInspectPageTool), the judge's page window, joint verification."""
import json
import random

import pytest

from docagent.agent import Agent
from docagent.config import Config
from docagent.evaluate import make_runner
from docagent.llm import ScriptedLLM
from docagent.schema import Page
from docagent.tools import (CalculatorTool, InspectPageTool, PagedInspectPageTool, ToolBox, default_toolbox,
                            split_page, validate_args)
from docagent.verifier import (JOINT_SCHEMA, JOINT_SYSTEM, JUDGE_SCHEMA, VerifiedAgent, Verifier)

D = "41586_2024_Article_8328"
# a page shaped like p.10 of the TabPFN paper: the fact sits at ~6,900 characters
FILLER = "\n".join(f"Line {i} of the methods section, with words and the number {i}.0 in it." for i in range(100))
CAP = "We sample the number of training samples for each dataset uniformly up to 2,048 and use a fixed validation set size of 128."
LONG = Page(D, 10, FILLER + "\n" + CAP + "\nThe end of the page.", [], None)
RANGE = Page(D, 1, "TabPFN outperforms all previous methods on datasets with up to 10,000 samples by a wide margin.")
SHORT = Page(D, 2, "A short page about the 2D TabPFN layer.")


def act(action, **args):
    return {"thought": "t", "action": action, "args": args}


# ----------------------------------------------------------------------------- split_page
def test_split_page_short_and_empty():
    assert split_page("abc def", 100) == ["abc def"]
    assert split_page("", 100) == [""]


def test_split_page_never_cuts_a_token():
    rng = random.Random(0)
    for _ in range(300):
        words = ["".join(rng.choices("ab,.0123456789", k=rng.randint(1, 12))) for _ in range(rng.randint(0, 400))]
        text = "".join(w + rng.choice([" ", "\n", "  "]) for w in words)
        size = rng.randint(30, 600)
        parts = split_page(text, size)
        assert all(len(p) <= size for p in parts)
        assert " ".join(parts).split() == text.split()           # every token, whole, in order


def test_split_page_without_spaces_cuts_at_size():
    assert split_page("x" * 25, 10) == ["x" * 10, "x" * 10, "x" * 5]


# ----------------------------------------------------------------------------- PagedInspectPageTool
def test_short_page_is_unchanged(tmp_path):
    img = tmp_path / "p2.png"
    img.write_bytes(b"png")
    page = Page(D, 2, SHORT.text, [], str(img))
    old, new = InspectPageTool([page])(D, 2), PagedInspectPageTool([page])(D, 2)
    assert (new.text, new.pages, new.images) == (old.text, old.pages, old.images)


def test_long_page_in_parts(tmp_path):
    img = tmp_path / "p10.png"
    img.write_bytes(b"png")
    page = Page(D, 10, LONG.text, [], str(img))
    assert len(page.text) > 6000 and page.text.index("2,048") > 6000          # out of reach before
    assert "2,048" not in InspectPageTool([page])(D, 10).text[:5000]           # what the agent saw before
    tool = PagedInspectPageTool([page])
    one, two = tool(D, 10), tool(D, 10, part=2)
    assert one.text.startswith(f"{D} p.10 (part 1/2)\n") and one.text.endswith("continues in part 2]")
    assert two.text.startswith(f"{D} p.10 (part 2/2)\n") and "2,048" in two.text and "continues" not in two.text
    assert max(len(one.text), len(two.text)) <= 5000                           # the agent cuts at 5000
    assert one.images == [str(img)] and two.images == []                       # the image once
    assert one.pages == two.pages == [page.ref]
    err = tool(D, 10, part=3)
    assert err.error and "2 parts" in err.text
    assert tool("nope", 1).error and tool(D, 99).error                        # _lookup errors, unchanged


def test_part_argument_is_validated():
    schema = PagedInspectPageTool([LONG]).parameters
    assert validate_args({"doc_id": D, "page": 10}, schema) == []
    assert validate_args({"doc_id": D, "page": 10, "part": 2}, schema) == []
    assert validate_args({"doc_id": D, "page": 10, "part": 0}, schema) == ["'part' must be >= 1"]
    assert validate_args({"doc_id": D, "page": 10, "part": "2"}, schema) == ["'part' must be of type integer, got str"]


def test_agent_reads_the_second_part():
    tb = ToolBox([PagedInspectPageTool([LONG, RANGE]), CalculatorTool()])
    llm = ScriptedLLM([act("inspect_page", doc_id=D, page=10),
                       act("inspect_page", doc_id=D, page=10, part=2),
                       act("calculator", expression="10000/2048"),
                       act("final_answer", answer="4.88", abstain=False,
                           citations=[{"doc_id": D, "page": 10}, {"doc_id": D, "page": 1}])])
    res = Agent(llm, tb).run("How many times larger?")
    assert res.answer == "4.88" and [s.error for s in res.steps] == [None] * 4
    seen = llm.calls[-1]["messages"]
    assert "2,048" in seen[-3]["content"] and "[... truncated" not in json.dumps(seen)
    assert "part: integer (optional)" in seen[0]["content"]                    # the system prompt shows it


def test_default_toolbox_and_runner_wiring():
    cfg = Config()
    tb = default_toolbox(None, [LONG], cfg.page_part_chars)
    assert isinstance(tb.tools["inspect_page"], PagedInspectPageTool) and tb.tools["inspect_page"].max_chars == 4800
    runner = make_runner("agent_verified", cfg, [LONG], None, ScriptedLLM([]))
    assert isinstance(runner.__self__, VerifiedAgent)
    verifier = runner.__self__.verifier
    assert verifier.joint is True and verifier.max_chars == cfg.judge_chars == 12000


# ----------------------------------------------------------------------------- the judge's window
def test_judge_sees_the_end_of_a_long_page():
    for max_chars, visible in ((6000, False), (12000, True)):
        llm = ScriptedLLM([{"supported": False, "quote": ""}])
        Verifier(llm, [LONG], max_chars=max_chars).check("Q", "2048", [LONG.ref], consulted=[LONG.ref])
        assert ("2,048" in llm.calls[0]["messages"][1]["content"]) is visible


# ----------------------------------------------------------------------------- joint verification
NO = {"supported": False, "quote": ""}
REFS = [LONG.ref, RANGE.ref]
Q, A = "How many times larger is the 10,000-sample limit than the cap?", "4.88"
QUOTE_CAP = "uniformly up to 2,048 and use a fixed validation set size of 128"
QUOTE_RANGE = "on datasets with up to 10,000 samples"


def joint(supported=True, quotes=(QUOTE_CAP, QUOTE_RANGE)):
    return {"supported": supported, "quotes": list(quotes)}


def test_joint_accepts_a_computation_across_two_pages():
    llm = ScriptedLLM([NO, NO, joint()])
    v = Verifier(llm, [LONG, RANGE], max_chars=12000, joint=True).check(Q, A, REFS, consulted=REFS)
    assert v.ok and v.supported == REFS and v.rejected == [] and v.n_calls == 3
    call = llm.calls[2]
    assert call["schema"] == JOINT_SCHEMA and call["messages"][0]["content"] == JOINT_SYSTEM
    user = call["messages"][1]["content"]
    assert "2,048" in user and "10,000 samples" in user and f"Page {LONG.ref}:" in user


@pytest.mark.parametrize("reply", [
    joint(False),                                            # the judge says no
    joint(quotes=(QUOTE_CAP, "")),                           # one page only: it already failed alone
    joint(quotes=(QUOTE_CAP, "on datasets with up to 50,000 samples")),   # a number not on the page
    joint(quotes=(QUOTE_RANGE, QUOTE_CAP)),                  # quotes in the wrong order
    "not json",
    {"supported": True, "quotes": "a string"},
])
def test_joint_rejections(reply):
    v = Verifier(ScriptedLLM([NO, NO, reply]), [LONG, RANGE], max_chars=12000, joint=True).check(
        Q, A, REFS, consulted=REFS)
    assert not v.ok and v.supported == [] and v.n_calls == 3
    assert v.rejected == [(r, "judge: not supported, alone or together") for r in REFS]


def test_no_joint_call_when_not_needed():
    ok = {"supported": True, "quote": QUOTE_CAP}
    for replies, cited, consulted, joint_flag in (
            ([NO, NO], REFS, REFS, False),                   # joint off: the per-page verdicts, unchanged
            ([ok, NO], REFS, REFS, True),                    # one page is enough alone
            ([NO], REFS, [LONG.ref], True)):                 # the other page fails a free check
        llm = ScriptedLLM(replies)                           # a third call would raise
        Verifier(llm, [LONG, RANGE], max_chars=12000, joint=joint_flag).check(Q, A, cited, consulted)
        assert len(llm.calls) == len(replies) and all(c["schema"] == JUDGE_SCHEMA for c in llm.calls)


def test_joint_off_keeps_the_old_reasons():
    v = Verifier(ScriptedLLM([NO, NO]), [LONG, RANGE], max_chars=12000).check(Q, A, REFS, consulted=REFS)
    assert v.rejected == [(r, "judge: not supported") for r in REFS] and v.n_calls == 2


def test_joint_budget_cuts_each_page():
    big = [Page("big", i, f"page {i} " + "word " * 20000) for i in (1, 2, 3, 4)]
    refs = [p.ref for p in big]
    llm = ScriptedLLM([NO] * 4 + [joint(False)])
    v = Verifier(llm, big, max_chars=12000, joint=True).check("Q", "A", refs, consulted=refs)
    user = llm.calls[-1]["messages"][1]["content"]
    assert len(user) < 30000 + 500 and "big p.4" not in user           # 3 pages at most, 10,000 chars each
    assert [why for _, why in v.rejected] == ["judge: not supported, alone or together"] * 3 + ["judge: not supported"]


def test_verified_agent_keeps_both_pages():
    tb = ToolBox([PagedInspectPageTool([LONG, RANGE]), CalculatorTool()])
    final = act("final_answer", answer="4.88", abstain=False,
                citations=[{"doc_id": D, "page": 10}, {"doc_id": D, "page": 1}])
    llm = ScriptedLLM([act("inspect_page", doc_id=D, page=10, part=2), act("inspect_page", doc_id=D, page=1),
                       final, NO, NO, joint()])
    pred = VerifiedAgent(Agent(llm, tb), Verifier(llm, [LONG, RANGE], max_chars=12000, joint=True)).run(Q)
    assert (pred.answer, pred.abstain, pred.citations) == ("4.88", False, REFS)
    assert pred.meta["attempts"] == 1 and pred.n_llm_calls == 6
