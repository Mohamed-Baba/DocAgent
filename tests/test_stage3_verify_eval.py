"""Stage 3 tests: quote grounding (P23), verifier (P24), verified agent (P25),
answer metrics (P26-P27), McNemar and Holm (P28-P29), pass^k / pass@k (P30), evaluation glue."""
import math

import pytest
from scipy.stats import binomtest

from docagent.agent import Agent
from docagent.evaluate import compare, run_config, score_all, summarize
from docagent.llm import ScriptedLLM
from docagent.metrics_answers import (citation_precision, citation_recall, exact_match, holm,
                                      is_correct, mcnemar_exact, normalize_answer, numeric_match,
                                      parse_number, pass_at_k, pass_hat_k, token_f1)
from docagent.schema import Page, PageRef, Prediction, Question
from docagent.tools import InspectPageTool, ToolBox
from docagent.verifier import Verdict, VerifiedAgent, Verifier, quote_is_grounded

PAGE = Page("cuturi2013", 2, "## 3 Sinkhorn\nThe **Sinkhorn** algorithm alternates row and column "
                             "scalings of the kernel matrix K = exp(-C/eps). | eps | time |\n| 0.1 | 2s |")
OTHER = Page("lipman2023", 1, "Flow matching regresses vector fields of probability paths.")


# ----------------------------------------------------------------------------- P23
def test_quote_grounding():
    assert quote_is_grounded("the Sinkhorn algorithm alternates row and column scalings", PAGE.text)
    assert quote_is_grounded("Sinkhorn algorithm alternates rows and column scalings of the kernel", PAGE.text)
    assert not quote_is_grounded("the algorithm converges in one iteration for all costs", PAGE.text)
    assert not quote_is_grounded("Sinkhorn", PAGE.text)                     # too short
    assert not quote_is_grounded("ab c d e", "xab c d e")                   # whole tokens only


# ----------------------------------------------------------------------------- P24
def _judge(*verdicts):
    return ScriptedLLM([{"supported": s, "quote": q} for s, q in verdicts])


def test_verifier_detects_false_citations():
    seen = [PAGE.ref, OTHER.ref]
    v = Verifier(_judge((True, "alternates row and column scalings of the kernel matrix")), [PAGE, OTHER])
    ok = v.check("What does Sinkhorn alternate?", "row and column scalings", [PAGE.ref, PAGE.ref], seen)
    assert ok.ok and ok.supported == [PAGE.ref] and ok.n_calls == 1      # duplicates checked once
    v = Verifier(_judge(), [PAGE, OTHER])
    bad = v.check("q", "a", [PageRef("cuturi2013", 99), OTHER.ref], consulted=[PAGE.ref])
    assert not bad.ok and [why for _, why in bad.rejected] == ["page does not exist", "page was never consulted"]
    v = Verifier(_judge((False, ""), (True, "the kernel is computed by a neural network")), [PAGE, OTHER])
    bad = v.check("q", "a", [OTHER.ref, PAGE.ref], seen)
    assert [why for _, why in bad.rejected] == ["judge: not supported", "judge quote not found on the page"]
    v = Verifier(ScriptedLLM(["not json"]), [PAGE])
    assert v.check("q", "a", [PAGE.ref], [PAGE.ref]).rejected[0][1] == "judge reply is not JSON"


# ----------------------------------------------------------------------------- P25
def _final(answer, ref=None, abstain=False):
    cits = [] if ref is None else [{"doc_id": ref.doc_id, "page": ref.page}]
    return {"thought": "", "action": "final_answer", "args": {"answer": answer, "citations": cits, "abstain": abstain}}


def _inspect(ref):
    return {"thought": "", "action": "inspect_page", "args": {"doc_id": ref.doc_id, "page": ref.page}}


def _verified(agent_replies, judge_replies):
    agent_llm = ScriptedLLM(agent_replies)
    agent = Agent(agent_llm, ToolBox([InspectPageTool([PAGE, OTHER], with_image=False)]))
    return VerifiedAgent(agent, Verifier(_judge(*judge_replies), [PAGE, OTHER])), agent_llm


def test_verified_agent_repairs_once():
    system, agent_llm = _verified(
        [_inspect(OTHER.ref), _final("row and column scalings", OTHER.ref),      # attempt 1: wrong page
         _inspect(PAGE.ref), _final("row and column scalings", PAGE.ref)],       # attempt 2: right page
        [(False, ""), (True, "alternates row and column scalings")])
    pred = system.run("What does Sinkhorn alternate?")
    assert not pred.abstain and pred.citations == [PAGE.ref] and pred.meta["attempts"] == 2
    assert "do not support your answer (lipman2023 p.1: judge: not supported)" in agent_llm.calls[2]["messages"][1]["content"]
    assert pred.n_llm_calls == 4 + 2                                             # agent calls + judge calls


def test_verified_agent_abstains():
    system, _ = _verified([_final("42", OTHER.ref), _final("42")], [(False, "")])
    pred = system.run("q")                    # page never consulted, then no citation at all
    assert pred.abstain and pred.answer == "" and pred.citations == [] and pred.meta["attempts"] == 2
    system, _ = _verified([_final("", abstain=True)], [])
    assert system.run("q").abstain


# ----------------------------------------------------------------------------- P26-P27
def test_text_metrics():
    assert normalize_answer("The  Sinkhorn-Knopp algorithm!") == "sinkhornknopp algorithm"
    assert exact_match("the Sinkhorn algorithm", ["Sinkhorn algorithm", "SK"]) == 1.0
    assert exact_match("anything", []) == 0.0
    assert token_f1("not convex", "convex") == pytest.approx(2 / 3)               # high F1, wrong answer
    assert token_f1("", "convex") == 0.0


@pytest.mark.parametrize("s,x", [("about 1,234.5 steps", 1234.5), ("\u22120.5", -0.5), (".5", 0.5),
                                 ("lr = 1.2e-3", 1.2e-3), ("87.3%", 87.3), ("none", None)])
def test_parse_number(s, x):
    assert parse_number(s) == (None if x is None else pytest.approx(x))


def test_is_correct_and_citations():
    q_num = Question("q1", "multistep", "?", ["0.873"], [PAGE.ref], "number")
    q_txt = Question("q2", "simple", "?", ["row and column scalings"], [PAGE.ref])
    q_none = Question("q3", "unanswerable", "?", [], [])
    pred = Prediction("0.87", False, [PAGE.ref, OTHER.ref], [PAGE.ref])
    assert numeric_match("0.87", "0.873") and not numeric_match("0.85", "0.873")
    assert is_correct(pred, q_num) and not is_correct(pred, q_txt) and not is_correct(pred, q_none)
    abst = Prediction("", True, [], [])
    assert is_correct(abst, q_none) and not is_correct(abst, q_num)
    assert citation_precision(pred, q_num) == 0.5 and citation_recall(pred, q_num) == 1.0
    assert math.isnan(citation_precision(abst, q_num)) and math.isnan(citation_recall(pred, q_none))


# ----------------------------------------------------------------------------- P28-P30
@pytest.mark.parametrize("n01,n10", [(6, 0), (3, 3), (7, 2), (0, 0), (1, 9)])
def test_mcnemar_matches_binomial_test(n01, n10):
    a = [True] * n01 + [False] * n10 + [True] * 5 + [False] * 4
    b = [False] * n01 + [True] * n10 + [True] * 5 + [False] * 4
    m01, m10, p = mcnemar_exact(a, b)
    assert (m01, m10) == (n01, n10)
    ref = 1.0 if n01 + n10 == 0 else binomtest(n01, n01 + n10, 0.5).pvalue
    assert p == pytest.approx(ref)
    if (n01, n10) == (6, 0):
        assert p == pytest.approx(2 / 64)


def test_holm():
    rejected = holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.005}, alpha=0.05)
    # thresholds 0.05/4, 0.05/3, 0.05/2, 0.05/1 for the sorted p-values 0.005, 0.01, 0.03, 0.04
    assert rejected == {"d": True, "a": True, "c": False, "b": False}
    assert holm({"x": 0.2}) == {"x": False}


@pytest.mark.parametrize("n,k,p", [(5, 2, 0.3), (10, 3, 0.8), (4, 4, 0.5)])
def test_pass_estimators_are_unbiased(n, k, p):
    weights = [math.comb(n, c) * p ** c * (1 - p) ** (n - c) for c in range(n + 1)]
    assert sum(w * pass_hat_k(n, c, k) for c, w in enumerate(weights)) == pytest.approx(p ** k)
    assert sum(w * pass_at_k(n, c, k) for c, w in enumerate(weights)) == pytest.approx(1 - (1 - p) ** k)
    with pytest.raises(ValueError):
        pass_hat_k(n, 0, n + 1)


# ----------------------------------------------------------------------------- glue
def test_run_config_resumes_and_summarize(tmp_path):
    qs = [Question("q1", "simple", "a?", ["x"], [PAGE.ref]), Question("q2", "unanswerable", "b?", [], [])]
    calls = []

    def runner(question):
        calls.append(question)
        return Prediction("x" if question == "a?" else "", question != "a?", [PAGE.ref] if question == "a?" else [], [PAGE.ref])

    out = tmp_path / "agent.jsonl"
    preds = run_config("agent", runner, qs, out)
    again = run_config("agent", runner, qs, out)
    assert len(calls) == 2 and again.keys() == preds.keys() and again["q1"].citations == [PAGE.ref]
    scores = score_all(again, qs)
    rows = {r["category"]: r for r in summarize(scores)}
    assert rows["all"]["accuracy"] == 1.0 and rows["unanswerable"]["correct_abstention"] == 1.0
    assert rows["simple"]["cit_precision"] == 1.0
    res = compare({"agent": scores, "rag": score_all({"q1": Prediction("y", False, [], []),
                                                      "q2": Prediction("", True, [], [])}, qs)},
                  [("agent", "rag")])
    assert res[0]["n01"] == 1 and res[0]["n10"] == 0 and res[0]["p"] == 1.0
    assert Verdict(ok=False).supported == []
