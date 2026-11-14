"""Mandatory verification of sourced answers: deterministic checks, an LLM judge, quote grounding."""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field

from .bm25 import tokenize
from .schema import Page, PageRef, Prediction

JUDGE_SCHEMA = {"type": "object",
                "properties": {"supported": {"type": "boolean"}, "quote": {"type": "string"}},
                "required": ["supported", "quote"]}

JUDGE_SYSTEM = (
    "You check whether ONE page of a scientific paper supports an answer to a question.\n"
    "Reply with a JSON object {\"supported\": ..., \"quote\": ...}.\n"
    "- supported: true only if the page states the answer, or if the answer follows from numbers "
    "of the page by a direct computation.\n"
    "- quote: a passage of the page of 5 to 50 words (a sentence, a formula or a table row) that "
    "supports the answer, copied verbatim; \"\" if unsupported."
)

JOINT_SCHEMA = {"type": "object",
                "properties": {"supported": {"type": "boolean"},
                               "quotes": {"type": "array", "items": {"type": "string"}}},
                "required": ["supported", "quotes"]}

JOINT_SYSTEM = (
    "You check whether SEVERAL pages of scientific papers, taken together, support an answer to a "
    "question: for instance a ratio of two numbers found on different pages, or a comparison of two papers.\n"
    "Reply with a JSON object {\"supported\": ..., \"quotes\": [...]}.\n"
    "- supported: true only if the pages together state the answer, or if the answer follows from their "
    "numbers by a direct computation.\n"
    "- quotes: one string per page, in the order of the pages: a passage of that page of 5 to 50 words "
    "(a sentence, a formula or a table row) that the answer uses, copied verbatim; \"\" for a page that "
    "the answer does not use."
)
NOT_SUPPORTED = "judge: not supported"
MAX_JOINT_PAGES = 3          # pages judged together, at most
JOINT_CHARS = 30000          # characters of page text in the joint call (~9k tokens, num_ctx = 16384)


def quote_is_grounded(quote: str, page_text: str, min_overlap: float = 0.8, min_tokens: int = 4) -> bool:
    """Does `quote` really appear on the page, up to case, punctuation and Markdown noise?

    Tokens = bm25.tokenize. A quote with fewer than min_tokens tokens is never evidence.
    Score = best multiset overlap |window & quote| / |quote| over the windows of |quote| tokens.
    """
    q, t = tokenize(quote), tokenize(page_text)
    n = len(q)
    if n < min_tokens or len(t) < n:
        return False
    if f" {' '.join(q)} " in f" {' '.join(t)} ":     # fast path; the spaces stop "ab c" matching in "xab c"
        return True
    need, window = Counter(q), Counter(t[:n])
    common = sum(min(c, window[w]) for w, c in need.items())     # |window & quote| (multisets)
    best = common
    for i in range(n, len(t)):                       # slide by one token, update common in O(1)
        if best == n:                                # full match: no window can do better
            break
        new, old = t[i], t[i - n]
        if window[new] < need[new]:                  # the incoming token fills a missing occurrence
            common += 1
        window[new] += 1
        window[old] -= 1
        if window[old] < need[old]:                  # the outgoing token was a needed occurrence
            common -= 1
        best = max(best, common)
    return best / n >= min_overlap


def quote_problem(quote: str, page_text: str) -> str | None:
    """Why a judge quote is not evidence from this page; None if it is (never trust the judge)."""
    if not quote_is_grounded(quote, page_text):
        return "judge quote not found on the page"
    page_tokens = set(tokenize(page_text))
    if any(tok.isdigit() and tok not in page_tokens for tok in tokenize(quote)):
        return "a number of the judge quote is not on the page"
    return None


@dataclass
class Verdict:
    ok: bool
    supported: list[PageRef] = field(default_factory=list)
    rejected: list[tuple[PageRef, str]] = field(default_factory=list)   # (page, reason)
    n_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0


class Verifier:
    def __init__(self, llm, pages: list[Page], max_chars: int = 6000, require_consulted: bool = True,
                 joint: bool = False):
        self.llm = llm
        self.pages = {p.ref: p for p in pages}
        self.max_chars = max_chars
        self.require_consulted = require_consulted
        self.joint = joint             # judge together the pages that fail alone (see _check_together)

    def check(self, question: str, answer: str, citations: list[PageRef],
              consulted: list[PageRef]) -> Verdict:
        """Keep the cited pages that really support the answer; ok = at least one does.
        With joint=True, the pages that each fail alone are then judged together, once."""
        v = Verdict(ok=False)
        seen = set(consulted)
        for ref in dict.fromkeys(citations):                 # each distinct page once, in order
            # 1. free and certain checks first
            page = self.pages.get(ref)
            if page is None:
                v.rejected.append((ref, "page does not exist"))
                continue
            if self.require_consulted and ref not in seen:
                v.rejected.append((ref, "page was never consulted"))
                continue
            # 2. the judge: one constrained call per page, its cost goes into the verdict
            prompt = f"Question: {question}\nAnswer: {answer}\n\nPage {ref}:\n{page.text[: self.max_chars]}"
            reply = self.llm.chat([{"role": "system", "content": JUDGE_SYSTEM},
                                   {"role": "user", "content": prompt}], schema=JUDGE_SCHEMA, max_tokens=400)
            self._add_cost(v, reply)
            try:
                out = json.loads(reply.text)
            except json.JSONDecodeError:
                out = None
            if not isinstance(out, dict):
                v.rejected.append((ref, "judge reply is not JSON"))
                continue
            if out.get("supported") is not True:             # strict: the string "false" is truthy
                v.rejected.append((ref, NOT_SUPPORTED))
                continue
            # 3. never trust the judge's quote: find it on the page, numbers included
            problem = quote_problem(str(out.get("quote", "")), page.text)
            if problem:
                v.rejected.append((ref, problem))
                continue
            v.supported.append(ref)
        if self.joint and not v.supported:
            alone = [ref for ref, why in v.rejected if why == NOT_SUPPORTED]
            if len(alone) >= 2:
                self._check_together(question, answer, alone[:MAX_JOINT_PAGES], v)
        v.ok = bool(v.supported)
        return v

    def _check_together(self, question: str, answer: str, refs: list[PageRef], v: Verdict) -> None:
        """A ratio of numbers from two pages, or a comparison of two papers, fails on every page alone:
        one call judges the pages together, then each quote is checked on ITS page as above. At least
        two pages must give a real quote, otherwise one page would be enough and has already failed."""
        per_page = min(self.max_chars, JOINT_CHARS // len(refs))
        blocks = "\n\n".join(f"Page {ref}:\n{self.pages[ref].text[:per_page]}" for ref in refs)
        reply = self.llm.chat([{"role": "system", "content": JOINT_SYSTEM},
                               {"role": "user", "content": f"Question: {question}\nAnswer: {answer}\n\n{blocks}"}],
                              schema=JOINT_SCHEMA, max_tokens=600)
        self._add_cost(v, reply)
        try:
            out = json.loads(reply.text)
        except json.JSONDecodeError:
            out = None
        passed = []
        if isinstance(out, dict) and out.get("supported") is True and isinstance(out.get("quotes"), list):
            passed = [ref for ref, quote in zip(refs, out["quotes"])
                      if str(quote).strip() and quote_problem(str(quote), self.pages[ref].text) is None]
        if len(passed) >= 2:
            v.supported.extend(passed)
            v.rejected = [(ref, why) for ref, why in v.rejected if ref not in passed]
        else:
            v.rejected = [(ref, f"{why}, alone or together" if ref in refs else why) for ref, why in v.rejected]

    @staticmethod
    def _add_cost(v: Verdict, reply) -> None:
        v.n_calls += 1
        v.prompt_tokens += reply.prompt_tokens
        v.completion_tokens += reply.completion_tokens
        v.latency_s += reply.latency_s


class VerifiedAgent:
    """Agent + mandatory verification: at most max_repairs retries with feedback, then abstention."""

    def __init__(self, agent, verifier: Verifier, max_repairs: int = 1):
        self.agent, self.verifier, self.max_repairs = agent, verifier, max_repairs

    def run(self, question: str) -> Prediction:
        attempts, verdicts = [], []
        feedback = None
        for _ in range(1 + self.max_repairs):
            res = self.agent.run(question, feedback=feedback)
            attempts.append(res)
            if res.abstain:                                  # a legitimate answer: returned as is
                return self._merge(attempts, verdicts, "", True, [])
            if not res.citations:                            # nothing to check: no judge call
                verdicts.append(Verdict(ok=False))
                feedback = "your answer cites no page. Find the page that states the answer and cite it, or abstain."
                continue
            verdict = self.verifier.check(question, res.answer, res.citations, res.consulted)
            verdicts.append(verdict)
            if verdict.ok:                                   # keep only the pages that passed
                return self._merge(attempts, verdicts, res.answer, False, verdict.supported)
            feedback = ("the cited pages do not support your answer ("
                        + "; ".join(f"{ref}: {why}" for ref, why in verdict.rejected)
                        + "). Search again and cite a page that states the answer, or abstain.")
        return self._merge(attempts, verdicts, "", True, [])  # every attempt failed: abstain

    @staticmethod
    def _merge(attempts, verdicts, answer: str, abstain: bool, citations: list[PageRef]) -> Prediction:
        preds = [a.to_prediction() for a in attempts]
        return Prediction(
            answer=answer, abstain=abstain, citations=list(citations),
            consulted=list(dict.fromkeys(p for pr in preds for p in pr.consulted)),
            n_llm_calls=sum(p.n_llm_calls for p in preds) + sum(v.n_calls for v in verdicts),
            prompt_tokens=sum(p.prompt_tokens for p in preds) + sum(v.prompt_tokens for v in verdicts),
            completion_tokens=sum(p.completion_tokens for p in preds) + sum(v.completion_tokens for v in verdicts),
            latency_s=sum(p.latency_s for p in preds) + sum(v.latency_s for v in verdicts),
            trace=[t for p in preds for t in p.trace],
            meta={"attempts": len(attempts), "stop_reasons": [a.stop_reason for a in attempts],
                  "verdicts": [{"ok": v.ok, "supported": [str(r) for r in v.supported],
                                "rejected": [[str(r), why] for r, why in v.rejected]} for v in verdicts]},
        )
