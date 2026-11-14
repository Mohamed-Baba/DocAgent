"""The agent's tools: search, inspect_page, extract_table, calculator (all read-only)."""
from __future__ import annotations

import ast
import difflib
import math
import operator
from dataclasses import dataclass, field

from .schema import Page, PageRef


@dataclass
class ToolResult:
    text: str                                              # the observation shown to the model
    pages: list[PageRef] = field(default_factory=list)     # pages shown (= consulted)
    images: list[str] = field(default_factory=list)        # page images for the VLM
    error: bool = False


class Tool:
    name: str = ""
    description: str = ""
    parameters: dict = {}          # JSON schema of the arguments

    def __call__(self, **args) -> ToolResult:
        raise NotImplementedError


# ----------------------------------------------------------------------------- validation
_JSON_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool,
               "object": dict, "array": list}


def validate_args(args, schema: dict) -> list[str]:
    """Tiny JSON-schema validator: object, required, properties (type, enum, minimum),
    no additional property. Returns the list of error messages (empty = valid)."""
    if not isinstance(args, dict):
        return ["args must be a JSON object"]
    props = schema.get("properties", {})
    errors = []
    for name in schema.get("required", []):
        if name not in args:
            errors.append(f"missing required argument '{name}'")
    for name, value in args.items():
        if name not in props:
            errors.append(f"unexpected argument '{name}' (allowed: {sorted(props)})")
            continue
        spec = props[name]
        expected = spec["type"]
        # bool is a subclass of int: True must not pass for "integer" or "number"
        wrong_bool = isinstance(value, bool) and expected != "boolean"
        if wrong_bool or not isinstance(value, _JSON_TYPES[expected]):
            errors.append(f"'{name}' must be of type {expected}, got {type(value).__name__}")
            continue    # no enum/minimum check on a value of the wrong type ("3" < 1 raises)
        if "enum" in spec and value not in spec["enum"]:
            errors.append(f"'{name}' must be one of {spec['enum']}")
        if "minimum" in spec and value < spec["minimum"]:
            errors.append(f"'{name}' must be >= {spec['minimum']}")
    return errors


# ----------------------------------------------------------------------------- calculator
_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
           ast.Pow: operator.pow}
_UNOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {"sqrt": math.sqrt, "exp": math.exp, "log": math.log, "log10": math.log10,
          "log2": math.log2, "sin": math.sin, "cos": math.cos, "tan": math.tan,
          "abs": abs, "min": min, "max": max, "round": round}
_CONSTS = {"pi": math.pi, "e": math.e}
MAX_EXPONENT = 1000


def safe_eval(expression: str):
    """Evaluate an arithmetic expression WITHOUT eval(): walk a whitelisted AST."""
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant):
            if type(node.value) in (int, float):        # type(), not isinstance(): True is an int
                return node.value
            raise ValueError(f"unsupported constant {node.value!r}")
        if isinstance(node, ast.Name):
            if node.id in _CONSTS:
                return _CONSTS[node.id]
            raise ValueError(f"unknown name '{node.id}' (allowed: {', '.join(_CONSTS)})")
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNOPS:
            return _UNOPS[type(node.op)](ev(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
                raise ValueError(f"exponent {right} too large (max {MAX_EXPONENT})")
            return _BINOPS[type(node.op)](left, right)
        if isinstance(node, ast.Call):
            if not (isinstance(node.func, ast.Name) and node.func.id in _FUNCS):
                raise ValueError(f"unsupported function (allowed: {', '.join(_FUNCS)})")
            if node.keywords:
                raise ValueError("keyword arguments are not supported")
            return _FUNCS[node.func.id](*[ev(a) for a in node.args])
        raise ValueError(f"unsupported element: {type(node).__name__}")

    return ev(ast.parse(expression, mode="eval"))


class CalculatorTool(Tool):
    name = "calculator"
    description = ("Evaluate an arithmetic expression: numbers, + - * / // % **, sqrt, exp, log, "
                   "log10, log2, sin, cos, tan, abs, min, max, round, pi, e. Check every computation.")
    parameters = {"type": "object", "properties": {"expression": {"type": "string"}},
                  "required": ["expression"]}

    def __call__(self, expression: str) -> ToolResult:
        try:
            value = safe_eval(expression)
        except Exception as e:      # SyntaxError, ValueError, ZeroDivisionError, OverflowError...
            return ToolResult(text=f"Error: {type(e).__name__}: {e}", error=True)
        return ToolResult(text=f"{expression} = {value!r}")


# ----------------------------------------------------------------------------- corpus tools
class SearchTool(Tool):
    name = "search"
    description = ("Search the corpus; returns short excerpts of the best passages (cut after a fixed "
                   "length) with their doc_id and page. Optional doc_id restricts the search to one paper.")
    parameters = {"type": "object",
                  "properties": {"query": {"type": "string"}, "doc_id": {"type": "string"}},
                  "required": ["query"]}

    def __init__(self, retriever, known_docs: set[str], k: int = 5, mode: str = "hybrid+rerank",
                 max_chars: int = 700):
        self.retriever, self.known_docs = retriever, set(known_docs)
        self.k, self.mode, self.max_chars = k, mode, max_chars

    def __call__(self, query: str, doc_id: str | None = None) -> ToolResult:
        doc_id = (doc_id or "").strip() or None          # "" from the model means: no filter
        if doc_id is not None and doc_id not in self.known_docs:
            close = difflib.get_close_matches(doc_id, sorted(self.known_docs), n=3)
            hint = "" if close else " Search without doc_id to find the right one."
            return ToolResult(text=f"Error: unknown doc_id '{doc_id}'. Close matches: {close}.{hint}", error=True)
        hits = self.retriever.search(query, k=self.k, mode=self.mode,
                                     doc_ids={doc_id} if doc_id else None)
        if not hits:
            return ToolResult(text="No result.")
        blocks = [f"[{i}] {h.chunk.doc_id} p.{h.chunk.page}\n{h.chunk.text[: self.max_chars]}"
                  for i, h in enumerate(hits, start=1)]
        return ToolResult(text="\n\n".join(blocks), pages=list(dict.fromkeys(h.chunk.ref for h in hits)))


class _PageTool(Tool):
    parameters = {"type": "object",
                  "properties": {"doc_id": {"type": "string"}, "page": {"type": "integer", "minimum": 1}},
                  "required": ["doc_id", "page"]}

    def __init__(self, pages: list[Page], max_chars: int = 6000):
        self.pages = {p.ref: p for p in pages}
        self.last_page: dict[str, int] = {}
        for p in pages:
            self.last_page[p.doc_id] = max(p.page, self.last_page.get(p.doc_id, 0))
        self.max_chars = max_chars

    def _lookup(self, doc_id: str, page: int) -> tuple[Page | None, ToolResult | None]:
        if doc_id not in self.last_page:
            close = difflib.get_close_matches(doc_id, sorted(self.last_page), n=3)
            return None, ToolResult(text=f"Error: unknown doc_id '{doc_id}'. Close matches: {close}", error=True)
        p = self.pages.get(PageRef(doc_id, page))
        if p is None:
            return None, ToolResult(text=f"Error: {doc_id} has pages 1 to {self.last_page[doc_id]}.", error=True)
        return p, None


class InspectPageTool(_PageTool):
    name = "inspect_page"
    description = ("Read one full page and see its image (figures, equations, tables). "
                   "Use it to check details before answering.")

    def __init__(self, pages: list[Page], max_chars: int = 6000, with_image: bool = True):
        super().__init__(pages, max_chars)
        self.with_image = with_image

    def __call__(self, doc_id: str, page: int) -> ToolResult:
        p, err = self._lookup(doc_id, page)
        if err:
            return err
        images = [p.image_path] if self.with_image and p.image_path else []
        return ToolResult(text=f"{p.doc_id} p.{p.page}\n{p.text[: self.max_chars]}", pages=[p.ref], images=images)


def split_page(text: str, size: int) -> list[str]:
    """Cut a page into parts of at most `size` characters, at a line break or a space when possible
    (never inside a number such as 2,048). An empty page is one empty part."""
    parts = []
    while len(text) > size:
        start = max(1, size // 2)                           # look in the second half of the window
        cut = text.rfind("\n", start, size + 1)             # a line break first,
        if cut == -1:
            cut = text.rfind(" ", start, size + 1)          # else a space,
        if cut == -1:
            cut = size                                      # else anywhere (one huge token)
        parts.append(text[:cut])
        text = text[cut:].lstrip()
    return parts + [text]


class PagedInspectPageTool(InspectPageTool):
    """inspect_page that can reach the end of a long page.

    InspectPageTool cuts a page at max_chars and the agent cuts every observation at 5000 characters:
    the end of a long page (a Nature methods page has 7,000 to 9,000) was never shown. Here max_chars
    is the size of one part, below the observation limit, and part=2, 3... reads the rest: the context
    cost of one observation does not change.
    """
    description = ("Read one page and see its image (figures, equations, tables). A long page comes in "
                   "parts: the reply says 'part 1/2' and part=2 reads the rest.")
    parameters = {"type": "object",
                  "properties": {"doc_id": {"type": "string"}, "page": {"type": "integer", "minimum": 1},
                                 "part": {"type": "integer", "minimum": 1}},
                  "required": ["doc_id", "page"]}

    def __init__(self, pages: list[Page], max_chars: int = 4800, with_image: bool = True):
        super().__init__(pages, max_chars, with_image)

    def __call__(self, doc_id: str, page: int, part: int = 1) -> ToolResult:
        p, err = self._lookup(doc_id, page)
        if err:
            return err
        parts = split_page(p.text, self.max_chars)
        n = len(parts)
        if part > n:
            return ToolResult(text=f"Error: {doc_id} p.{page} has {n} part{'s' if n > 1 else ''} only.",
                              error=True)
        head = f"{p.doc_id} p.{p.page}" + (f" (part {part}/{n})" if n > 1 else "")
        tail = f"\n[end of part {part}/{n}: the page continues in part {part + 1}]" if part < n else ""
        images = [p.image_path] if self.with_image and p.image_path and part == 1 else []
        return ToolResult(text=f"{head}\n{parts[part - 1]}{tail}", pages=[p.ref], images=images)


class ExtractTableTool(_PageTool):
    name = "extract_table"
    description = "Return the tables of a page as Markdown (structure recovered by Docling)."

    def __call__(self, doc_id: str, page: int) -> ToolResult:
        p, err = self._lookup(doc_id, page)
        if err:
            return err
        if not p.tables:
            return ToolResult(text=f"No table detected on {doc_id} p.{page}.", pages=[p.ref])
        body = "\n\n".join(f"Table {i}:\n{t}" for i, t in enumerate(p.tables, start=1))
        return ToolResult(text=f"{doc_id} p.{page}\n{body}"[: self.max_chars], pages=[p.ref])


# ----------------------------------------------------------------------------- toolbox
class ToolBox:
    def __init__(self, tools: list[Tool]):
        self.tools = {t.name: t for t in tools}

    def describe(self) -> str:
        """One line per tool, for the system prompt."""
        lines = []
        for t in self.tools.values():
            props, req = t.parameters.get("properties", {}), set(t.parameters.get("required", []))
            sig = ", ".join(f"{k}: {v['type']}" + ("" if k in req else " (optional)") for k, v in props.items())
            lines.append(f"- {t.name}({sig}): {t.description}")
        return "\n".join(lines)

    def action_schema(self, final_only: bool = False) -> dict:
        """Schema of one agent step; key order = generation order (thought first)."""
        actions = ["final_answer"] if final_only else [*self.tools, "final_answer"]
        return {"type": "object",
                "properties": {"thought": {"type": "string"},
                               "action": {"type": "string", "enum": actions},
                               "args": {"type": "object"}},
                "required": ["thought", "action", "args"]}

    def call(self, name: str, args: dict) -> ToolResult:
        try:
            return self.tools[name](**args)
        except Exception as e:  # a tool must never crash the agent
            return ToolResult(text=f"Error while running {name}: {type(e).__name__}: {e}", error=True)


def default_toolbox(retriever, pages: list[Page], page_part_chars: int = 4800) -> ToolBox:
    """The agent's tools. A part of a page (page_part_chars) + its header and footer must stay below the
    agent's observation limit (Agent.max_obs_chars = 5000), or the end of the part is cut again."""
    docs = {p.doc_id for p in pages}
    return ToolBox([SearchTool(retriever, docs), PagedInspectPageTool(pages, page_part_chars),
                    ExtractTableTool(pages), CalculatorTool()])
