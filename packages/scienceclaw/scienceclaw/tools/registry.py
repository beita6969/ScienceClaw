"""Catalog of the scientific tool library (``scilib``) with availability probing and retrieval.

Every ``scilib`` module is parsed statically (nothing heavy is imported to build the catalog) into :class:`ToolEntry` records:
its public functions with signatures and documentation, the third-party packages it needs, the weight assets it reads
(from :mod:`scienceclaw.tools.weights`), whether it can fall back to a remote GPU worker and which benchmark disciplines use it.
:func:`probe` then asks a module whether it can actually run here (``available()``), and :func:`search` retrieves tools for a
natural-language need with the same BM25 index the Skill/Operator retriever uses.
"""
from __future__ import annotations

import ast
import importlib
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from scienceclaw.core.retrieval import BM25Index, tokenize
from scienceclaw.tools import weights as W

SCILIB_DIR = Path(__file__).resolve().parents[2] / "scilib"
BASE_STACK = frozenset({"numpy", "scipy", "pandas", "sklearn", "yaml"})
_SKIP_MODULES = frozenset({"_pretrained", "_remote", "_weyler"})


@dataclass(frozen=True)
class ToolEntry:
    id: str                       # "tsfm.forecast"
    module: str                   # "tsfm"
    name: str                     # "forecast"
    kind: str                     # "pretrained" (reads model weights) | "library"
    summary: str
    signature: str
    doc: str
    requires: tuple[str, ...]     # packages beyond numpy / scipy / pandas / scikit-learn
    weights: tuple[str, ...]      # ids in weights.json
    remote: bool                  # can run on a remote GPU worker when it cannot run locally
    tasks: tuple[str, ...]        # benchmark disciplines whose adapters use the module
    tags: tuple[str, ...]

    @property
    def import_line(self) -> str:
        return f"from scilib import {self.module}"

    def card(self, full: bool = False) -> str:
        head = f"{self.id}{self.signature[len(self.name):] if self.signature.startswith(self.name) else ''}"
        lines = [f"{head}  [{self.kind}]", f"  {self.summary}", f"  use: {self.import_line}; {self.module}.{self.name}(...)"]
        if self.requires:
            lines.append(f"  needs: {', '.join(self.requires)}")
        if self.weights:
            lines.append(f"  weights: {', '.join(self.weights)}")
        if self.remote:
            lines.append("  runs on a remote GPU worker when local resources are missing")
        if self.tasks:
            lines.append(f"  used by: {', '.join(self.tasks)}")
        if full:
            lines += ["", self.doc.strip()]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "module": self.module, "name": self.name, "kind": self.kind, "summary": self.summary,
                "signature": self.signature, "requires": list(self.requires), "weights": list(self.weights),
                "remote": self.remote, "tasks": list(self.tasks), "tags": list(self.tags)}


# ---------------------------------------------------------------------------------------------------- static analysis
def _first_sentence(text: str, limit: int = 220) -> str:
    text = " ".join((text or "").split())
    m = re.search(r"(?<=[.!?])\s", text)
    s = text[: m.start()] if m else text
    return s if len(s) <= limit else s[: limit - 3] + "..."


def _signature(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    ret = f" -> {ast.unparse(fn.returns)}" if fn.returns is not None else ""
    return f"{fn.name}({ast.unparse(fn.args)}){ret}"


def _third_party_imports(tree: ast.AST) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module]
        for n in names:
            top = n.split(".")[0]
            if top not in sys.stdlib_module_names and top not in {"scilib", "scienceclaw", "__future__"}:
                found.add(top)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "have_module" and node.args:
            a0 = node.args[0]
            if isinstance(a0, ast.Constant) and isinstance(a0.value, str):
                found.add(a0.value.split(".")[0])
    return found - BASE_STACK


def _public_names(tree: ast.Module) -> list[str] | None:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "__all__" for t in node.targets):
            try:
                return [str(x) for x in ast.literal_eval(node.value)]
            except (ValueError, SyntaxError):
                return None
    return None


@lru_cache(maxsize=1)
def _task_usage() -> dict[str, tuple[str, ...]]:
    """module -> benchmark disciplines whose adapter references it (static scan of ``bench/tasks``)."""
    tasks_dir = Path(__file__).resolve().parents[1] / "bench" / "tasks"
    usage: dict[str, set[str]] = {}
    for f in sorted(tasks_dir.glob("for*.py")):
        m = re.match(r"for(\d+)_", f.name)
        if not m:
            continue
        code = f"FoR{m.group(1)}"
        for mod in set(re.findall(r"\bscilib(?:\.|\s+import\s+)([a-z_0-9]+)", f.read_text(encoding="utf-8"))):
            usage.setdefault(mod, set()).add(code)
    return {k: tuple(sorted(v)) for k, v in usage.items()}


@lru_cache(maxsize=1)
def _weights_by_module() -> dict[str, tuple[str, ...]]:
    out: dict[str, list[str]] = {}
    for a in W.load():
        for mod in a.used_by:
            out.setdefault(mod, []).append(a.id)
    return {k: tuple(v) for k, v in out.items()}


@lru_cache(maxsize=1)
def catalog() -> tuple[ToolEntry, ...]:
    entries: list[ToolEntry] = []
    usage, wmap = _task_usage(), _weights_by_module()
    for path in sorted(SCILIB_DIR.glob("*.py")):
        mod = path.stem
        if mod.startswith("__") or mod in _SKIP_MODULES:
            continue
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src)
        mdoc = ast.get_docstring(tree) or ""
        names = _public_names(tree)
        funcs = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
        public = [n for n in (names if names is not None else [*funcs, *classes])
                  if not n.startswith("_") and n != "available"]
        msummary = _first_sentence(mdoc)
        requires = tuple(sorted(_third_party_imports(tree) - STAGED_SOURCES))
        weights = wmap.get(mod, ())
        kind = "pretrained" if weights or "model_path(" in src or "MODEL_ENV" in src else "library"
        remote = "_remote" in src
        tasks = usage.get(mod, ())
        domain_tags = tuple(sorted({t for t in re.split(r"[_\W]+", mod) if t} | ({"pretrained", "weights"} if kind == "pretrained" else set())))
        for name in public:
            node = funcs.get(name) or classes.get(name)
            if node is None or not isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                continue
            fdoc = ast.get_docstring(node) or ""
            if isinstance(node, ast.ClassDef):
                init = next((n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None)
                sig = f"{name}({ast.unparse(init.args) if init else ''})".replace("(self, ", "(").replace("(self)", "()")
            else:
                sig = _signature(node)
            doc = fdoc or mdoc
            about = _line_about(mdoc, name)
            summary = _first_sentence(fdoc) if fdoc else (_first_sentence(about) if about and not about.startswith("`") else msummary)
            entries.append(ToolEntry(id=f"{mod}.{name}", module=mod, name=name, kind=kind, summary=summary, signature=sig,
                                     doc=doc, requires=requires, weights=weights, remote=remote, tasks=tasks, tags=domain_tags))
    return tuple(entries)


def _line_about(module_doc: str, name: str) -> str:
    """The module docstring documents entry points as ``name(args) -> result`` followed by an indented description."""
    lines = module_doc.splitlines()
    for i, ln in enumerate(lines):
        if re.match(rf"\s*{re.escape(name)}\s*\(", ln) and i + 1 < len(lines):
            return " ".join(x.strip() for x in lines[i + 1:i + 3])
    return ""


# ---------------------------------------------------------------------------------------------------- queries
# packages that are not installed with pip: their sources are staged next to the weights (see the post steps of weights.json)
STAGED_SOURCES = frozenset({"buildings_bench"})


def modules() -> list[str]:
    return sorted({e.module for e in catalog()})


def get(tool_id: str) -> ToolEntry:
    for e in catalog():
        if e.id == tool_id:
            return e
    raise KeyError(f"unknown tool {tool_id!r}")


def module_doc(module: str) -> str:
    """The interface description of a whole module (the text the policy sees for ``scilib.<module>``)."""
    path = SCILIB_DIR / f"{module}.py"
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", module) or not path.is_file() or module in _SKIP_MODULES:
        raise KeyError(f"unknown scilib module {module!r}")
    return ast.get_docstring(ast.parse(path.read_text(encoding="utf-8"))) or ""


@lru_cache(maxsize=1)
def _index() -> BM25Index[ToolEntry]:
    es = list(catalog())
    texts = [" ".join([e.id, e.name, e.module, e.summary, e.doc[:2500], " ".join(e.tags), " ".join(e.tasks), " ".join(e.requires)]) for e in es]
    return BM25Index(es, texts)


def search(query: str, k: int = 8, *, kind: str | None = None, task: str | None = None,
           available_only: bool = False) -> list[ToolEntry]:
    """Tools ranked for a natural-language need. ``task`` limits to a discipline code ("FoR37"); ``available_only`` probes."""
    idx = _index()
    scores = idx.scores(tokenize(query))
    ranked = sorted(zip(scores, idx.items), key=lambda p: (-p[0], p[1].id))
    out: list[ToolEntry] = []
    for s, e in ranked:
        if s <= 0:
            break
        if kind and e.kind != kind or task and task not in e.tasks:
            continue
        if available_only and not probe(e.module)["available"]:
            continue
        out.append(e)
        if len(out) >= k:
            break
    return out


# ---------------------------------------------------------------------------------------------------- availability
_PROBES: dict[str, dict[str, Any]] = {}


def probe(module: str, *, refresh: bool = False) -> dict[str, Any]:
    """Can ``scilib.<module>`` run here? Imports the module (and may import heavy packages), calls its ``available()`` if it has one."""
    if module in _PROBES and not refresh:
        return _PROBES[module]
    entries = [e for e in catalog() if e.module == module]
    if not entries:
        raise KeyError(f"unknown scilib module {module!r}")
    e0 = entries[0]
    missing = [p for p in e0.requires if not W._have(p)]
    result: dict[str, Any] = {"module": module, "kind": e0.kind, "available": False, "missing_packages": missing,
                              "weights": [W.status(w) for w in e0.weights], "remote": e0.remote, "reason": ""}
    try:
        mod = importlib.import_module(f"scilib.{module}")
        fn = getattr(mod, "available", None)
        if callable(fn):
            result["available"] = bool(fn())
            if not result["available"]:
                result["reason"] = _why_not(result)
        else:
            result["available"] = not missing
            if missing:
                result["reason"] = f"missing packages: {', '.join(missing)}"
    except Exception as ex:                                     # an import failure is an answer, not a crash
        result["reason"] = f"import failed: {type(ex).__name__}: {str(ex)[:160]}"
    _PROBES[module] = result
    return result


def _why_not(res: dict[str, Any]) -> str:
    bits = []
    if res["missing_packages"]:
        bits.append(f"missing packages: {', '.join(res['missing_packages'])}")
    absent = [w["id"] for w in res["weights"] if not w["present"]]
    if absent:
        bits.append(f"weights not staged: {', '.join(absent)}")
    if res["remote"]:
        bits.append("no remote GPU worker configured (SCIENCECLAW_REMOTE_SPOOL)")
    return "; ".join(bits) or "available() returned False"


def status_table(modules_: list[str] | None = None) -> list[dict[str, Any]]:
    rows = []
    for m in modules_ or modules():
        p = probe(m)
        rows.append({"module": m, "kind": p["kind"], "available": p["available"], "reason": p["reason"]})
    return rows
