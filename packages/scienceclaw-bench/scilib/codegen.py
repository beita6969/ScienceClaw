"""Python code generation from problem statements: prompt builders, reply sanitizer, repair prompts, candidate selection.

Problems are the dicts of tool ``load_eval_inputs`` (keys ``prompt``, ``entry_point``, ``visible_tests``, ``kind``);
``results`` are the output of tool ``run_visible_tests`` (the whole dict or its ``results`` list).

build_prompts(problems, style="direct", time_limit_s=6) -> list[str]   one generation prompt per problem; styles "direct", "plan", "examples"
sanitize(texts, problems, index=None, base=None) -> list[str]          model reply texts -> program sources y[i] (assembly rule of the task)
repair_prompts(problems, codes, results) -> {"prompts", "index"}       prompts for the problems whose visible tests fail (index = their positions)
select_best(candidate_sets, results, return_choice=False) -> list[str] per problem the candidate that passes the most visible tests
assembled_program(problem, y) -> str                                   program the evaluation executes for y
is_implemented(problem, y) -> (bool, str)                              the hard constraint `implemented` for one problem

Shapes around an llm node. The template ``{item}`` inserts a string item unchanged, so an llm node with template ``{item}``
whose input port ``items`` receives a list of strings (the list from build_prompts, or repair_prompts(...)["prompts"]) sends
one request per string. Its output port ``outputs`` is a list with one entry per item, in order: the reply text
(``config.parse = "text"``, whitespace-stripped) or None for an empty, failed or timed-out request; sanitize takes
that list as ``texts``. An llm node takes at most the number of items stated as the llm-item limit of the task budget.
The reply length is ``config.max_tokens`` (default 2000; larger values are clamped to the executor limit, 2000 in the
default configurations). Code nodes cannot call the language model and tools run as tool nodes, so generation, visible
tests and repair are separate nodes. Values move between nodes whole, but the agent sees an output port only as a short
summary (type, length, first ~160 characters) and a code node's stdout only as a tail of at most 800 characters: the
problem texts are not readable in the feedback. Identical prompts give identical replies (requests are cached, temperature 0);
different candidates need different prompts (the styles) or ``config.seed`` / ``config.temperature``.

Assembly rule (as in the task objective). Python-stub problems (``kind == "python_function_stub"``): the executed program
is ``prompt`` (+ a newline if missing) followed by y[i]; other problems: y[i] alone. The hidden run appends the hidden
tests to that program and gives every statement, including the program itself, a limit of 6 s. `implemented` holds when
the assembled program parses and its last top-level definition of ``entry_point`` has a statement beyond its docstring.
`run_visible_tests` assembles the same way and reads config ``test_timeout_s`` (0.5-30 s, default 5).

build_prompts: each prompt names the function, gives the task text, the public tests (``visible_tests``) and the time
limit ``time_limit_s``, and asks for one fenced python block with the complete function. "direct" is the plain form,
"plan" asks for comment lines (algorithm, edge cases, return type) before the code, "examples" puts the public tests
before the task. Only the fields of the problem dict are used.

sanitize: for every reply it picks the fenced python blocks (unterminated fences included; all blocks together and the
raw text are further candidates), parses them and keeps the imports that are used, the definition of ``entry_point`` and
what it reaches (functions, classes, constants, decorators); demonstration code (calls, prints, ``input()``,
``__main__`` guards, asserts) and unused imports are dropped and ``from __future__`` lines are removed. A function of the
reply with another name is renamed to ``entry_point`` when the reply has none with that name. For a stub problem a reply that
is only the function body is handled (the prompt's stub is completed). A prelude (typing star-import, math, re, collections,
itertools, functools, heapq, bisect, string, sys and a few collections/functools names) is prepended. The result is checked with
is_implemented on the assembled program; when no candidate passes the check (None reply, prose only, truncated
code) y[i] is a syntactically valid placeholder that fails the tests but satisfies `implemented`. With ``index`` (positions of
the problems the texts belong to, as returned by repair_prompts) and ``base`` (a full list of program sources), the result is
``base`` with those positions replaced.

repair_prompts: for each problem with ``results[i]["passed"]`` false it returns a prompt with the task, the current code, the
failing public tests (up to 3) and the error text of the run; a program that is the placeholder of sanitize gets a generation
prompt (style "plan") instead. ``prompts`` and ``index`` are aligned lists of equal length (empty when everything passes).

select_best: ``candidate_sets`` is a list of program lists (each as long as ``problems``) and ``results`` the matching list
of run_visible_tests outputs. For each problem the candidate with ``passed`` true and the largest ``n_passed`` is taken; on a
tie the earlier set wins. ``return_choice=True`` also returns the chosen set index per problem.
"""
from __future__ import annotations

import ast
import copy
import difflib
import re
import textwrap
import warnings

__all__ = ["STYLES", "build_prompts", "sanitize", "sanitize_one", "repair_prompts", "select_best", "assembled_program",
           "is_implemented"]

STYLES = ("direct", "plan", "examples")
STUB_KIND = "python_function_stub"
PRELUDE = ("from typing import *\n"
           "import math, re, collections, itertools, functools, heapq, bisect, string, sys\n"
           "from collections import Counter, defaultdict, deque, OrderedDict\n"
           "from functools import lru_cache, reduce\n")
_UNSAFE_CALLS = {"input", "print", "exit", "quit", "open", "breakpoint", "main"}
_MAX_TEXT = 200_000
_MAX_TEST_CHARS = 3000
_MAX_ERROR_CHARS = 700
_PYTHON_TAGS = {"", "python", "python3", "py", "py3", "python-repl", "cpython"}


# ------------------------------------------------------------------------------------------------ problems
def _check_problems(problems) -> list[dict]:
    if not isinstance(problems, (list, tuple)):
        raise TypeError("problems must be the list of problem dicts of load_eval_inputs, got " + type(problems).__name__)
    for i, p in enumerate(problems):
        if not isinstance(p, dict):
            raise TypeError(f"problems[{i}] must be a dict with keys prompt, entry_point, visible_tests, kind, got {type(p).__name__}")
        for k in ("prompt", "entry_point"):
            if not isinstance(p.get(k), str):
                raise ValueError(f"problems[{i}] has no string field {k!r} (keys: {sorted(p)})")
    return list(problems)


def _is_stub(problem: dict) -> bool:
    return str(problem.get("kind", "")) == STUB_KIND


def _tests_text(problem: dict) -> str:
    t = problem.get("visible_tests") or ""
    if isinstance(t, (list, tuple)):
        t = "\n".join(str(x) for x in t)
    t = str(t).strip()
    return t if len(t) <= _MAX_TEST_CHARS else t[:_MAX_TEST_CHARS] + "\n# ... (more tests omitted)"


def _parse(src: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return ast.parse(src)
        except (SyntaxError, ValueError, MemoryError, RecursionError):
            return None


def assembled_program(problem: dict, y: str) -> str:
    """The program executed for problem ``problem`` and source ``y`` (before the test suffix is appended)."""
    if _is_stub(problem):
        p = problem["prompt"]
        return p + ("" if p.endswith("\n") else "\n") + y
    return y


def is_implemented(problem: dict, y: str) -> tuple[bool, str]:
    """`implemented` for one problem: the assembled program parses and its last top-level def of the entry point has a real body."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(assembled_program(problem, y))
    except (SyntaxError, ValueError, MemoryError, RecursionError) as ex:
        return False, f"syntax error: {getattr(ex, 'msg', ex)}"
    name = problem["entry_point"]
    defs = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name]
    if not defs:
        return False, f"no top-level function named {name!r}"
    body = defs[-1].body
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]
    if not body:
        return False, f"function {name!r} has no body beyond its docstring"
    return True, ""


def _placeholder(problem: dict) -> str:
    name = problem["entry_point"]
    return "    pass\n" if _is_stub(problem) else f"def {name}(*args, **kwargs):\n    pass\n"


# ------------------------------------------------------------------------------------------------ prompts
def _task_block(problem: dict) -> str:
    p = problem["prompt"].rstrip()
    if _is_stub(problem):
        return "Complete this function (signature and docstring are given):\n\n" + p
    return "Task:\n" + p


def _limit_sentence(time_limit_s) -> str:
    if time_limit_s is None:
        return ""
    return (f"Each test call is run with a time limit of {float(time_limit_s):g} s, so the solution must be efficient. "
            "The function must return its result (not print it).")


def build_prompts(problems, style: str = "direct", time_limit_s: float | None = 6) -> list[str]:
    """One generation prompt (a string) per problem. ``style``: "direct" | "plan" | "examples" (see the module docstring)."""
    if style not in STYLES:
        raise ValueError(f"unknown style {style!r}; available styles: {list(STYLES)}")
    out = []
    for p in _check_problems(problems):
        name = p["entry_point"]
        tests = _tests_text(p)
        limit = _limit_sentence(time_limit_s)
        tests_block = f"Public tests (must pass):\n{tests}" if tests else "(No public tests are given for this problem.)"
        ending = (f"Reply with ONE ```python block containing the complete function `{name}` (all imports and helpers "
                  "included, no prints, no test code, no explanation outside the block).")
        if style == "direct":
            text = f"Write a Python 3 function `{name}`.\n\n{_task_block(p)}\n\n{tests_block}\n\n{limit}\n\n{ending}"
        elif style == "plan":
            text = (f"Write a Python 3 function `{name}`.\n\n{_task_block(p)}\n\n{tests_block}\n\n{limit}\n\n"
                    "Inside the block, start with 2-4 comment lines that state the algorithm, the edge cases (empty input, "
                    "zero, negative numbers, duplicates, ties) and the exact type and format of the return value shown "
                    f"by the public tests; then write the complete function `{name}` (all imports and helpers included, no "
                    "prints, no test code). Reply with ONE ```python block and nothing outside it.")
        else:  # examples
            head = (f"These assert statements show the required behaviour of the Python 3 function `{name}`:\n{tests}\n\n"
                    if tests else "(No public tests are given for this problem.)\n\n")
            text = (f"{head}{_task_block(p)}\n\nWork out from the examples the type and format of every argument and of the "
                    f"return value, and keep the name `{name}` and the argument order of the examples. {limit}\n\n{ending}")
        out.append(text)
    return out


def _clip_tail(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else "..." + s[-n:]


def _feedback(res: dict) -> str:
    lines = []
    for t in (res.get("failing_tests") or [])[:3]:
        lines.append(str(t).strip())
    err = _clip_tail(str(res.get("error") or ""), _MAX_ERROR_CHARS)
    if lines:
        text = "These public tests fail:\n" + "\n".join(lines)
    else:
        text = "The program could not be run against the public tests."
    if err:
        text += "\nError of the run: " + err
    n_t, n_p = res.get("n_tests"), res.get("n_passed")
    if isinstance(n_t, int) and n_t > 0 and isinstance(n_p, int):
        text += f"\n({n_p} of {n_t} public tests pass.)"
    return text


def _unwrap_results(results, n: int, what: str = "results") -> list[dict]:
    if isinstance(results, dict) and "results" in results:
        results = results["results"]
    if not isinstance(results, (list, tuple)) or not all(isinstance(r, dict) for r in results):
        raise TypeError(f"{what} must be the output of run_visible_tests (a dict with key 'results', or the list of per-problem dicts)")
    if len(results) != n:
        raise ValueError(f"{what} has {len(results)} entries, expected {n}")
    return list(results)


def _display_code(problem: dict, y: str) -> str:
    """Code shown in a repair prompt: y without the prelude; a body-only y is shown inside its stub."""
    code = y[len(PRELUDE):] if y.startswith(PRELUDE) else y
    if _is_stub(problem) and f"def {problem['entry_point']}" not in code:
        code = assembled_program(problem, y)
    return code.strip("\n")


def repair_prompts(problems, codes, results) -> dict:
    """Prompts for the problems whose visible run did not pass: ``{"prompts": [str, ...], "index": [int, ...]}`` (aligned)."""
    probs = _check_problems(problems)
    if not isinstance(codes, (list, tuple)) or len(codes) != len(probs):
        raise ValueError(f"codes must be a list of {len(probs)} program sources (got {type(codes).__name__}"
                         + (f" of length {len(codes)}" if isinstance(codes, (list, tuple)) else "") + ")")
    res = _unwrap_results(results, len(probs))
    prompts, index = [], []
    for i, (p, y, r) in enumerate(zip(probs, codes, res)):
        if r.get("passed"):
            continue
        y = y if isinstance(y, str) else ""
        index.append(i)
        if y.strip() == _placeholder(p).strip() or not y.strip():
            prompts.append(build_prompts([p], "plan")[0])
            continue
        name = p["entry_point"]
        prompts.append(
            f"The Python function `{name}` below fails a public test.\n\n{_task_block(p)}\n\n"
            f"Current code:\n```python\n{_display_code(p, y)}\n```\n\n{_feedback(r)}\n\n"
            f"Reply with ONE ```python block that contains the complete corrected function `{name}` (all imports and helpers "
            "included, no prints, no test code). Start the block with two comment lines that name the cause of the "
            "failure.")
    return {"prompts": prompts, "index": index}


# ------------------------------------------------------------------------------------------------ reply parsing
def _blocks(text: str) -> list[tuple[str, str, bool]]:
    """Fenced blocks of ``text`` as (language tag, code, complete); an unterminated last fence is kept (complete False)."""
    blocks, cur, lang = [], None, ""
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("```"):
            if cur is None:
                cur, lang = [], s[3:].strip().lower()
                rest = lang.split()
                lang = rest[0] if rest else ""
            else:
                blocks.append((lang, "\n".join(cur), True))
                cur = None
        elif cur is not None:
            cur.append(line)
    if cur is not None:
        blocks.append((lang, "\n".join(cur), False))
    return blocks


_CODE_START = re.compile(r"(def |async def |class |import |from \S+ import |@)")


def _trim(text: str) -> str:
    """Start at the first code-looking line and drop trailing lines until the rest parses (longest valid prefix)."""
    lines = text.splitlines()
    start = next((k for k, l in enumerate(lines) if _CODE_START.match(l)), None)
    if start is None:
        return ""
    lines = lines[start:start + 600]
    while lines and _parse("\n".join(lines)) is None:
        lines.pop()
    return "\n".join(lines)


def _candidates(text: str) -> list[str]:
    """Code candidates in order of preference: python blocks, all blocks together, other blocks, raw text, trimmed text."""
    blocks = _blocks(text)
    py = [c for lang, c, _ok in blocks if lang in _PYTHON_TAGS and c.strip()]
    other = [c for lang, c, _ok in blocks if lang not in _PYTHON_TAGS and c.strip()]
    cands = list(py)
    if len(py) > 1:
        cands.append("\n\n".join(py))
    cands += other
    if not blocks:
        cands.append(text)
    cands += [_trim(c) for c in py + other if _parse(textwrap.dedent(c)) is None]
    cands.append(_trim(text))
    seen, out = set(), []
    for c in cands:
        if c.strip() and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _names(node) -> set[str]:
    return ({n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
            | {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)})


def _targets(node) -> set[str]:
    ts = node.targets if isinstance(node, ast.Assign) else [node.target]
    return {n.id for t in ts for n in ast.walk(t) if isinstance(n, ast.Name)}


def _unsafe_assign(node) -> bool:
    return any(isinstance(c, ast.Call) and getattr(c.func, "id", "") in _UNSAFE_CALLS for c in ast.walk(node))


def _bound(imp) -> set[str]:
    if isinstance(imp, ast.Import):
        return {(a.asname or a.name.split(".")[0]) for a in imp.names}
    return {(a.asname or a.name) for a in imp.names}


def _import_only_try(node) -> bool:
    return isinstance(node, ast.Try) and all(isinstance(s, (ast.Import, ast.ImportFrom, ast.Pass)) for s in node.body) \
        and all(isinstance(s, (ast.Import, ast.ImportFrom, ast.Pass)) for h in node.handlers for s in h.body) \
        and not node.orelse and not node.finalbody


def _extract(tree: ast.Module, entry: str) -> str | None:
    """Source of the imports used, the definition of ``entry`` and everything it reaches; None without such a definition."""
    imports, symbols, pres = [], {}, []
    for n in tree.body:
        if isinstance(n, ast.ImportFrom) and n.module == "__future__":
            continue
        if isinstance(n, (ast.Import, ast.ImportFrom)) or _import_only_try(n):
            imports.append(n)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols[n.name] = n
        elif isinstance(n, (ast.Assign, ast.AnnAssign)):
            if not _unsafe_assign(n) and (isinstance(n, ast.Assign) or n.value is not None):
                for name in _targets(n):
                    symbols[name] = n
        elif isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) \
                and ast.unparse(n.value.func) == "sys.setrecursionlimit":
            pres.append(n)
    root = symbols.get(entry)
    if not isinstance(root, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    need, todo, nodes = set(), [entry], {}
    while todo:
        name = todo.pop()
        node = symbols.get(name)
        if name in need or node is None:
            continue
        need.add(name)
        nodes[id(node)] = node
        todo.extend(_names(node))
    kept = sorted(nodes.values(), key=lambda n: n.lineno) + pres
    used = set().union(*[_names(n) for n in kept]) if kept else set()
    body = []
    for imp in imports:
        if _import_only_try(imp):
            names = set().union(*[_bound(s) for s in imp.body if isinstance(s, (ast.Import, ast.ImportFrom))]) if imp.body else set()
            if names & used:
                body.append(imp)
        elif any(a.name == "*" for a in imp.names) or _bound(imp) & used:
            body.append(imp)
    try:
        return ast.unparse(ast.Module(body=body + kept, type_ignores=[]))
    except (ValueError, RecursionError, AttributeError):
        return None


def _rename_target(tree: ast.Module, entry: str) -> ast.Module | None:
    """Tree in which the most plausible top-level function is renamed to ``entry`` (None if there is no function)."""
    fns = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if not fns:
        return None
    called = set()
    for f in fns:
        called |= {x for x in _names(f) if x != f.name}
    roots = [f for f in fns if f.name not in called] or fns
    best = max(enumerate(roots), key=lambda kv: (difflib.SequenceMatcher(None, kv[1].name, entry).ratio(), kv[0]))[1]
    old = best.name

    class _Rename(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id == old:
                node.id = entry
            return node

        def visit_FunctionDef(self, node):
            if node.name == old:
                node.name = entry
            self.generic_visit(node)
            return node

        visit_AsyncFunctionDef = visit_FunctionDef

    return _Rename().visit(tree)


def _returns_outside_function(tree: ast.Module) -> bool:
    """True when a ``return`` / ``yield`` sits at module level (the text is a function body, not a module)."""
    todo = list(tree.body)
    while todo:
        n = todo.pop()
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        if isinstance(n, (ast.Return, ast.Yield, ast.YieldFrom)):
            return True
        todo.extend(ast.iter_child_nodes(n))
    return False


def _finish(code: str | None, problem: dict) -> str | None:
    if not code:
        return None
    y = PRELUDE + code.rstrip("\n") + "\n"
    ok, _why = is_implemented(problem, y)
    return y if ok else None


def sanitize_one(text, problem: dict) -> str:
    """Program source y for one model reply ``text`` (None allowed) and its problem dict; never raises for odd replies."""
    entry, stub = problem["entry_point"], _is_stub(problem)
    text = "" if text is None else str(text)[:_MAX_TEXT]
    cands = _candidates(text)
    trees = [(c, t) for c in cands for t in [_parse(textwrap.dedent(c))] if t is not None]
    for _c, t in trees:                                     # 1. a definition with the required name
        y = _finish(_extract(t, entry), problem)
        if y:
            return y
    for _c, t in trees:                                     # 2. a function with another name -> renamed
        if stub and _returns_outside_function(t):
            continue                                        # a function body, handled in step 3
        t2 = _rename_target(copy.deepcopy(t), entry)
        y = _finish(_extract(t2, entry), problem) if t2 is not None else None
        if y:
            return y
    if stub:                                                # 3. only the body of the stub
        for c in cands:
            t = _parse(textwrap.dedent(c))
            if t is None or not _returns_outside_function(t):
                continue                                    # not a function body
            body = textwrap.indent(textwrap.dedent(c).strip("\n"), "    ")
            whole = _parse(assembled_program(problem, body + "\n"))
            if whole is None:
                continue
            y = _finish(_extract(whole, entry), problem)
            if y:
                return y
            if is_implemented(problem, body + "\n")[0]:
                return body + "\n"
    return _placeholder(problem)


def sanitize(texts, problems, index=None, base=None) -> list[str]:
    """Program sources for model replies. See the module docstring for ``index`` / ``base``."""
    probs = _check_problems(problems)
    if not isinstance(texts, (list, tuple)):
        raise TypeError("texts must be the list 'outputs' of an llm node (str or None entries), got " + type(texts).__name__)
    if index is None:
        if len(texts) != len(probs):
            raise ValueError(f"got {len(texts)} reply texts for {len(probs)} problems; pass index=<positions> for a subset")
        return [sanitize_one(t, p) for t, p in zip(texts, probs)]
    index = [int(i) for i in index]
    if len(index) != len(texts):
        raise ValueError(f"index has {len(index)} entries but there are {len(texts)} reply texts")
    if any(not 0 <= i < len(probs) for i in index):
        raise ValueError("index contains positions outside the problem list")
    if base is None:
        out = [_placeholder(p) for p in probs]
    else:
        if not isinstance(base, (list, tuple)) or len(base) != len(probs):
            raise ValueError(f"base must be a list of {len(probs)} program sources")
        out = [b if isinstance(b, str) else _placeholder(p) for b, p in zip(base, probs)]
    for t, i in zip(texts, index):
        out[i] = sanitize_one(t, probs[i])
    return out


# ------------------------------------------------------------------------------------------------ selection
def select_best(candidate_sets, results, return_choice: bool = False):
    """Per problem the candidate with ``passed`` and the most ``n_passed`` public tests; the earlier set wins ties."""
    if not isinstance(candidate_sets, (list, tuple)) or not candidate_sets:
        raise ValueError("candidate_sets must be a non-empty list of program lists")
    n = len(candidate_sets[0])
    for j, c in enumerate(candidate_sets):
        if not isinstance(c, (list, tuple)) or len(c) != n:
            raise ValueError(f"candidate_sets[{j}] must be a list of {n} program sources")
    if not isinstance(results, (list, tuple)) or len(results) != len(candidate_sets):
        raise ValueError(f"results must hold one run_visible_tests output per candidate set ({len(candidate_sets)})")
    res = [_unwrap_results(r, n, f"results[{j}]") for j, r in enumerate(results)]
    out, choice = [], []
    for i in range(n):
        best = max(range(len(candidate_sets)),
                   key=lambda j: (bool(res[j][i].get("passed")), int(res[j][i].get("n_passed") or 0), -j))
        out.append(candidate_sets[best][i])
        choice.append(best)
    return (out, choice) if return_choice else out
