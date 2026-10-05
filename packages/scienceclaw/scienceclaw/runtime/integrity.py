"""Static leakage / sandbox scan for generated ``code`` nodes (DESIGN.md sections 3.4 and 8.3).

Code nodes may only compute on their ``inputs`` and write inside their working directory. The scan
(AST + string-literal regexes) rejects, before execution:

* absolute filesystem paths, ``..`` traversal, home directories (``~``, ``Path.home()``,
  ``expanduser``) and ``os.chdir`` -- i.e. reading or writing outside the working directory;
* references to the protected data root (the deployment's datasets) and to the ScienceClaw
  package itself (adapters / evaluators);
* network modules (socket, urllib, requests, http, httpx, ftplib, ...), process spawning
  (subprocess, os.system, os.popen, os.exec*/spawn*, pty, ctypes);
* ``eval`` / ``exec`` / ``compile`` / ``__import__`` and ``importlib.import_module`` of non-literal
  or forbidden modules, ``getattr`` tricks to reach forbidden functions, frame/closure introspection;
* environment and credential access (``os.environ``, ``os.getenv``, ``.config``, key files).

The scan is a guard against accidental leakage by a model-generated program, not a security
boundary against adversarial code; the worker additionally runs with a scrubbed environment,
``HOME`` set to its working directory and imports of ``scienceclaw`` blocked.
Violations are returned as human-readable strings ``"line N: ..."``.
"""
from __future__ import annotations

import ast
import os
import re

NETWORK_MODULES = frozenset({
    "socket", "socketserver", "ssl", "urllib", "urllib2", "urllib3", "requests", "http", "httpx", "httplib2",
    "ftplib", "aiohttp", "smtplib", "telnetlib", "poplib", "imaplib", "nntplib", "xmlrpc", "websocket",
    "websockets", "paramiko", "pycurl", "asyncssh", "grpc", "zmq", "webbrowser", "openai", "anthropic",
    "boto3", "botocore", "gcsfs", "s3fs", "fsspec", "huggingface_hub", "wget", "datasets", "tensorflow_datasets",
    "pooch", "kaggle", "gdown", "yfinance", "pandas_datareader", "wikipedia", "ogb", "tdc", "medmnist",
})
# Library entry points that download data (possibly including held-out labels) from the network.
NETWORK_DOTTED = ("sklearn.datasets.fetch_", "torch.hub", "torchvision.datasets", "torch_geometric.datasets",
                  "deepchem.molnet", "keras.utils.get_file", "tensorflow.keras.utils.get_file", "nltk.download",
                  "seaborn.load_dataset", "Bio.Entrez", "Bio.ExPASy", "gensim.downloader", "pubchempy")
PROCESS_MODULES = frozenset({"subprocess", "pty", "ctypes", "cffi", "pexpect", "sh", "plumbum"})
INTERNAL_MODULES = frozenset({"scienceclaw"})
DYNAMIC_IMPORTERS = frozenset({"importlib.import_module", "importlib.__import__", "builtins.__import__"})

OS_PROCESS = frozenset({
    "system", "popen", "fork", "forkpty", "kill", "killpg", "execl", "execle", "execlp", "execlpe", "execv",
    "execve", "execvp", "execvpe", "spawnl", "spawnle", "spawnlp", "spawnlpe", "spawnv", "spawnve", "spawnvp",
    "spawnvpe", "posix_spawn", "posix_spawnp", "startfile",
})
OS_ENV = frozenset({"environ", "environb", "getenv", "getenvb", "putenv", "unsetenv"})
OS_ESCAPE = frozenset({"chdir", "fchdir", "chroot"})
FORBIDDEN_DOTTED = {
    **{f"os.{n}": "process spawning" for n in OS_PROCESS},
    **{f"posix.{n}": "process spawning" for n in OS_PROCESS},
    **{f"os.{n}": "environment access" for n in OS_ENV},
    **{f"posix.{n}": "environment access" for n in OS_ENV},
    **{f"os.{n}": "changing the working directory" for n in OS_ESCAPE},
    "os.path.expanduser": "home-directory access",
    "os.pardir": "parent-directory traversal", "os.path.pardir": "parent-directory traversal",
    "os.path.expandvars": "environment access",
    "sys.modules": "access to loaded modules",
    "builtins.eval": "eval", "builtins.exec": "exec", "builtins.compile": "compile",
}
FORBIDDEN_CALL_NAMES = {"eval": "eval of strings", "exec": "exec of strings", "compile": "compile of strings",
                        "__import__": "dynamic import via __import__", "breakpoint": "interactive debugger"}
FORBIDDEN_METHODS = {"home": "home-directory access (Path.home())", "expanduser": "home-directory access"}
INTROSPECTION_ATTRS = frozenset({"__subclasses__", "__globals__", "__builtins__", "__code__", "__closure__",
                                 "f_globals", "f_locals", "f_back", "gi_frame", "__loader__"})
FORBIDDEN_GETATTR_NAMES = OS_PROCESS | OS_ENV | OS_ESCAPE | {"eval", "exec", "compile", "__import__"}

KNOWN_ROOTS = frozenset({   # single-component absolute paths flagged on their own ("/tmp", "/Users")
    "Users", "home", "root", "etc", "tmp", "var", "private", "usr", "opt", "proc", "sys", "dev", "mnt", "Volumes",
    "scratch", "gpfs", "lustre",
})
_URL_PREFIXES = ("http://", "https://", "ftp://", "ftps://", "s3://", "gs://", "gcs://", "hf://", "file://", "ssh://")
_PATH_COMPONENT = re.compile(r"[A-Za-z0-9_.@+~-]+")
_WINDOWS_ABS = re.compile(r"^[A-Za-z]:[\\/]")
_DATASET_MARKERS = ("datasets/scienceclaw", ".cache/scienceclaw/datasets")
_CREDENTIAL_MARKERS = (".config/scienceclaw", "client.json", "/.ssh", ".ssh/", "id_rsa", "id_ed25519", ".netrc", ".aws/",
                       "api_key", "api-key", "apikey", "secret_key", "access_token", "credentials.json",
                       "openai_api_key", "anthropic_api_key", ".pgpass", ".git-credentials")


def _dataset_markers() -> tuple[str, ...]:
    markers = list(_DATASET_MARKERS)
    root = os.environ.get("SCIENCECLAW_DATA_ROOT", "")
    if root and len(root) > 4:
        markers.append(root.lower())
    return tuple(markers)


def _is_network_dotted(dotted: str) -> bool:
    for p in NETWORK_DOTTED:
        if p.endswith("_"):
            if dotted.startswith(p):
                return True
        elif dotted == p or dotted.startswith(p + "."):
            return True
    return dotted.endswith(".get_rdataset")


def _string_violation(s: str, *, leading: bool = True, docstring: bool = False) -> str | None:
    """Classify a string literal; None when harmless.

    ``leading`` is False for literal pieces that continue an expression (``f"{d}/x"``, ``a + "/x"``),
    which cannot start an absolute path; ``docstring`` literals are only checked for dataset /
    credential references.
    """
    low = s.lower()
    for m in _dataset_markers():
        if m in low:
            return f"reference to the protected data root ({s[:80]!r})"
    for m in _CREDENTIAL_MARKERS:
        if m in low:
            return f"reference to credentials / key material ({s[:80]!r})"
    if docstring:
        return None
    st = s.strip()
    if st == ".config" or ".config/" in st or ".config\\" in st or st.endswith("/.config"):
        return f"reference to a configuration directory ({s[:80]!r})"
    if st == ".." or "../" in st or "..\\" in st or st.endswith("/..") or st.endswith("\\.."):
        return f"'..' path traversal ({s[:80]!r})"
    if low.lstrip().startswith(_URL_PREFIXES):
        return f"URL ({s[:80]!r}): code nodes have no network access; use the provided inputs"
    if not leading:
        return None
    if st == "~" or st.startswith("~/") or st.startswith("~\\"):
        return f"home-directory path ({s[:80]!r})"
    if _WINDOWS_ABS.match(st):
        return f"absolute filesystem path ({s[:80]!r})"
    if st.startswith("/") and not st.startswith("//"):
        comps = [c for c in st.split("/") if c]
        if comps and all(_PATH_COMPONENT.fullmatch(c) for c in comps):
            if comps[0] in KNOWN_ROOTS or len(comps) >= 2:
                kind = "home directory" if comps[0] in ("Users", "home", "root") else "absolute filesystem path"
                return f"{kind} ({s[:80]!r}); code nodes may only use their inputs and relative paths"
    return None


class _Scanner(ast.NodeVisitor):
    def __init__(self) -> None:
        self.viol: list[str] = []
        self.aliases: dict[str, str] = {}   # local name -> dotted module / object path

    def add(self, node: ast.AST, msg: str) -> None:
        line = getattr(node, "lineno", 0)
        text = f"line {line}: {msg}"
        if text not in self.viol:
            self.viol.append(text)

    # ------------------------------------------------------------------ imports
    def _check_module(self, node: ast.AST, mod: str, how: str = "import") -> None:
        root = mod.split(".")[0]
        if _is_network_dotted(mod):
            self.add(node, f"{how} of data-download / network API {mod!r} is not allowed")
        elif root in NETWORK_MODULES:
            self.add(node, f"{how} of network module {mod!r} is not allowed")
        elif root in PROCESS_MODULES:
            self.add(node, f"{how} of process/FFI module {mod!r} is not allowed")
        elif root in INTERNAL_MODULES:
            self.add(node, f"{how} of the ScienceClaw package ({mod!r}) is not allowed in code nodes")

    def visit_Import(self, node: ast.Import) -> None:
        for a in node.names:
            self._check_module(node, a.name)
            if a.asname:
                self.aliases[a.asname] = a.name
            else:
                self.aliases[a.name.split(".")[0]] = a.name.split(".")[0]
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = node.module or ""
        if node.level and node.level > 0:
            self.add(node, "relative imports are not allowed in code nodes")
        self._check_module(node, mod)
        for a in node.names:
            full = f"{mod}.{a.name}" if mod else a.name
            if _is_network_dotted(full) and not _is_network_dotted(mod):
                self.add(node, f"import of data-download / network API {full!r} is not allowed")
            if full in FORBIDDEN_DOTTED:
                self.add(node, f"import of {full!r} ({FORBIDDEN_DOTTED[full]}) is not allowed")
            elif mod in ("builtins",) and a.name in FORBIDDEN_CALL_NAMES:
                self.add(node, f"import of builtins.{a.name} is not allowed")
            elif mod == "importlib" and a.name == "import_module":
                pass  # checked at call sites (literal, allowed module names only)
            self.aliases[a.asname or a.name] = full
        self.generic_visit(node)

    # --------------------------------------------------------------- names
    def _dotted(self, node: ast.AST) -> str | None:
        parts: list[str] = []
        cur = node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if not isinstance(cur, ast.Name):
            return None
        base = self.aliases.get(cur.id, cur.id)
        return ".".join([base] + list(reversed(parts)))

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in INTROSPECTION_ATTRS:
            self.add(node, f"introspection attribute {node.attr!r} is not allowed")
        dotted = self._dotted(node)
        if dotted is not None and _is_network_dotted(dotted):
            self.add(node, f"use of data-download / network API {dotted!r} is not allowed")
        if dotted is not None:
            for key, why in FORBIDDEN_DOTTED.items():
                if dotted == key or dotted.startswith(key + "."):
                    self.add(node, f"use of {key} ({why}) is not allowed")
                    break
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        target = self.aliases.get(node.id)
        if target and target != node.id and target in FORBIDDEN_DOTTED:
            self.add(node, f"use of {target} ({FORBIDDEN_DOTTED[target]}) is not allowed")
        if node.id == "__builtins__":
            self.add(node, "access to __builtins__ is not allowed")
        self.generic_visit(node)

    # --------------------------------------------------------------- calls
    def visit_Call(self, node: ast.Call) -> None:
        f = node.func
        if isinstance(f, ast.Name):
            target = self.aliases.get(f.id, f.id)
            if f.id in FORBIDDEN_CALL_NAMES and f.id not in self.aliases:
                self.add(node, f"{FORBIDDEN_CALL_NAMES[f.id]} is not allowed")
            if target in DYNAMIC_IMPORTERS:
                self._check_dynamic_import(node)
            if f.id == "getattr" and len(node.args) >= 2:
                a1 = node.args[1]
                if isinstance(a1, ast.Constant) and isinstance(a1.value, str) and a1.value in FORBIDDEN_GETATTR_NAMES:
                    self.add(node, f"getattr(..., {a1.value!r}) reaches a forbidden function")
        elif isinstance(f, ast.Attribute):
            dotted = self._dotted(f)
            if dotted in DYNAMIC_IMPORTERS:
                self._check_dynamic_import(node)
            if f.attr in FORBIDDEN_METHODS and dotted != "os.path.expanduser":
                recv = self._dotted(f.value) or ""
                # Path.home(), pathlib.Path.home(), Path(...).expanduser(); other .home() methods are fine
                if f.attr == "expanduser" or recv == "Path" or recv.endswith(".Path"):
                    self.add(node, f"{FORBIDDEN_METHODS[f.attr]} is not allowed")
        self.generic_visit(node)

    def _check_dynamic_import(self, node: ast.Call) -> None:
        if not node.args or not (isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
            self.add(node, "dynamic import of a non-literal module name is not allowed")
            return
        self._check_module(node, node.args[0].value, how="dynamic import")

    # ------------------------------------------------------------- literals
    def _check_str(self, node: ast.AST, val: object, *, leading: bool = True, docstring: bool = False) -> None:
        if isinstance(val, bytes):
            val = val.decode("latin-1")
        if isinstance(val, str) and val:
            why = _string_violation(val, leading=leading, docstring=docstring)
            if why:
                self.add(node, why)

    def visit_Constant(self, node: ast.Constant) -> None:
        self._check_str(node, node.value)

    def visit_Expr(self, node: ast.Expr) -> None:
        if isinstance(node.value, ast.Constant):      # docstring / bare string statement
            self._check_str(node.value, node.value.value, docstring=True)
            return
        self.generic_visit(node)

    def visit_JoinedStr(self, node: ast.JoinedStr) -> None:
        for i, part in enumerate(node.values):
            if isinstance(part, ast.Constant):
                self._check_str(part, part.value, leading=(i == 0))
            else:
                self.visit(part)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        self.visit(node.left)
        if isinstance(node.op, ast.Add) and isinstance(node.right, ast.Constant):
            self._check_str(node.right, node.right.value, leading=False)
        else:
            self.visit(node.right)


def scan_code(code: str) -> list[str]:
    """Return human-readable leakage / sandbox violations of generated code ([] if clean).

    Code that does not parse only gets the raw-text dataset-root check (it cannot run anyway; the
    worker reports the syntax error).
    """
    if not isinstance(code, str):
        return ["code is not a string"]
    viol: list[str] = []
    low = code.lower()
    for m in _dataset_markers():
        if m in low:
            line = low[: low.index(m)].count("\n") + 1
            viol.append(f"line {line}: reference to the protected data root is not allowed")
            break
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return viol
    sc = _Scanner()
    sc.visit(tree)
    for v in sc.viol:
        if viol and "protected data root" in v:
            continue   # already reported once by the raw-text check
        if v not in viol:
            viol.append(v)
    return viol
