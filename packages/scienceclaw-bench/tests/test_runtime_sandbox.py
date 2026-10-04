"""runtime.sandbox / node_worker: subprocess execution of code nodes."""
from __future__ import annotations

import ast
import threading
import time
from pathlib import Path

import numpy as np

from scienceclaw.core.graph import Node
from scienceclaw.core.schema import PortSchema
from scienceclaw.runtime import sandbox
from scienceclaw.runtime.sandbox import run_code_node


def code_node(code: str, outputs=("y",), config=None, nid="c1") -> Node:
    return Node(nid, "code", code=code, config=dict(config or {}), inputs={"x": PortSchema("any")},
                outputs={o: PortSchema("any") for o in outputs})


def test_ok_run_seeded_and_stdout_captured(tmp_path) -> None:
    code = ("import numpy as np, random\n"
            "print('hello from node')\n"
            "def run(inputs, config):\n"
            "    print('mean', float(np.mean(inputs['x'])))\n"
            "    return {'y': np.random.rand(3) + inputs['x'].sum(), 'r': random.random(), 'extra': 1}\n")
    node = code_node(code, outputs=("y", "r"), config={"seed": 7})
    x = np.arange(4.0)
    out1, meta1 = run_code_node(node, {"x": x}, tmp_path, timeout_s=30)
    out2, meta2 = run_code_node(node, {"x": x}, tmp_path, timeout_s=30)
    assert meta1["status"] == "ok", meta1
    assert set(out1) == {"y", "r"}                         # extra keys dropped
    np.testing.assert_array_equal(out1["y"], out2["y"])     # seeded -> reproducible
    assert out1["r"] == out2["r"]
    rng = np.random.RandomState(7)
    np.testing.assert_allclose(out1["y"], rng.rand(3) + 6.0)
    assert "hello from node" in meta1["stdout_tail"] and "mean 1.5" in meta1["stdout_tail"]
    assert meta1["returncode"] == 0 and not meta1["timed_out"]
    work = Path(meta1["work_dir"])
    assert work.parent == tmp_path / "work" and (work / "node_code.py").exists()
    assert not (work / "_inputs.pkl").exists() and not (work / "_outputs.pkl").exists()
    assert meta1["work_dir"] != meta2["work_dir"]


def test_error_traceback_points_to_node_code(tmp_path) -> None:
    code = "def helper(v):\n    return 1 / v\n\ndef run(inputs, config):\n    return {'y': helper(0)}\n"
    out, meta = run_code_node(code_node(code), {"x": 1}, tmp_path, timeout_s=30)
    assert out is None and meta["status"] == "error"
    err = meta["error"]
    assert "ZeroDivisionError" in err and 'File "node_code.py", line 2' in err
    assert "node_worker" not in err                        # worker frames are hidden
    assert len(err) <= 4000


def test_missing_output_and_non_dict(tmp_path) -> None:
    out, meta = run_code_node(code_node("def run(inputs, config):\n    return {'z': 1}\n"), {"x": 1}, tmp_path, 30)
    assert out is None and "declared output port(s) ['y']" in meta["error"]
    out, meta = run_code_node(code_node("def run(inputs, config):\n    return [1]\n"), {"x": 1}, tmp_path, 30)
    assert out is None and "must return a dict" in meta["error"]
    out, meta = run_code_node(code_node("x = 1\n"), {"x": 1}, tmp_path, 30)
    assert out is None and "must define a function run" in meta["error"]
    out, meta = run_code_node(code_node("def run(inputs, config):\n    raise SystemExit(3)\n"), {"x": 1}, tmp_path, 30)
    assert out is None and "SystemExit" in meta["error"]
    out, meta = run_code_node(code_node("def run(inputs, config):\n    return {'y': lambda: 1}\n"), {"x": 1}, tmp_path, 30)
    assert out is None and meta["status"] == "error" and "must be picklable" in meta["error"]


def test_timeout_kills_process_group(tmp_path) -> None:
    code = ("import time, subprocess\n"
            "def run(inputs, config):\n"
            "    time.sleep(60)\n"
            "    return {'y': 1}\n")
    t0 = time.monotonic()
    out, meta = run_code_node(code_node(code), {"x": 1}, tmp_path, timeout_s=1.0)
    assert time.monotonic() - t0 < 15
    assert out is None and meta["status"] == "timeout" and meta["timed_out"]
    assert "exceeded 1.0 s" in meta["error"]


def test_environment_is_scrubbed_and_home_is_workdir(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FAKE_SERVICE_API_KEY", "should-not-leak")
    monkeypatch.setenv("SOME_TOKEN", "nope")
    code = ("import os, sys\n"
            "def run(inputs, config):\n"
            "    env = dict(os.environ)\n"
            "    return {'y': {'env': env, 'cwd': os.getcwd(), 'llm_loaded': 'scienceclaw.llm' in sys.modules,\n"
            "                  'loaded': sorted(m for m in sys.modules if m.startswith('scienceclaw'))}}\n")
    out, meta = run_code_node(code_node(code), {"x": 1}, tmp_path, 30)
    assert meta["status"] == "ok", meta
    env = out["y"]["env"]
    assert "FAKE_SERVICE_API_KEY" not in env and "SOME_TOKEN" not in env
    assert not any("should-not-leak" in v for v in env.values())
    assert Path(env["HOME"]).resolve() == Path(out["y"]["cwd"]).resolve() == Path(meta["work_dir"]).resolve()
    assert env["PYTHONHASHSEED"] == "0" and env["OMP_NUM_THREADS"] == "2" and env["OPENBLAS_NUM_THREADS"] == "2"
    assert out["y"]["llm_loaded"] is False
    assert set(out["y"]["loaded"]) <= {"scienceclaw", "scienceclaw.runtime", "scienceclaw.runtime.node_worker"}


def test_scienceclaw_imports_blocked_in_node(tmp_path) -> None:
    code = "def run(inputs, config):\n    import scienceclaw.bench.task\n    return {'y': 1}\n"
    out, meta = run_code_node(code_node(code), {"x": 1}, tmp_path, 30)
    assert out is None and "may not import 'scienceclaw.bench'" in meta["error"]


def test_worker_module_never_imports_llm() -> None:
    src = Path(sandbox.__file__).with_name("node_worker.py").read_text()
    tree = ast.parse(src)
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            mods.add(n.module or "")
    assert not any(m.startswith("scienceclaw") or m.startswith(".") or m == "" for m in mods), mods


def test_global_semaphore_limits_concurrency(tmp_path) -> None:
    code = ("import time\n"
            "def run(inputs, config):\n"
            "    t0 = time.time()\n"
            "    time.sleep(0.4)\n"
            "    return {'y': (t0, time.time())}\n")
    old = sandbox.max_concurrent_subprocesses()
    sandbox.set_max_concurrent_subprocesses(1)
    try:
        results: list = []

        def go(i: int) -> None:
            results.append(run_code_node(code_node(code, nid=f"n{i}"), {"x": i}, tmp_path, 30))

        threads = [threading.Thread(target=go, args=(i,)) for i in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        sandbox.set_max_concurrent_subprocesses(old)
    spans = sorted(r[0]["y"] for r in results)
    assert all(r[1]["status"] == "ok" for r in results)
    assert spans[0][1] <= spans[1][0] + 0.05               # the second started after the first finished


def test_network_is_refused_inside_nodes(tmp_path) -> None:
    # the static scan would reject this code; run it directly to exercise the worker-level guard
    code = ("import socket, os\n"
            "def run(inputs, config):\n"
            "    out = {}\n"
            "    try:\n"
            "        socket.create_connection(('93.184.216.34', 80), timeout=2)\n"
            "        out['connect'] = 'connected'\n"
            "    except PermissionError as ex:\n"
            "        out['connect'] = str(ex)\n"
            "    try:\n"
            "        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
            "        s.settimeout(2)\n"
            "        s.connect(('93.184.216.34', 80))\n"
            "        out['raw'] = 'connected'\n"
            "    except PermissionError as ex:\n"
            "        out['raw'] = str(ex)\n"
            "    try:\n"
            "        socket.getaddrinfo('example.org', 80)\n"
            "        out['dns'] = 'resolved'\n"
            "    except PermissionError as ex:\n"
            "        out['dns'] = str(ex)\n"
            "    a, b = socket.socketpair()\n"
            "    a.sendall(b'ok')\n"
            "    out['local'] = b.recv(2).decode()\n"
            "    out['proxy'] = os.environ.get('HTTPS_PROXY')\n"
            "    return {'y': out}\n")
    out, meta = run_code_node(code_node(code), {"x": 1}, tmp_path, 30)
    assert meta["status"] == "ok", meta
    y = out["y"]
    assert "disabled inside code nodes" in y["connect"] and "disabled inside code nodes" in y["raw"]
    assert "DNS" in y["dns"]
    assert y["local"] == "ok" and y["proxy"] == "http://127.0.0.1:9"
