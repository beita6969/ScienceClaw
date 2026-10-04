from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
from scienceclaw.bench.tasks.for46_swebench import Adapter

def _git(cwd: Path, *args: str) -> str:
    p = subprocess.run(["git", *args], cwd=cwd, text=True, capture_output=True, check=True); return p.stdout.strip()

def _fixture(tmp_path: Path):
    root = tmp_path / "data"; repo = root / "for46-swebench-verified" / "repos" / "demo" / "pkg"; repo.mkdir(parents=True)
    (repo / "calc.py").write_text("def add(x):\n    return x - 1\n")
    (repo / "tests").mkdir(); (repo / "tests/test_calc.py").write_text("from calc import add\n\ndef test_base():\n    assert 1 + 1 == 2\n")
    _git(repo, "init", "-q"); _git(repo, "config", "user.email", "test@example.invalid"); _git(repo, "config", "user.name", "test"); _git(repo, "add", "."); _git(repo, "commit", "-qm", "base"); base = _git(repo, "rev-parse", "HEAD")
    hidden = "diff --git a/tests/test_calc.py b/tests/test_calc.py\n--- a/tests/test_calc.py\n+++ b/tests/test_calc.py\n@@ -1,4 +1,8 @@\n from calc import add\n \n def test_base():\n     assert 1 + 1 == 2\n+\n+\n+def test_issue():\n+    assert add(2) == 3\n"
    row = {"instance_id":"demo__calc-1", "repo":"demo/pkg", "base_commit":base, "problem_statement":"Make add increment.", "version":"1", "repo_path":str(repo), "test_patch":hidden, "FAIL_TO_PASS":["tests/test_calc.py::test_issue"], "PASS_TO_PASS":["tests/test_calc.py::test_base"], "eval_command":[sys.executable,"-m","pytest","-q","{tests}"]}
    d = root / "for46-swebench-verified"; d.mkdir(exist_ok=True); (d / "instances.jsonl").write_text(json.dumps(row)+"\n"); return root

def test_patch_evaluation_and_no_gold_exposure(tmp_path: Path):
    a = Adapter(data_root=_fixture(tmp_path), hidden_timeout_s=30); ok, why = a.available(); assert ok, why; ep = a.build_episodes("ood", 1, 7, 1)[0]
    public = ep.tool("load_eval_inputs").fn({}, {})["instances"][0]; assert "test_patch" not in public and "FAIL_TO_PASS" not in public
    patch = "diff --git a/calc.py b/calc.py\n--- a/calc.py\n+++ b/calc.py\n@@ -1,2 +1,2 @@\n def add(x):\n-    return x - 1\n+    return x + 1\n"
    r = ep.evaluate([patch], None); assert r.primary == 1.0 and r.h["patch_format"]

def test_test_changes_and_empty_patch_rejected(tmp_path: Path):
    ep = Adapter(data_root=_fixture(tmp_path), hidden_timeout_s=30).build_episodes("ood", 1, 7, 1)[0]
    bad = "diff --git a/tests/test_calc.py b/tests/test_calc.py\n--- a/tests/test_calc.py\n+++ b/tests/test_calc.py\n@@ -1 +1 @@\n-x\n+y\n"
    assert not ep.evaluate([bad], None).h["patch_format"]; assert not ep.evaluate([""], None).h["patch_format"]
