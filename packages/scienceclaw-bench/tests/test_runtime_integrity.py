"""runtime.integrity.scan_code: leakage / sandbox violations, with few false positives."""
from __future__ import annotations

import pytest

from scienceclaw.runtime.integrity import scan_code

POSITIVES = {
    "abs_path_read": "import pandas as pd\ndef run(inputs, config):\n    return {'d': pd.read_csv('/Users/admin/Datasets/x.csv')}",
    "abs_path_generic": "def run(inputs, config):\n    open('/tmp/out.txt', 'w').write('x')\n    return {}",
    "abs_path_single_root": "import os\ndef run(inputs, config):\n    return {'l': os.listdir('/etc')}",
    "windows_path": "def run(inputs, config):\n    return {'p': open('C:\\\\data\\\\x.csv').read()}",
    "dotdot": "def run(inputs, config):\n    return {'t': open('../labels.csv').read()}",
    "dotdot_alone": "import os\ndef run(inputs, config):\n    return {'l': os.listdir('..')}",
    "chdir": "import os\ndef run(inputs, config):\n    os.chdir('x')\n    return {}",
    "pardir": "import os\ndef run(inputs, config):\n    return {'l': os.listdir(os.pardir)}",
    "dataset_root_marker": "ROOT = 'ScienceClaw-rebuild-20260928'\ndef run(inputs, config):\n    return {}",
    "dataset_root_comment": "# data lives in /Users/admin/Datasets/ScienceClaw-rebuild\ndef run(inputs, config):\n    return {}",
    "home_tilde": "def run(inputs, config):\n    return {'t': open('~/notes.txt').read()}",
    "home_path_home": "from pathlib import Path\ndef run(inputs, config):\n    return {'p': str(Path.home())}",
    "home_expanduser": "import os\ndef run(inputs, config):\n    return {'p': os.path.expanduser('x')}",
    "socket": "import socket\ndef run(inputs, config):\n    return {}",
    "urllib": "from urllib.request import urlopen\ndef run(inputs, config):\n    return {}",
    "requests_alias": "import requests as rq\ndef run(inputs, config):\n    return {}",
    "http_client": "import http.client\ndef run(inputs, config):\n    return {}",
    "httpx": "import httpx\ndef run(inputs, config):\n    return {}",
    "ftplib": "import ftplib\ndef run(inputs, config):\n    return {}",
    "url_literal": "import pandas as pd\ndef run(inputs, config):\n    return {'d': pd.read_csv('https://example.org/x.csv')}",
    "subprocess": "import subprocess\ndef run(inputs, config):\n    return {}",
    "os_system": "import os\ndef run(inputs, config):\n    os.system('ls')\n    return {}",
    "os_popen_alias": "import os as o\ndef run(inputs, config):\n    return {'x': o.popen('ls').read()}",
    "from_os_system": "from os import system\ndef run(inputs, config):\n    return {}",
    "pty": "import pty\ndef run(inputs, config):\n    return {}",
    "eval": "def run(inputs, config):\n    return {'v': eval('1+1')}",
    "exec": "def run(inputs, config):\n    exec('x = 1')\n    return {}",
    "compile": "def run(inputs, config):\n    return {'c': compile('1', 'f', 'eval')}",
    "dunder_import": "def run(inputs, config):\n    return {'m': __import__('os')}",
    "importlib_forbidden": "import importlib\ndef run(inputs, config):\n    return {'m': importlib.import_module('socket')}",
    "importlib_dynamic": "import importlib\ndef run(inputs, config):\n    return {'m': importlib.import_module(config['m'])}",
    "getattr_trick": "import os\ndef run(inputs, config):\n    getattr(os, 'system')('ls')\n    return {}",
    "environ": "import os\ndef run(inputs, config):\n    return {'k': os.environ.get('OPENAI_KEY')}",
    "getenv": "import os\ndef run(inputs, config):\n    return {'k': os.getenv('X')}",
    "from_os_environ": "from os import environ\ndef run(inputs, config):\n    return {'k': dict(environ)}",
    "credentials": "import json\ndef run(inputs, config):\n    return {'k': json.load(open('.config/student-api/client.json'))}",
    "config_dir": "def run(inputs, config):\n    return {'k': open('.config').read()}",
    "api_key_literal": "def run(inputs, config):\n    return {'k': config['api_key']}",
    "scienceclaw_import": "from scienceclaw.bench.tasks import for34_molhiv\ndef run(inputs, config):\n    return {}",
    "introspection": "def run(inputs, config):\n    return {'s': ().__class__.__base__.__subclasses__()}",
    "builtins": "def run(inputs, config):\n    return {'b': __builtins__}",
    "sys_modules": "import sys\ndef run(inputs, config):\n    return {'m': list(sys.modules)}",
}

NEGATIVES = {
    "numpy_pandas_sklearn": """
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

def run(inputs, config):
    '''Fit a ridge model; see https://scikit-learn.org for details.'''
    X = np.asarray(inputs["X"], dtype=float)
    y = np.asarray(inputs["y"], dtype=float)
    m = Ridge(alpha=config.get("alpha", 1.0)).fit(X, y)
    df = pd.DataFrame({"pred": m.predict(X)})
    df = df.eval("pred2 = pred * 2")
    return {"pred": df["pred"].to_numpy(), "note": f"{len(y)} rows / {X.shape[1]} cols"}
""",
    "regex_and_units": """
import re
def run(inputs, config):
    pat = re.compile(r"(\\d+)/(\\d+)")
    unit = "mg" + "/L"
    rate = f"{inputs['n']}/min"
    ratio = "a/b"
    return {"u": unit, "r": rate, "m": pat.findall("1/2"), "q": ratio, "s": "/", "e": "..."}
""",
    "relative_files_and_tempfile": """
import os, json, tempfile, pathlib
def run(inputs, config):
    os.makedirs("out", exist_ok=True)
    with open(os.path.join("out", "x.json"), "w") as f:
        json.dump({"a": 1}, f)
    p = pathlib.Path("out") / "y.txt"
    p.write_text("ok")
    with tempfile.TemporaryDirectory() as d:
        pass
    return {"n": len(os.listdir("out")), "cwd": os.getcwd(), "sep": os.path.sep}
""",
    "scipy_statsmodels_math": """
import math, statistics, random, itertools, collections
import scipy.stats as st
import statsmodels.api as sm
def run(inputs, config):
    rng = random.Random(config.get("seed", 0))
    xs = [rng.random() for _ in range(10)]
    return {"p": st.ttest_1samp(xs, 0.5).pvalue, "m": statistics.mean(xs), "c": math.comb(5, 2)}
""",
    "user_function_named_home": """
class Grid:
    def home(self):
        return 0
def run(inputs, config):
    return {"h": Grid().home()}
""",
    "from_re_import_compile": """
from re import compile
def run(inputs, config):
    return {"m": bool(compile("a+").match("aa"))}
""",
}


@pytest.mark.parametrize("name", sorted(POSITIVES))
def test_scan_flags_violations(name: str) -> None:
    v = scan_code(POSITIVES[name])
    assert v, f"{name} not flagged"
    assert all(isinstance(s, str) and s.startswith("line ") for s in v), v


@pytest.mark.parametrize("name", sorted(NEGATIVES))
def test_scan_accepts_ordinary_scientific_code(name: str) -> None:
    assert scan_code(NEGATIVES[name]) == []


def test_violation_messages_are_specific() -> None:
    v = scan_code("import requests\ndef run(inputs, config):\n    return {'p': open('/etc/passwd').read()}")
    assert any("line 1" in s and "network module 'requests'" in s for s in v)
    assert any("line 3" in s and "absolute filesystem path" in s for s in v)
    v = scan_code("import os\ndef run(inputs, config):\n    return {'k': os.environ['HOME']}")
    assert v == ["line 3: use of os.environ (environment access) is not allowed"]


def test_unparsable_code_only_gets_text_checks() -> None:
    assert scan_code("def run(inputs, config)\n    return {}") == []
    v = scan_code("def run(:\n    x = '/Users/admin/Datasets/ScienceClaw-rebuild'")
    assert len(v) == 1 and "dataset root" in v[0]


def test_env_configured_dataset_root(monkeypatch) -> None:
    monkeypatch.setenv("SCIENCECLAW_DATA_ROOT", "/data/private/benchmark_root")
    v = scan_code("def run(inputs, config):\n    return {'p': 'benchmark_root'}")
    assert v == []
    v = scan_code("P = 'x'  # /data/private/benchmark_root\ndef run(inputs, config):\n    return {}")
    assert v and "dataset root" in v[0]


@pytest.mark.parametrize("code", [
    "from sklearn.datasets import fetch_openml\ndef run(inputs, config):\n    return {}",
    "import sklearn.datasets as skd\ndef run(inputs, config):\n    return {'d': skd.fetch_california_housing()}",
    "import seaborn as sns\ndef run(inputs, config):\n    return {'d': sns.load_dataset('iris')}",
    "import statsmodels.api as sm\ndef run(inputs, config):\n    return {'d': sm.datasets.get_rdataset('x')}",
    "from ogb.graphproppred import PygGraphPropPredDataset\ndef run(inputs, config):\n    return {}",
    "from datasets import load_dataset\ndef run(inputs, config):\n    return {}",
    "import torch\ndef run(inputs, config):\n    return {'m': torch.hub.load('a', 'b')}",
    "from Bio import Entrez\ndef run(inputs, config):\n    return {}",
])
def test_data_download_apis_are_flagged(code: str) -> None:
    assert any("network" in v for v in scan_code(code)), scan_code(code)


def test_local_dataset_helpers_are_fine() -> None:
    code = ("from sklearn.datasets import make_regression\nfrom Bio.Seq import Seq\n"
            "def run(inputs, config):\n    X, y = make_regression(n_samples=5, random_state=0)\n"
            "    return {'s': str(Seq('ACGT').reverse_complement()), 'n': len(y)}\n")
    assert scan_code(code) == []
