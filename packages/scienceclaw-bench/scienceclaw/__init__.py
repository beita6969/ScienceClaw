"""ScienceClaw rebuild."""
import sys as _sys
from pathlib import Path as _Path

# ``scilib`` (domain libraries for code nodes) sits next to this package; an editable install only maps ``scienceclaw``.
_ROOT = str(_Path(__file__).resolve().parents[1])
if (_Path(_ROOT) / "scilib").is_dir() and _ROOT not in _sys.path:
    _sys.path.append(_ROOT)
