"""Node tracebacks keep the author's own frame and shorten library paths (FoR42: the node frame was clipped away)."""
from __future__ import annotations

from scienceclaw.runtime import node_worker as nw


def _deep(n: int, filename: str):
    src = "def f(k):\n    if k == 0:\n        raise ValueError('bad shape')\n    return f(k - 1)\n"
    ns: dict = {}
    exec(compile(src, filename, "exec"), ns)
    return ns["f"](n)


def test_format_node_exception_keeps_node_frame_and_marks_omitted_library_frames():
    def node_run():
        return _deep(12, "/venv/lib/python3.11/site-packages/sklearn/x.py")
    node_run.__code__ = node_run.__code__.replace(co_filename=nw.CODE_FILENAME)
    try:
        node_run()
    except ValueError as ex:
        text = nw.format_node_exception(ex)
    assert nw.CODE_FILENAME in text and "library frame(s) omitted" in text
    assert "<site>/sklearn/x.py" in text and "site-packages" not in text
    assert text.rstrip().endswith("ValueError: bad shape")
    assert len(text) < 1500
