import scilib


def test_describe_returns_module_doc(monkeypatch):
    monkeypatch.delenv("SCIENCECLAW_NO_SCILIB", raising=False)
    assert scilib.describe("ocrfix").startswith("Library `scilib.ocrfix`")


def test_describe_empty_when_switched_off(monkeypatch):
    monkeypatch.setenv("SCIENCECLAW_NO_SCILIB", "1")
    assert scilib.describe("ocrfix") == ""
