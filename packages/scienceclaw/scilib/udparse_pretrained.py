"""Frozen pretrained French dependency parser (Stanza 1.14 "gsd_camembert-large" models, weights stored locally, no network).

Input sentences as for ``scilib.udparse``: dicts ``{"words": [{"form": ...}, ...]}`` or plain lists of form strings; the
words are used as given (no tokenisation, no multiword-token splitting). Output per sentence is
``{"head": [int] * n, "deprel": [str] * n, "upos": [str] * n}`` with 1-based heads and 0 for the root, i.e. the parse
format of ``scilib.udparse`` plus the predicted UPOS tags; the trees are always valid (one root, in range, acyclic).

available() -> bool
    True when torch, transformers and stanza can be imported and the stored weights are present, or when a remote GPU
    worker is attached (then parsing runs on that worker and returns the same values; calls are cached by their
    arguments).
parse_gold_tokens(sentences, batch_size=32, device="auto", model=None) -> list[dict]
    Tags (UPOS, lemma) and parses the given words with the stored pretrained pipeline: a CamemBERT-large encoder with
    the Stanza tagger and a deep-biaffine dependency parser on top. Relation labels carry UD subtypes after ':' (for
    example ``nsubj:pass``), which the universal-relation LAS ignores. Words are batched by length. This project keeps
    this interface in frozen-inference mode; adaptation and training calls are disabled before weights, a remote
    worker, or a trainer can be touched.

Models: Stanza (Qi et al., ACL 2020 demos; Apache-2.0 code), the French-GSD models of release 1.14.0. They were trained on
the UD French-GSD training file of a recent UD release (CamemBERT-large from the ALMAnaCH group as the encoder, and
word vectors / character language models trained on news text), with its development file used for model selection.
Sentences from the French-GSD training and development files were seen in training; sentences of other treebanks (for
example French-PUD) were not. The annotation of the current French-GSD release differs from UD 2.2 in places (relation
choices, lemma and feature conventions), so a UD 2.2 gold tree can disagree with a correct parse of the current scheme.
Cost: about 20-40 s to load the models in every process; a few hundred sentences per second on a GPU, on CPU (2-4
threads) roughly 5-10 sentences per second. Predictions are deterministic on CPU.
"""
from __future__ import annotations

import os

from . import _remote
from ._pretrained import have_module, model_path, set_cpu_threads, switched_off, torch_device

__all__ = ["available", "parse_gold_tokens"]

_DIR = "stanza_fr"
_PIPE: dict = {}


def _stanza_dir():
    d = model_path(_DIR)
    return d if d is not None and (d / "fr" / "depparse" / "gsd_camembert-large.pt").is_file() else None


def _local_ok() -> bool:
    return (not switched_off() and have_module("torch") and have_module("transformers") and have_module("stanza")
            and _stanza_dir() is not None)


def available() -> bool:
    return _local_ok() or _remote.enabled()


def _pipeline(device: str, model: str | None = None):
    key = (device, model)
    if key in _PIPE:
        return _PIPE[key]
    hf = model_path("hf")
    if hf is not None:
        os.environ.setdefault("HF_HOME", str(hf))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import stanza
    d = _stanza_dir()
    p = lambda kind, name: str(d / "fr" / kind / name)                       # noqa: E731
    charlm = {"forward_charlm_path": p("forward_charlm", "newswiki.pt"), "backward_charlm_path": p("backward_charlm", "newswiki.pt")}
    kw: dict = {
        "lang": "fr", "dir": str(d), "processors": "tokenize,pos,lemma,depparse", "tokenize_pretokenized": True,
        "download_method": None, "use_gpu": device != "cpu", "logging_level": "ERROR",
        "pos_model_path": p("pos", "gsd_camembert-large.pt"), "pos_pretrain_path": p("pretrain", "conll17.pt"),
        "depparse_model_path": model or p("depparse", "gsd_camembert-large.pt"), "depparse_pretrain_path": p("pretrain", "conll17.pt"),
        "lemma_model_path": p("lemma", "gsd_nocharlm.pt"),
    }
    for proc in ("pos", "depparse"):
        for k, v in charlm.items():
            kw[f"{proc}_{k}"] = v
    _PIPE[key] = stanza.Pipeline(**kw)
    return _PIPE[key]


def _forms(s) -> list[str]:
    if isinstance(s, dict):
        return [str(w["form"]) for w in s["words"]]
    return [str(w) for w in s]


def _valid_tree(heads: list[int]) -> bool:
    n = len(heads)
    if any(h < 0 or h > n for h in heads) or sum(1 for h in heads if h == 0) != 1:
        return False
    for i in range(1, n + 1):
        seen, j = set(), i
        while j != 0:
            if j in seen:
                return False
            seen.add(j)
            j = heads[j - 1]
    return True


def _safe_tree(heads: list[int], rels: list[str]) -> tuple[list[int], list[str]]:
    """Return the parse unchanged when it is a valid tree, else a right-branching chain (never expected with Stanza)."""
    if _valid_tree(heads):
        return heads, rels
    n = len(heads)
    return [i + 2 if i + 1 < n else 0 for i in range(n)], ["dep" if i + 1 < n else "root" for i in range(n)]


def parse_gold_tokens(sentences, batch_size: int = 32, device: str = "auto", model: str | None = None) -> list[dict]:
    forms = [_forms(s) for s in sentences]
    out: list[dict | None] = [None] * len(forms)
    todo = [i for i, f in enumerate(forms) if f]
    for i, f in enumerate(forms):
        if not f:
            out[i] = {"head": [], "deprel": [], "upos": []}
    if todo and not _local_ok() and _remote.enabled():
        res = _remote.call("udparse_pretrained", "parse_gold_tokens",
                           {"sentences": [forms[i] for i in todo], "batch_size": batch_size, "device": device,
                            "model": str(model) if model else None})
        for i, r in zip(todo, res):
            out[i] = r
        todo = []
    if todo:
        if not _local_ok():
            raise RuntimeError("parse_gold_tokens: torch, transformers, stanza or the stored Stanza weights are not available here")
        dev = torch_device(device)
        if dev == "cpu":
            set_cpu_threads(4)
        nlp = _pipeline(dev, str(model) if model else None)
        todo.sort(key=lambda i: len(forms[i]))
        step = max(1, int(batch_size))
        for a in range(0, len(todo), step):
            idx = todo[a:a + step]
            doc = nlp([forms[i] for i in idx])
            if len(doc.sentences) != len(idx):
                raise RuntimeError(f"Stanza returned {len(doc.sentences)} sentences for {len(idx)}")
            for i, sent in zip(idx, doc.sentences):
                words = sent.words
                if len(words) != len(forms[i]):
                    raise RuntimeError(f"Stanza changed the word count of a sentence ({len(forms[i])} -> {len(words)})")
                heads = [int(w.head) for w in words]
                rels = [str(w.deprel) for w in words]
                heads, rels = _safe_tree(heads, rels)
                out[i] = {"head": heads, "deprel": rels, "upos": [str(w.upos) for w in words]}
    return out  # type: ignore[return-value]


_EXPL_PRONOUN = {"se", "s'", "me", "m'", "te", "t'", "nous", "vous"}
_EXPL_SUBJECT = {"il", "ce", "c'", "ça", "cela", "ils"}


def _closest_label(rel: str, form: str, known: set[str]) -> str:
    if rel in known:
        return rel
    base = rel.split(":")[0]
    if base == "expl":
        f = form.lower()
        want = "expl:pv" if f in _EXPL_PRONOUN else "expl:subj" if f in _EXPL_SUBJECT else "expl:comp"
        if want in known:
            return want
    if base in known:
        return base
    same = sorted(k for k in known if k.split(":")[0] == base)
    return same[0] if same else "dep"


def _conllu(sentences, known: set[str]) -> str:
    out = []
    for i, s in enumerate(sentences):
        words = s["words"]
        n = len(words)
        rows = []
        for j, w in enumerate(words, 1):
            h = int(w["head"])
            if h < 0 or h > n:
                raise ValueError("finetune: head index out of range")
            rel = _closest_label(str(w["deprel"]), str(w["form"]), known)
            cell = lambda k: (str(w.get(k)) if w.get(k) not in (None, "") else "_").replace("\t", " ")   # noqa: E731
            rows.append("\t".join([str(j), cell("form"), cell("lemma"), cell("upos"), "_", cell("feats"), str(h), rel, "_", "_"]))
        out.append(f"# sent_id = s{i}\n" + "\n".join(rows) + "\n")
    return "\n".join(out) + "\n"


def finetune(train_sentences, steps: int = 400, lr: float = 2e-4, seed: int = 0, batch_words: int = 1500,
             device: str = "auto", out_path: str | None = None) -> str:
    # This is deliberately the first operation.  Do not validate the training
    # data, inspect model roots, initialize a local pipeline, enqueue a remote
    # request, or construct a trainer before the policy rejection.
    raise RuntimeError(
        "udparse_pretrained.finetune is disabled by ScienceClaw policy; "
        "the parser is frozen-inference only"
    )
