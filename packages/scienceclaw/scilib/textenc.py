"""Scores of text pairs and text embeddings from three pretrained English models: a sentence encoder (BGE-large-en-v1.5), a relevance
cross-encoder (BGE-reranker-large) and a natural-language-inference cross-encoder (DeBERTa-v3-large trained on SNLI and MultiNLI).
Computing them needs torch, the weights and a GPU; when this interpreter cannot run them and a remote GPU worker is configured, the
calls run on the worker's GPU host and return the same values (answers are stored, so an identical call returns the same result).
None of the models saw the benchmark's labels; their training corpora are described under "Training data" below.

available() -> bool
    True when torch, transformers and the weights are available here, or a remote GPU worker is configured.
embed(texts, model="bge_large_en", query=False) -> float32 array (n_texts, 1024)
    L2-normalised sentence embeddings (CLS pooling, at most 512 tokens per text). ``query=True`` prepends the model's short retrieval
    instruction ("Represent this sentence for searching relevant passages: ") to every text; the dot product of two rows is their cosine.
relevance(passages, queries, model="bge_reranker_large") -> float32 array (n_passages, n_queries)
    Cross-encoder relevance logit of every (query, passage) pair (higher = more relevant; at most 256 tokens per pair). The full
    cross product is scored, so n_passages * n_queries pairs are run through the network.
nli(premises, hypotheses, model="nli_deberta_v3_large") -> float32 array (n_premises, n_hypotheses, 3)
    Log-probabilities over ``NLI_CLASSES`` = (contradiction, entailment, neutral) of every (premise, hypothesis) pair (at most 256 tokens
    per pair; log-softmax of the model's three logits). The full cross product is scored.
MODELS                                                dict name -> (kind, subdirectory of the model root)

Training data: BGE-large-en-v1.5 is a contrastively trained encoder on large public text-pair corpora (BAAI); BGE-reranker-large is a
cross-encoder initialised from XLM-RoBERTa-large and trained on multilingual relevance data (BAAI); the NLI model is DeBERTa-v3-large
fine-tuned on SNLI and MultiNLI (sentence-transformers cross-encoder). Texts of this benchmark may or may not be part of the public
corpora behind their pre-training; that cannot be ruled out.

Cost on one shared A100/H800-class GPU in half precision (measured on contract sentences of ~40 tokens): about 500 pairs per second for the
relevance cross-encoder, about 4,000 pairs per second for the NLI cross-encoder, about 280 texts per second for the sentence encoder. A
remote call adds a start-up of a few seconds; its answer is stored.
"""
from __future__ import annotations

import numpy as np

from . import _remote
from ._pretrained import have_module, model_path, switched_off

__all__ = ["available", "embed", "relevance", "nli", "MODELS", "NLI_CLASSES"]

MODELS = {"bge_large_en": ("embed", "textenc/bge_large_en"),
          "bge_reranker_large": ("relevance", "textenc/bge_reranker_large"),
          "nli_deberta_v3_large": ("nli", "textenc/nli_deberta_v3_large")}
NLI_CLASSES = ("contradiction", "entailment", "neutral")
_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_CACHE: dict = {}


def _dir(model: str):
    return model_path(*MODELS[model][1].split("/")) if model in MODELS else None


def _local_ok(model: str) -> bool:
    d = _dir(model)
    return (not switched_off() and have_module("torch") and have_module("transformers") and d is not None
            and any(d.glob("*.safetensors")))


def available() -> bool:
    return _local_ok("bge_large_en") or _remote.enabled()


def _check_model(model: str, kind: str) -> None:
    if model not in MODELS or MODELS[model][0] != kind:
        raise ValueError(f"model must be one of {[k for k, v in MODELS.items() if v[0] == kind]} for {kind}()")


def _check_texts(x, name: str) -> list[str]:
    if isinstance(x, str) or not hasattr(x, "__len__") or any(not isinstance(t, str) for t in x):
        raise ValueError(f"{name} must be a list of strings")
    return [t if t.strip() else "." for t in x]


def _load(model: str):
    if model not in _CACHE:
        import torch
        from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
        d = str(_dir(model))
        tok = AutoTokenizer.from_pretrained(d)
        cls = AutoModel if MODELS[model][0] == "embed" else AutoModelForSequenceClassification
        net = cls.from_pretrained(d).eval()
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        if dev == "cuda":
            net = net.half()
        _CACHE[model] = (tok, net.to(dev), dev)
    return _CACHE[model]


def _batches(lengths: np.ndarray, budget: int):
    order = np.argsort(-lengths, kind="stable")
    i = 0
    while i < len(order):
        n = max(1, budget // max(int(lengths[order[i]]), 1))
        yield order[i:i + n]
        i += n


def _embed_local(texts: list[str], model: str, query: bool) -> np.ndarray:
    import torch
    tok, net, dev = _load(model)
    texts = [(_QUERY_PREFIX + t) if query else t for t in texts]
    lens = np.array([len(t) for t in texts])
    out = np.zeros((len(texts), net.config.hidden_size), dtype=np.float32)
    with torch.inference_mode():
        for idx in _batches(lens, 40000):
            enc = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=512, return_tensors="pt").to(dev)
            h = net(**enc).last_hidden_state[:, 0].float()
            out[idx] = torch.nn.functional.normalize(h, dim=-1).cpu().numpy()
    return out


def _segment_ids(tok, texts: list[str]) -> list[list[int]]:
    """token ids of every text without special tokens (each text is tokenised once, however many pairs it takes part in)."""
    out: list[list[int]] = []
    for i in range(0, len(texts), 2048):
        out += tok(texts[i:i + 2048], add_special_tokens=False, truncation=True, max_length=254)["input_ids"]
    return out


def _template(tok):
    """special tokens (before the first segment, between the segments, after the second) of the tokenizer's pair encoding."""
    a = tok("hello", add_special_tokens=False)["input_ids"]
    b = tok("world", add_special_tokens=False)["input_ids"]
    full = tok("hello", "world")["input_ids"]

    def find(x, start):
        return next(i for i in range(start, len(full) - len(x) + 1) if full[i:i + len(x)] == x)
    ia = find(a, 0)
    ib = find(b, ia + len(a))
    return full[:ia], full[ia + len(a):ib], full[ib + len(b):]


def _pairs_local(left: list[str], right: list[str], model: str, kind: str) -> np.ndarray:
    """scores for every (left[i], right[j]): left = passages / premises, right = queries / hypotheses."""
    import torch
    tok, net, dev = _load(model)
    nl, nr = len(left), len(right)
    k = 1 if kind == "relevance" else 3
    out = np.zeros((nl, nr, k), dtype=np.float32)
    if nl == 0 or nr == 0:
        return out
    order_map = None
    if kind == "nli":
        cfg = {int(i): str(n).lower() for i, n in net.config.id2label.items()}
        order_map = [next(i for i, n in cfg.items() if n.startswith(c[:5])) for c in NLI_CLASSES]
    L, R = _segment_ids(tok, left), _segment_ids(tok, right)
    pre, mid, post = _template(tok)
    n_special = len(pre) + len(mid) + len(post)
    max_len = 256
    use_types = getattr(net.config, "type_vocab_size", 0) > 1
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    ii, jj = np.divmod(np.arange(nl * nr), nr)
    ids: list = []
    types: list = []
    for a, b in zip(ii, jj):
        x, y = (R[b], L[a]) if kind == "relevance" else (L[a], R[b])   # relevance: query first, passage second
        cut = max_len - n_special
        if len(x) + len(y) > cut:                                      # the passage / premise is the segment that is shortened
            if kind == "relevance":
                y = y[:max(cut - len(x), 1)]
            else:
                x = x[:max(cut - len(y), 1)]
        ids.append(pre + x + mid + y + post)
        if use_types:
            types.append([0] * (len(pre) + len(x) + len(mid)) + [1] * (len(y) + len(post)))
    lens = np.array([len(t) for t in ids])
    with torch.inference_mode():
        for idx in _batches(lens, 24000):
            width = int(lens[idx].max())
            inp = np.full((len(idx), width), pad, dtype=np.int64)
            att = np.zeros((len(idx), width), dtype=np.int64)
            for r, t in enumerate(idx):
                inp[r, :lens[t]] = ids[t]
                att[r, :lens[t]] = 1
            feed = {"input_ids": torch.from_numpy(inp).to(dev), "attention_mask": torch.from_numpy(att).to(dev)}
            if use_types:
                tt = np.zeros((len(idx), width), dtype=np.int64)
                for r, t in enumerate(idx):
                    tt[r, :lens[t]] = types[t]
                feed["token_type_ids"] = torch.from_numpy(tt).to(dev)
            lg = net(**feed).logits.float()
            if kind == "nli":
                lg = torch.log_softmax(lg[:, order_map], dim=-1)
            out[ii[idx], jj[idx]] = lg.cpu().numpy().reshape(len(idx), -1)
    return out


def embed(texts, model: str = "bge_large_en", query: bool = False) -> np.ndarray:
    _check_model(model, "embed")
    texts = _check_texts(texts, "texts")
    if not texts:
        return np.zeros((0, 1024), dtype=np.float32)
    if not _local_ok(model) and _remote.enabled():
        return np.asarray(_remote.call("textenc", "embed", {"texts": texts, "model": model, "query": bool(query)}), dtype=np.float32)
    if _local_ok(model):
        return _embed_local(texts, model, bool(query))
    raise RuntimeError("embed: neither local weights nor a remote GPU worker are available")


def _pairs(kind: str, left, right, model: str, lname: str, rname: str) -> np.ndarray:
    _check_model(model, kind)
    left, right = _check_texts(left, lname), _check_texts(right, rname)
    if not _local_ok(model) and _remote.enabled():
        res = _remote.call("textenc", kind, {lname: left, rname: right, "model": model})
        return np.asarray(res, dtype=np.float32)
    if _local_ok(model):
        out = _pairs_local(left, right, model, kind)
        return out[..., 0] if kind == "relevance" else out
    raise RuntimeError(f"{kind}: neither local weights nor a remote GPU worker are available")


def relevance(passages, queries, model: str = "bge_reranker_large") -> np.ndarray:
    return _pairs("relevance", passages, queries, model, "passages", "queries")


def nli(premises, hypotheses, model: str = "nli_deberta_v3_large") -> np.ndarray:
    return _pairs("nli", premises, hypotheses, model, "premises", "hypotheses")
