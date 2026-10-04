"""Amino-acid log-probabilities from a pretrained protein language model (ESM-2 650M, Lin et al., Science 2023) for the positions of
a wild-type sequence. The model is a masked language model trained on UniRef50 sequences (no experimental fitness labels).
Computing it needs torch, the weights and a GPU; when this interpreter cannot run them and a remote GPU worker is configured, the
calls run on the worker's GPU host and return the same values (answers are stored, so an identical call returns the same result).

available() -> bool
    True when torch, transformers and the weights are available here, or a remote GPU worker is configured.
site_logprobs(wild_type, positions=None, masked=True, model="esm2_650m") -> float array (n_positions, 20)
    Natural-log probabilities of the 20 amino acids (columns in the order of ``AA``; log-softmax over the model's whole vocabulary,
    then the 20 residue columns) at each of ``positions`` (1-based, rows in the order given; default all positions). ``wild_type`` is one sequence
    string (-> one array) or a dict name -> sequence (-> dict name -> array; ``positions`` is then None or a dict name ->
    positions). ``masked=True``: the position is replaced by the mask token and the model predicts it from the rest of the
    sequence (masked marginal, one forward pass per position). ``masked=False``: one forward pass of the unmasked sequence,
    the rows are the outputs at the positions (wild-type marginal). Sequences longer than 1,022 residues are scored in
    1,022-residue windows (each position in the window where it is closest to the centre).
plm_features(table, wild_type, model="esm2_650m") -> DataFrame
    One row per row of ``table`` (same index; columns ``position``, ``wt_aa``, ``mut_aa``; a column ``assay_id`` or ``item``
    selects the wild type when ``wild_type`` is a dict), columns:
    ``esm_llr_masked`` log p(mut) - log p(wt) of the masked marginal at the position; ``esm_llr_wt`` the same difference from the
    wild-type marginal; ``esm_logp_mut_masked`` / ``esm_logp_wt_masked`` the two log-probabilities behind the first column;
    ``esm_entropy_masked`` entropy (nats) of the masked marginal renormalised over the 20 residues; ``esm_site_mean_llr`` mean over the 19
    substitutions of the masked log-ratio at the position. Higher log-ratio = the model finds the mutant residue more likely
    than the wild-type residue in that context.

Cost on one GPU: the masked marginal takes about 0.02-0.05 s per position for a 300-residue sequence; a table of 16 assays takes
under two minutes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import _remote
from ._pretrained import have_module, model_path, switched_off

__all__ = ["available", "site_logprobs", "plm_features", "AA", "MODELS", "FEATURES"]

AA = "ARNDCQEGHILKMFPSTWYV"
MODELS = {"esm2_650m": "esm2_650m"}
FEATURES = ["esm_llr_masked", "esm_llr_wt", "esm_logp_mut_masked", "esm_logp_wt_masked", "esm_entropy_masked", "esm_site_mean_llr"]
_WIN = 1022
_CACHE: dict = {}


def _local_ok(model: str = "esm2_650m") -> bool:
    return (not switched_off() and have_module("torch") and have_module("transformers")
            and model in MODELS and model_path(MODELS[model], "model.safetensors") is not None)


def available() -> bool:
    return _local_ok() or _remote.enabled()


def _load(model: str):
    if model not in _CACHE:
        import torch
        from transformers import AutoTokenizer, EsmForMaskedLM
        d = str(model_path(MODELS[model]))
        tok = AutoTokenizer.from_pretrained(d)
        net = EsmForMaskedLM.from_pretrained(d).eval()
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _CACHE[model] = (tok, net.to(dev), dev, [tok.convert_tokens_to_ids(a) for a in AA])
    return _CACHE[model]


def _windows(L: int, positions: np.ndarray) -> dict[int, list[int]]:
    """window start -> 0-based positions scored from that window."""
    if L <= _WIN:
        return {0: [int(p) for p in positions]}
    out: dict[int, list[int]] = {}
    for p in positions:
        s = int(min(max(int(p) - _WIN // 2, 0), L - _WIN))
        out.setdefault(s, []).append(int(p))
    return out


def _score_one(seq: str, positions: np.ndarray, masked: bool, model: str, batch_tokens: int = 24000) -> np.ndarray:
    import torch
    tok, net, dev, aa_ids = _load(model)
    ids = np.array(tok(seq, add_special_tokens=False)["input_ids"], dtype=np.int64)
    L = len(ids)
    out = np.zeros((len(positions), 20), dtype=np.float32)
    where = {int(p): k for k, p in enumerate(positions)}
    cls_id, eos_id, mask_id = tok.cls_token_id, tok.eos_token_id, tok.mask_token_id
    with torch.inference_mode():
        for s, ps in _windows(L, positions).items():
            win = ids[s:s + _WIN]
            base = np.concatenate([[cls_id], win, [eos_id]])
            if masked:
                step = max(1, batch_tokens // len(base))
                for b in range(0, len(ps), step):
                    chunk = ps[b:b + step]
                    x = np.tile(base, (len(chunk), 1))
                    for r, p in enumerate(chunk):
                        x[r, p - s + 1] = mask_id
                    lg = net(input_ids=torch.from_numpy(x).to(dev)).logits
                    sel = torch.tensor([p - s + 1 for p in chunk], device=dev)
                    lp = torch.log_softmax(lg[torch.arange(len(chunk), device=dev), sel].float(), dim=-1)[:, aa_ids]
                    for r, p in enumerate(chunk):
                        out[where[p]] = lp[r].cpu().numpy()
            else:
                lg = net(input_ids=torch.from_numpy(base[None]).to(dev)).logits[0].float()
                lp = torch.log_softmax(lg, dim=-1)[:, aa_ids].cpu().numpy()
                for p in ps:
                    out[where[p]] = lp[p - s + 1]
    return out


def _check_seq(seq) -> str:
    if not isinstance(seq, str) or not seq or any(c not in AA for c in seq):
        raise ValueError("wild_type must be a non-empty string of the 20 standard one-letter residue codes")
    return seq


def site_logprobs(wild_type, positions=None, masked: bool = True, model: str = "esm2_650m"):
    if model not in MODELS:
        raise ValueError(f"model must be one of {list(MODELS)}")
    single = isinstance(wild_type, str)
    seqs = {"_": wild_type} if single else dict(wild_type)
    pos = {"_": positions} if single else (positions if isinstance(positions, dict) else {k: positions for k in seqs})
    for s in seqs.values():
        _check_seq(s)
    uniq = {k: (np.arange(len(s)) + 1 if pos.get(k) is None else np.unique(np.asarray(pos[k], dtype=int))) for k, s in seqs.items()}
    for k, s in seqs.items():
        if len(uniq[k]) and (uniq[k].min() < 1 or uniq[k].max() > len(s)):
            raise ValueError("positions must lie within 1..len(wild_type)")
    if not _local_ok(model) and _remote.enabled():
        res = _remote.call("proteinplm", "site_logprobs",
                           {"wild_type": seqs, "positions": {k: [int(v) for v in uniq[k]] for k in seqs},
                            "masked": bool(masked), "model": model})
        out = {k: np.asarray(res[k], dtype=float) for k in seqs}
    elif _local_ok(model):
        out = {k: _score_one(s, uniq[k] - 1, bool(masked), model).astype(float) for k, s in seqs.items()}
    else:
        raise RuntimeError("site_logprobs: neither local weights nor a remote GPU worker are available")
    for k in seqs:
        if pos.get(k) is not None:
            out[k] = out[k][np.searchsorted(uniq[k], np.asarray(pos[k], dtype=int))]
    return out["_"] if single else out


def _wt_of(wild_type, key) -> str:
    if isinstance(wild_type, dict):
        if key in wild_type:
            return wild_type[key]
        if str(key) in wild_type:
            return wild_type[str(key)]
        raise ValueError(f"wild_type has no sequence for assay {key!r}; keys: {list(wild_type)[:5]}")
    return wild_type


def plm_features(table: pd.DataFrame, wild_type, model: str = "esm2_650m") -> pd.DataFrame:
    need = {"position", "wt_aa", "mut_aa"}
    if not isinstance(table, pd.DataFrame) or not need <= set(table.columns):
        raise ValueError(f"table must be a DataFrame with columns {sorted(need)}")
    gcol = next((g for g in ("assay_id", "item") if g in table.columns), None)
    groups = table[gcol].to_numpy() if gcol else np.zeros(len(table), dtype=int)
    keys = list(pd.unique(groups))
    seqs = {str(k): _check_seq(_wt_of(wild_type, k)) for k in keys}
    pos = {str(k): table["position"].to_numpy()[groups == k].astype(int) for k in keys}
    for k in keys:
        s, p = seqs[str(k)], pos[str(k)]
        if len(p) and (p.min() < 1 or p.max() > len(s)):
            raise ValueError(f"assay {k!r}: a position lies outside the sequence")
        wt = table["wt_aa"].to_numpy()[groups == k]
        if any(s[i - 1] != w for i, w in zip(p, wt)):
            raise ValueError(f"assay {k!r}: a wt_aa disagrees with the wild-type sequence")
    lm = site_logprobs(seqs, {k: sorted(set(v.tolist())) for k, v in pos.items()}, masked=True, model=model)
    lw = site_logprobs(seqs, {k: sorted(set(v.tolist())) for k, v in pos.items()}, masked=False, model=model)
    ai = {a: i for i, a in enumerate(AA)}
    cols = {c: np.full(len(table), np.nan) for c in FEATURES}
    for k in keys:
        rows = np.flatnonzero(groups == k)
        ps = sorted(set(pos[str(k)].tolist()))
        at = {p: i for i, p in enumerate(ps)}
        M, W = lm[str(k)], lw[str(k)]
        for r in rows:
            p = int(table["position"].iat[r])
            w, m = ai[table["wt_aa"].iat[r]], ai[table["mut_aa"].iat[r]]
            i = at[p]
            row = M[i]
            pr = np.exp(row - np.logaddexp.reduce(row))
            cols["esm_llr_masked"][r] = row[m] - row[w]
            cols["esm_llr_wt"][r] = W[i][m] - W[i][w]
            cols["esm_logp_mut_masked"][r] = row[m]
            cols["esm_logp_wt_masked"][r] = row[w]
            cols["esm_entropy_masked"][r] = float(-(pr * np.log(np.clip(pr, 1e-12, None))).sum())
            cols["esm_site_mean_llr"][r] = float(np.delete(row, w).mean() - row[w])
    return pd.DataFrame(cols, index=table.index)
