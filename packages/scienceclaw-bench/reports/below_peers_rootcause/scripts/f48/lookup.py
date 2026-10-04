"""Replace the three scilib.textenc calls by lookups in the precomputed table (same values as the GPU calls on the same texts)."""
import numpy as np

from scilib import textenc


def install(npz_path):
    z = np.load(npz_path, allow_pickle=True)
    tix = {t: i for i, t in enumerate(z["texts"])}
    hix = {t: j for j, t in enumerate(z["hyp_text"])}
    rr, nli, emb = z["rr"], z["nli"], z["emb"].astype(np.float32)
    hq, hp = z["hyp_emb_q"], z["hyp_emb_plain"]

    def fix(t):
        return t if t.strip() else "."

    def embed(texts, model="bge_large_en", query=False):
        if all(t in hix for t in texts):
            return (hq if query else hp)[[hix[t] for t in texts]]
        return emb[[tix[fix(t)] for t in texts]]

    def relevance(passages, queries, model="bge_reranker_large"):
        return rr[np.ix_([tix[fix(t)] for t in passages], [hix[t] for t in queries])]

    def nli_(premises, hypotheses, model="nli_deberta_v3_large"):
        return nli[np.ix_([tix[fix(t)] for t in premises], [hix[t] for t in hypotheses])]

    textenc.embed, textenc.relevance, textenc.nli = embed, relevance, nli_
    textenc.available = lambda: True
    return z
