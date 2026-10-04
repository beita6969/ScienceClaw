"""Zero-shot mAP of each pretrained score on the annotated pairs of the train partition (no fitting)."""
import sys

import numpy as np

import common as K
import lookup

z = lookup.install(sys.argv[1])
data = K.load()
tix = {t: i for i, t in enumerate(z["texts"])}
hix = {k: j for j, k in enumerate(z["hyp_keys"])}
rr, nli, emb, hq = z["rr"], z["nli"], z["emb"].astype(np.float32), z["hyp_emb_q"]
for pool in ("iid", "ood"):
    ids = data.train_docs[pool]
    res = {n: [] for n in ("overlap", "rr", "nli_ent", "nli_ev", "emb_cos", "rr_rank+cos")}
    for d, k in K.pairs_of(data, ids):
        doc = data.docs[d]
        rows = [tix[t if t.strip() else "."] for t in doc.span_texts()]
        g = K.gold(data, d, k)
        j = hix[k]
        sc = {"overlap": A_ov if (A_ov := K.A.overlap_scores(data.hypotheses[k]["hypothesis"], doc.span_texts())) is not None else None,
              "rr": rr[rows, j], "nli_ent": nli[rows, j, 1], "nli_ev": np.logaddexp(nli[rows, j, 0], nli[rows, j, 1]),
              "emb_cos": emb[rows] @ hq[j]}
        sc["rr_rank+cos"] = sc["rr"] / (sc["rr"].std() + 1e-6) + sc["emb_cos"] / (sc["emb_cos"].std() + 1e-6)
        for n, v in sc.items():
            res[n].append(K.C.average_precision(g, v))
    print(pool, len(res["rr"]), "pairs:", {n: round(float(np.mean(v)), 4) for n, v in res.items()}, flush=True)
