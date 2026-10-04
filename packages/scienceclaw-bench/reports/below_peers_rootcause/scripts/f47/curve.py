"""Learning-curve / ablation harness for scilib.udparse on FoR47 data (READ-ONLY use of repo + data).
usage: curve.py N SEED [epochs]   -> writes out/curve_N_SEED.json
Train: N sentences sampled from the official GSD 2.2 TRAIN file (2..80 words, seeded permutation).
Score: full pools (3..40 words, valid trees): dev (=src+val union), test (=id), PUD test (=ood). Never trains on scored sentences.
"""
import sys, os, json, time, collections
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1"); os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
REPO = "/Users/admin/Library/Application Support/Claude/scratch-workspaces/b286c85c-fb50-4909-a7a4-692fbcc9e2e6/67544b4d-5c42-4571-a504-64db11cd447d/scratch-2026-09-28-749df2/scienceclaw"
sys.path.insert(0, REPO)
import numpy as np
from scienceclaw.bench.tasks.for47_ud import parse_conllu, sentence_dict, tree_issues
from scilib import udparse as U

ROOT = "/Users/admin/Datasets/ScienceClaw-rebuild-20260928/datasets/for47-ud-conll2018/subset/ud-treebanks-v2.2"
def load(tb, f, prefix):
    return parse_conllu(open(f"{ROOT}/{tb}/{f}", encoding="utf-8").read(), prefix)

FAST = os.environ.get("FAST") == "1"
N = int(sys.argv[1]); SEED = int(sys.argv[2]); EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 8
train_all = [s for s in load("UD_French-GSD", "fr_gsd-ud-train.conllu", "tr/") if 2 <= s.n <= 80]
def ok(s): return 3 <= s.n <= 40 and not tree_issues(list(s.heads), list(s.deprels))
pools = {"dev": [s for s in load("UD_French-GSD", "fr_gsd-ud-dev.conllu", "dv/") if ok(s)],
         "test": [s for s in load("UD_French-GSD", "fr_gsd-ud-test.conllu", "te/") if ok(s)],
         "pud": [s for s in load("UD_French-PUD", "fr_pud-ud-test.conllu", "pu/") if ok(s)]}
eval_forms = {s.forms for p in pools.values() for s in p}
train_all = [s for s in train_all if s.forms not in eval_forms]
rng = np.random.default_rng(1000 + SEED)
perm = rng.permutation(len(train_all))
tr = [train_all[j] for j in perm[:N]]
train = [sentence_dict(s, True) for s in tr]
t0 = time.time()
m = U.UDParser(epochs=EPOCHS, seed=0).fit(train)
fit_s = time.time() - t0
res = {"n_train": len(tr), "n_train_words": int(sum(s.n for s in tr)), "seed": SEED, "epochs": EPOCHS, "fit_wall_s": fit_s,
       "fit_cpu_s": m.fit_seconds_, "oof_upos": m.oof_upos_accuracy_, "pools": {}}

def breakdown(gold, preds_heads, preds_rels, gold_upos, pred_upos):
    rel = collections.defaultdict(lambda: [0, 0, 0, 0])   # gold count, las correct, uas correct, predicted count with that label
    pos = collections.defaultdict(lambda: [0, 0, 0])
    for g, ph, pr, pu in zip(gold, preds_heads, preds_rels, pred_upos):
        for i in range(g.n):
            r = g.deprels[i].split(":")[0]
            uok = ph[i] == g.heads[i]
            lok = uok and pr[i].split(":")[0] == r
            rel[r][0] += 1; rel[r][1] += lok; rel[r][2] += uok
            rel[pr[i].split(":")[0]][3] += 1
            pos[g.upos[i]][0] += 1; pos[g.upos[i]][1] += lok; pos[g.upos[i]][2] += uok
    return {k: v for k, v in rel.items()}, {k: v for k, v in pos.items()}

for name, pool in pools.items():
    sents = [sentence_dict(s, False) for s in pool]
    t1 = time.time()
    tags = m.tag(sents)
    par = m.parse(sents, tags)
    parse_s = time.time() - t1
    heads = [p["head"] for p in par]; rels = [p["deprel"] for p in par]
    tot = sum(s.n for s in pool)
    las = sum(int(ph[i] == g.heads[i] and pr[i].split(":")[0] == g.deprels[i].split(":")[0]) for g, ph, pr in zip(pool, heads, rels) for i in range(g.n))
    uas = sum(int(ph[i] == g.heads[i]) for g, ph in zip(pool, heads) for i in range(g.n))
    upos = sum(int(tg[i] == g.upos[i]) for g, tg in zip(pool, tags) for i in range(g.n))
    if FAST:
        las_g = uas_g = lab_ok = lab_ok_g = float("nan")
    else:
        # oracle ablations
        gold_tags = [list(s.upos) for s in pool]
        par_g = m.parse(sents, gold_tags)
        las_g = sum(int(p["head"][i] == g.heads[i] and p["deprel"][i].split(":")[0] == g.deprels[i].split(":")[0]) for g, p in zip(pool, par_g) for i in range(g.n))
        uas_g = sum(int(p["head"][i] == g.heads[i]) for g, p in zip(pool, par_g) for i in range(g.n))
        # labeller given gold heads (with predicted tags)  -> label accuracy ceiling of the labeller
        sa = m._sents(sents, tags)
        lab_ok = 0
        for g, a in zip(pool, sa):
            lab = m.labeller_.predict(a, np.asarray(g.heads))
            lab_ok += sum(int(lab[i].split(":")[0] == g.deprels[i].split(":")[0]) for i in range(g.n))
        # labeller given gold heads AND gold tags
        sag = m._sents(sents, gold_tags)
        lab_ok_g = 0
        for g, a in zip(pool, sag):
            lab = m.labeller_.predict(a, np.asarray(g.heads))
            lab_ok_g += sum(int(lab[i].split(":")[0] == g.deprels[i].split(":")[0]) for i in range(g.n))
    # OOV rate (lowercased forms unseen in the training sample)
    vocab = {f.lower() for s in tr for f in s.forms}
    oov = sum(int(f.lower() not in vocab) for g in pool for f in g.forms) / tot
    rel, pos = breakdown(pool, heads, rels, None, tags)
    res["pools"][name] = {"n_sent": len(pool), "n_words": tot, "las": las / tot, "uas": uas / tot, "upos": upos / tot,
                          "las_goldtags": las_g / tot, "uas_goldtags": uas_g / tot, "label_acc_goldheads": lab_ok / tot,
                          "label_acc_goldheads_goldtags": lab_ok_g / tot, "oov": oov, "parse_s": parse_s, "rel": rel, "pos": pos}
    print(N, SEED, name, f"LAS {las/tot:.4f} UAS {uas/tot:.4f} UPOS {upos/tot:.4f} LAS|goldtags {las_g/tot:.4f} lab|goldhead {lab_ok/tot:.4f} oov {oov:.3f} fit {fit_s:.0f}s", flush=True)
json.dump(res, open(f"/private/tmp/claude-501/sc-scratch/fix/f47_work/out/curve_{N}_{SEED}{'_fast' if FAST else ''}{'' if EPOCHS==8 else '_e'+str(EPOCHS)}.json", "w"))
