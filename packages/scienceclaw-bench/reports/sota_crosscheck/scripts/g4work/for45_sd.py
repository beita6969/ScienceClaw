import json,numpy as np,os
from scilib import captions as C
from sacrebleu.metrics import CHRF
off=CHRF(word_order=2)
root='/Users/admin/Datasets/ScienceClaw-rebuild-20260928/datasets/for45-americasnlp-2026/'
allsc={}
for lang in ['bribri','guarani','maya','nahuatl','wixarika']:
    for d in [root+'reconstructed_v1/upstream/data/dev/',root+'upstream/data/dev/']:
        p=d+f'{lang}/{lang}.jsonl'
        if os.path.exists(p): break
    caps=[json.loads(l)['target_caption'].strip() for l in open(p)]
    caps=[c for c in caps if c]
    # smaller-train regime: train on 20 random captions, test on the rest, 5 repeats
    rng=np.random.default_rng(1); sc=[]
    for r in range(4):
        idx=rng.permutation(len(caps)); tr=[caps[i] for i in idx[:20]]; te=[caps[i] for i in idx[20:]]
        h=C.consensus_caption(tr)
        sc+= [off.sentence_score(h,[t]).score for t in te]
    allsc[lang]=(float(np.mean(sc)),float(np.std(sc)))
    print(lang,allsc[lang],flush=True)
print('pooled SD', np.mean([v[1] for v in allsc.values()]), 'mean', np.mean([v[0] for v in allsc.values()]))
