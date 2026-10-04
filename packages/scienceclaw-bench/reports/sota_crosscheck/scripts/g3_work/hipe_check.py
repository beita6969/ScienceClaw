import sys, numpy as np, collections
R=sys.argv[1]; sys.path.insert(0,R)
from pathlib import Path
from scienceclaw.bench.tasks import for43_hipe as F
root=Path('/Users/admin/Datasets/ScienceClaw-rebuild-20260928/datasets')
units=F._load_units(root, 12000)
cnt=collections.Counter((u.test_set,u.file_split) for u in units.values())
print(sorted(cnt.items()))
# copy-OCR cMER-micro per test set on OFFICIAL test files (iid) / dev (dta) / test+dev overproof
res={}
for ts_set,fs in ((F.IID_TEST_SETS,('test',)),(F.OOD_TEST_SETS,('dev','test'))):
    for ts in ts_set:
        us=[u for u in units.values() if u.test_set==ts and u.file_split in fs]
        c=np.zeros(4)
        for u in us: c+=np.array(F.char_counts(u.gt,u.ocr),float)
        res[ts]=(len(us),F.mer_from_counts(c))
        print(ts,fs,'n units',len(us),'copy-OCR cMER-micro %.4f'%res[ts][1], 'chars', int(c.sum()))
pub={'icdar2017/en':0.0260,'icdar2017/fr':0.0184,'impresso-snippets/de':0.0296,'impresso-snippets/en':0.0170,'impresso-snippets/fr':0.0169,'dta19-l0/de':0.0040,'dta19-l1/de':0.0240,'dta19-l2/de':0.0546}
iid=[t for t in F.IID_TEST_SETS]
print('published no-correction per set vs ours:')
for t in pub:
    if t in res: print('  ',t,'pub',pub[t],'ours %.4f'%res[t][1],'ratio %.2f'%(res[t][1]/pub[t]))
# weighted mean over the 8 official test sets (published overall 0.0226 needs overproof? no - 8 sets)
w={t:F.official_weight(t) for t in pub}
print('weighted baseline pub over 8 sets: %.4f'%(sum(w[t]*pub[t] for t in pub)/sum(w.values())))
have=[t for t in pub if t in res]
print('ours weighted over available: %.4f'%(sum(w[t]*res[t][1] for t in have)/sum(w[t] for t in have)))
# 16-unit episode noise of the copy baseline in the IID pool: draw 16 units balanced over 5 sets
rng=np.random.default_rng(0)
byts={ts:[u for u in units.values() if u.test_set==ts and u.file_split=='test'] for ts in F.IID_TEST_SETS}
cc={u.uid:np.array(F.char_counts(u.gt,u.ocr),float) for l in byts.values() for u in l}
vals=[]
for _ in range(2000):
    sel=[]
    for ts,l in byts.items():
        k=3 if ts!='impresso-snippets/fr' else 4
        sel+= [ (ts,cc[l[i].uid]) for i in rng.choice(len(l),k,replace=False)]
    v,_=F.weighted_cmer_micro([c for _,c in sel],[t for t,_ in sel]); vals.append(v)
vals=np.array(vals); print('IID 16-unit copy-OCR episode cMER: mean %.4f sd %.4f p5 %.4f p95 %.4f'%(vals.mean(),vals.std(),*np.percentile(vals,[5,95])))
