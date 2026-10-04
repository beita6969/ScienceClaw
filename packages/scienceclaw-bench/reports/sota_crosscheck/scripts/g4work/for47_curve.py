import sys, time, random
from pathlib import Path
from scienceclaw.bench.tasks import for47_ud as U
from scilib.udparse import UDParser, las_uas
root=Path('/Users/admin/Datasets/ScienceClaw-rebuild-20260928/datasets/for47-ud-conll2018/subset/ud-treebanks-v2.2')
def load(tb,fn):
    return U.parse_conllu((root/tb/fn).read_text(encoding='utf-8'),'x/')
def todict(s):
    return {"words":[{"id":i+1,"form":s.forms[i],"upos":s.upos[i],"feats":s.feats[i],"head":s.heads[i],"deprel":s.deprels[i]} for i in range(s.n)]}
train=[s for s in load('UD_French-GSD','fr_gsd-ud-train.conllu') if s.n<=80]
gsd=[s for s in load('UD_French-GSD','fr_gsd-ud-test.conllu') if 3<=s.n<=40]
pud=[s for s in load('UD_French-PUD','fr_pud-ud-test.conllu') if 3<=s.n<=40]
random.seed(0)
random.shuffle(train)
random.shuffle(gsd); random.shuffle(pud)
gsd=gsd[:250]; pud=pud[:250]
print(len(train),len(gsd),len(pud),flush=True)
for n in (1000,4000):
    t=time.time()
    P=UDParser().fit([todict(s) for s in train[:n]])
    for name,ts in (('GSD',gsd),('PUD',pud)):
        parses=P.parse([{"words":[{"id":i+1,"form":s.forms[i]} for i in range(s.n)]} for s in ts])
        r=las_uas([todict(s) for s in ts],parses)
        # own scorer
        las=tot=0
        for s,p in zip(ts,parses):
            a,b=U.attachment_counts(s,list(p['head']),list(p['deprel'])); las+=a; tot+=s.n
        print(n,name,'LAS(scilib) %.4f LAS(adapter) %.4f UAS %.4f words %d  [%.0fs]'%(r['las'],las/tot,r['uas'],tot,time.time()-t),flush=True)
