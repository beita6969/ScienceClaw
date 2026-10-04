import sys, types, re, pickle, time, numpy as np
for n in ('stempeg','musdb','musdb.audio_classes'):
    sys.modules[n]=types.ModuleType(n)
sys.modules['musdb'].__path__=[]
from museval import metrics
src=open('/Users/admin/Library/Application Support/Claude/scratch-workspaces/b286c85c-fb50-4909-a7a4-692fbcc9e2e6/67544b4d-5c42-4571-a504-64db11cd447d/scratch-2026-09-28-749df2/scienceclaw/scienceclaw/bench/tasks/for36_musdb.py').read()
a=src.index('def framewise_sdr'); b=src.index('# ====', a)
ns={'np':np,'SR':22050,'WIN_S':1.0}
exec(src[a:b],ns)
framewise_sdr=ns['framewise_sdr']; item_scores=ns['item_scores']; aggregate=ns['aggregate']
D=pickle.load(open('f36_items.pkl','rb'))
key='final_a0_main_id/FoR36-id-c8b64f24-00'
d=D[key]
mix,stems,y=d['mix'].astype(np.float64),d['stems'].astype(np.float64),d['y']
# our stem order: vocals, drums, bass, other (same in stems and y)
rows=[]
def museval_frames(ref,est):
    sdr,isr,sir,sar,perm=metrics.bss_eval(ref,est,window=22050,hop=22050,compute_permutation=False,filters_len=512,framewise_filters=False,bsseval_sources_version=False)
    return sdr
tests={}
def run(name, est_fn, items):
    t0=time.time(); ours=[]; theirs=[]
    for i in items:
        ref=stems[i]; est=est_fn(i)
        o=framewise_sdr(ref,est,22050)
        m=museval_frames(ref,est)
        ours.append(o); theirs.append(m)
    ours=np.array(ours); theirs=np.array(theirs)   # (items, nsrc, nwin)
    ok=np.isfinite(ours)&np.isfinite(theirs)
    diff=np.abs(ours-theirs)[ok]
    print(f"{name}: n_frames={ok.sum()} max|diff|={diff.max():.4f} dB mean|diff|={diff.mean():.5f}; nan_ours={np.isnan(ours).sum()} nan_museval={np.isnan(theirs).sum()}; median frame SDR ours={np.nanmedian(ours):.3f} museval={np.nanmedian(theirs):.3f}")
    return ours,theirs
items=list(range(4))
t0=time.time()
run('agent y', lambda i: y[i], items); print('t',time.time()-t0)
run('mixture-as-estimate (our reference)', lambda i: np.repeat(mix[i][None],4,axis=0), items)
run('0.001*mixture (near-silent)', lambda i: 1e-3*np.repeat(mix[i][None],4,axis=0), items)
run('0.5*agent y', lambda i: 0.5*y[i], items)
run('2*agent y', lambda i: 2*y[i], items)
run('oracle+noise (stems+0.1*stems.std noise)', lambda i: stems[i]+0.1*np.random.default_rng(0).normal(size=stems[i].shape)*stems[i].std(), items)
