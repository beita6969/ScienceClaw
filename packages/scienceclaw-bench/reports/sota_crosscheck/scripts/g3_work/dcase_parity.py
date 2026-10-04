import sys, numpy as np, sys as _s
sys.path.insert(0, sys.argv[1])
from scienceclaw.bench.tasks.for40_dcase import official_scores
from sklearn import metrics
import scipy.stats
rng=np.random.default_rng(0)
# reimplementation of the official evaluator's per-section logic (as fetched from nttcslab/dcase2024_task2_evaluator)
def official_ref(y_true, y_dom, y_pred):
    out=[]
    for d in (0,1):   # 0=source 1=target
        m=(y_dom==d)|(y_true!=0)
        out.append(metrics.roc_auc_score(y_true[m], y_pred[m]))
    out.append(metrics.roc_auc_score(y_true,y_pred,max_fpr=0.1))
    return out
mx=0
for t in range(200):
    n=200
    y=np.r_[np.zeros(100),np.ones(100)].astype(int)
    dom=np.r_[rng.permutation(np.r_[np.zeros(50),np.ones(50)]),rng.permutation(np.r_[np.zeros(50),np.ones(50)])].astype(int)
    s=rng.normal(size=n)+0.6*y+0.3*dom*(1-y)
    ref=official_ref(y,dom,s)
    ours=official_scores(y,np.where(dom==0,'source','target'),s)
    mx=max(mx,abs(ref[0]-ours['auc_source']),abs(ref[1]-ours['auc_target']),abs(ref[2]-ours['pauc']))
print('max abs diff vs official-logic on full-size sections:',mx)
# 16-clip slice noise: 4 per stratum (src-normal, src-anom, tgt-normal, tgt-anom); detector with fixed d'
def slice_scores(dprime, tshift=0.0):
    y=np.r_[np.zeros(8),np.ones(8)].astype(int)
    dom=np.array(['source']*4+['target']*4+['source']*4+['target']*4)
    s=rng.normal(size=16)+dprime*y+tshift*(dom=='target')*(1-y)
    return official_scores(y,dom,s)
for dp in (0.0,0.5,1.0,1.5):
    from scipy.stats import hmean
    v=np.array([slice_scores(dp)['official_score'] for _ in range(4000)])
    # true population value from a big sample
    big=np.mean([hmean([official_scores(*(lambda y,dom,s:(y,dom,s))(np.r_[np.zeros(500),np.ones(500)].astype(int),np.array((['source']*250+['target']*250)*2),rng.normal(size=1000)+dp*np.r_[np.zeros(500),np.ones(500)]))[k] for k in ('auc_source','auc_target','pauc')]) for _ in range(5)])
    print(f"d'={dp}: 16-clip hmean mean={v.mean():.3f} sd={v.std():.3f} p5-p95=[{np.percentile(v,5):.3f},{np.percentile(v,95):.3f}]  large-sample hmean={big:.3f}")
