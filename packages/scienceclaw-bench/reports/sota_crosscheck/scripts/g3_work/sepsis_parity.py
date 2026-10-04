import sys, importlib.util, numpy as np
R=sys.argv[1]; sys.path.insert(0,R)
spec=importlib.util.spec_from_file_location("off",sys.argv[2]); off=importlib.util.module_from_spec(spec); spec.loader.exec_module(off)
from scienceclaw.bench.tasks import for42_sepsis as F
rng=np.random.default_rng(0)
mx1=mx2=0; n=0
for it in range(3000):
    L=int(rng.integers(1,120))
    lab=np.zeros(L,int)
    if rng.random()<0.4 and L>1:
        s=int(rng.integers(0,L)); lab[s:]=1
    pr=(rng.random(L)<rng.random()).astype(int)
    o=off.compute_prediction_utility(lab,pr)
    a=F.compute_prediction_utility_official(lab,pr); b=F.prediction_utility(lab,pr)
    mx1=max(mx1,abs(o-a)); mx2=max(mx2,abs(o-b)); n+=1
print('per-stay utility: max|off-port|',mx1,' max|off-vectorized|',mx2,'over',n)
# normalization parity: official evaluate_scores builds best/inaction utilities by its own code -> replicate via its logic
import inspect
src=inspect.getsource(off.evaluate_scores) if hasattr(off,'evaluate_scores') else ''
print(src[:2500])
