import json,glob,collections,statistics as st,math
R=collections.defaultdict(list)
for f in sorted(glob.glob('out/curve_*_*.json')):
    d=json.load(open(f));
    if d['epochs']!=8: continue
    R[d['n_train']].append(d)
def ms(x):
    return (st.mean(x), st.stdev(x) if len(x)>1 else float('nan'))
print("n  seeds | pool: LAS(sd) UAS UPOS LAS|goldtags lab|goldhead oov")
for n in sorted(R):
    for p in ['dev','test','pud']:
        row=[ms([d['pools'][p][k] for d in R[n]]) for k in ['las','uas','upos','las_goldtags','label_acc_goldheads','oov']]
        print(n,len(R[n]),p,' '.join(f"{m:.4f}({s:.4f})" for m,s in row), f"fit_cpu~{st.mean(d['fit_cpu_s'] for d in R[n]):.0f}s words~{st.mean(d['n_train_words'] for d in R[n]):.0f}")
# log2 slope fit on test/dev/pud mean LAS
import numpy as np
for p in ['dev','test','pud']:
    xs=[];ys=[]
    for n in sorted(R):
        for d in R[n]: xs.append(math.log2(n)); ys.append(d['pools'][p]['las'])
    b,a=np.polyfit(xs,ys,1);
    # also quadratic
    c2=np.polyfit(xs,ys,2)
    print(p,'LAS ~ %.4f + %.4f*log2(n); pred n=1000 %.3f 4000 %.3f 8000 %.3f 14554 %.3f'%(a,b,a+b*math.log2(1000),a+b*math.log2(4000),a+b*math.log2(8000),a+b*math.log2(14554)),
      '| quad pred 4000 %.3f 14554 %.3f'%(np.polyval(c2,math.log2(4000)),np.polyval(c2,math.log2(14554))))
# per-word-count check
print({n:(round(np.mean([d['pools']['test']['n_words'] for d in R[n]])),) for n in R})
