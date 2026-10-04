import sys, numpy as np, warnings, json
sys.path.insert(0,'/Users/admin/Library/Application Support/Claude/scratch-workspaces/b286c85c-fb50-4909-a7a4-692fbcc9e2e6/67544b4d-5c42-4571-a504-64db11cd447d/scratch-2026-09-28-749df2/scienceclaw')
from pathlib import Path
from multiprocessing import Pool
warnings.filterwarnings('ignore')
from scienceclaw.bench.tasks.for35_tourism import read_tsf, snaive
from scienceclaw.bench.tasks._forecast_common import mase
p=Path('/Users/admin/Datasets/ScienceClaw-rebuild-20260928/datasets/for35-monash-tourism-monthly/data/tourism_monthly_dataset.tsf')
S=read_tsf(p,'tm')
def fit(args):
    i=args
    s=S[i]; y=np.maximum(s.history,1e-3)
    out={}
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel
    from statsmodels.tsa.forecasting.theta import ThetaModel
    best=None
    for err,trend,damped,seas in [('add',None,False,'add'),('add','add',True,'add'),('add',None,False,'mul'),('add','add',True,'mul'),('mul',None,False,'mul'),('mul','add',True,'mul')]:
        try:
            m=ETSModel(y,error=err,trend=trend,damped_trend=damped,seasonal=seas,seasonal_periods=12,initialization_method='heuristic')
            r=m.fit(disp=False,maxiter=200)
            aicc=r.aicc
            if np.isfinite(aicc) and (best is None or aicc<best[0]):
                fc=np.asarray(r.forecast(24))
                if np.all(np.isfinite(fc)): best=(aicc,fc)
        except Exception as e:
            pass
    ets=best[1] if best else snaive(s.history,24,12)
    try:
        th=np.asarray(ThetaModel(y,period=12,deseasonalize=True,method='auto').fit().forecast(24))
    except Exception:
        th=snaive(s.history,24,12)
    return i, ets, th
if __name__=='__main__':
    with Pool(8) as pool:
        res=pool.map(fit, range(len(S)), chunksize=4)
    res.sort()
    out={'ets':[r[1].tolist() for r in res],'theta':[r[2].tolist() for r in res]}
    json.dump(out,open('f35_forecasts.json','w'))
    for name in ('ets','theta'):
        ms=[mase(S[i].target,np.array(out[name][i]),S[i].history,12) for i in range(len(S))]
        print(name,'mean MASE all366',np.mean(ms))
    print('done')
