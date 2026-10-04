import os, sys
os.environ['SCIENCECLAW_TASK_CACHE']='/private/tmp/claude-501/sc-scratch/sota/g2_work/f39cache'
R='/Users/admin/Library/Application Support/Claude/scratch-workspaces/b286c85c-fb50-4909-a7a4-692fbcc9e2e6/67544b4d-5c42-4571-a504-64db11cd447d/scratch-2026-09-28-749df2/scienceclaw'
sys.path.insert(0,R)
import numpy as np, pandas as pd
from scienceclaw.bench.tasks.for39_eedi import Adapter, N_MASKS, N_Q
ad=Adapter(data_root='/Users/admin/Datasets/ScienceClaw-rebuild-20260928/datasets')
data=ad._data.get(); setup=ad._setup.get()
D='/Users/admin/Datasets/ScienceClaw-rebuild-20260928/datasets/for39-eedi-task4/source_members/data/'
# independent majority from raw train CSV (excluding the adapter's dev students)
trc=pd.read_csv(D+'train_data/train_task_3_4.csv')
dev_users=set(data.train.users[setup['dev_rows']].tolist())
trc=trc[~trc.UserId.isin(dev_users)]
g=trc.groupby('QuestionId')['IsCorrect'].agg(['sum','count']); maj=pd.Series(0,index=range(948)); maj.loc[g.index]=(g['sum']>g['count']-g['sum']).astype(int).values
print('majority equals adapter majority:',bool((maj.values==setup['majority']).all()))
pub=pd.read_csv(D+'test_data/test_public_task_4_more_splits.csv'); pub['maj']=pub.QuestionId.map(maj)
rng=np.random.default_rng(0)
mask_of=setup['mask_of']
# harness episode with mixed masks vs official-style formula on the same students with their own mask
ids=[i for i in setup['pools']['id']]
for trial in range(3):
    items=list(rng.choice(ids,16,replace=False))
    ep=ad._episode('id',trial,123,items)
    T={t.name:t for t in ep.tools}
    ev=T['load_eval_inputs'].fn({},{})
    # majority y and perfect y from truth (via a closure-free route: rebuild from raw CSV)
    y_maj=np.where(ev['targets'],setup['majority'][None,:],-1)
    r=ep.evaluate(y_maj,None)
    # official-style from raw csv: per mask pooled over the students holding that mask, then mean over masks
    accs=[]
    for m in sorted({mask_of[i] for i in items}):
        us=[int(i[6:]) for i in items if mask_of[i]==m]
        d=pub[pub.UserId.isin(us)&(pub[f'IsTarget_{m}']==1)]
        accs.append(((d.maj==d.IsCorrect).sum())/len(d))
    print('trial',trial,'harness ref',round(r.metrics['reference_accuracy'],6),'independent',round(float(np.mean(accs)),6),'masks present',len({mask_of[i] for i in items}))
    # perfect / inverted predictions from raw csv
    Y=np.full((16,948),-1)
    for j,i in enumerate(items):
        u=int(i[6:]); m=mask_of[i]; d=pub[(pub.UserId==u)&(pub[f'IsTarget_{m}']==1)]
        Y[j,d.QuestionId.values]=d.IsCorrect.values
    print('   perfect ->',ep.evaluate(Y,None).primary,' inverted ->',ep.evaluate(np.where(Y>=0,1-Y,-1),None).primary,'target cells match:',bool(((Y>=0)==ev['targets']).all()))
# official aggregation vs harness aggregation, all 615 students, majority predictor
res=np.array([ (lambda m: ( (pub[f'IsTarget_{m}']==1)&(pub.maj==pub.IsCorrect)).sum()/ (pub[f'IsTarget_{m}']==1).sum())(m) for m in range(10)])
print('official-style mean over 10 masks, majority, public, all students:',round(res.mean(),4))
# duplicates between train and test
trall=pd.read_csv(D+'train_data/train_task_3_4.csv'); print('AnswerId overlap train/public:',len(set(trall.AnswerId)&set(pub.AnswerId)),' (UserId,QuestionId) overlap:',len(set(zip(trall.UserId,trall.QuestionId))&set(zip(pub.UserId,pub.QuestionId))))
# student metadata availability
st=pd.read_csv(D+'metadata/student_metadata_task_3_4.csv'); print('students in metadata',len(st),'public covered',pub.UserId.isin(st.UserId).mean())
