import sys, time; sys.path.insert(0,'/private/tmp/claude-501/sc-scratch/fix/f30_work')
from common import *
a=adapter(2); g=groups(a)
tr=g['src']['P0030692']+g['src']['P0030855']   # 19
te=g['val']['P0030688'][:8]
t=time.time(); L=load_imgs(a,tr,('images','semantics')); T=load_imgs(a,te,('images',)); print('load',time.time()-t)
t=time.time(); m=ps.fit_pixel_classifier(L['images'],L['semantics'],seed=0); print('fit',time.time()-t, m.plant_size)
t=time.time(); pr=ps.predict_probs(m,T['images'],stride=2); print('pred',time.time()-t, pr.shape, pr.dtype)
t=time.time(); out=ps.panoptic_from_probs(pr, dict(ps.DEFAULT_PARAMS, plant_size=m.plant_size)); print('pano',time.time()-t)
t=time.time(); print(score(a,te,out)); print('score',time.time()-t)
