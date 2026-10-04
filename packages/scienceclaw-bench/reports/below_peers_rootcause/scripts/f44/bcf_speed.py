import pickle, time, sys, json, numpy as np
sys.path.insert(0, ".")
from scilib import causal
from threadpoolctl import threadpool_limits
from stochtree import BCFModel
d = pickle.load(open("/private/tmp/claude-501/sc-scratch/f44/devsrc.pkl", "rb"))
X = causal.design_matrix(d["cov"]); N = 10
full = json.load(open("/private/tmp/claude-501/sc-scratch/f44/devsrc_est_long.json"))["bcf"][:N]
def rm(est): e = (np.array(est) - d["satt"][:N]) / d["sd_y"][:N]; return float(np.sqrt(np.mean(e**2)))
print("full config first10:", round(rm(full), 4), flush=True)
def run(label, burn, mcmc, th, ntr=250):
    t0 = time.time(); est = []
    for z, y in zip(d["Z"][:N], d["Y"][:N]):
        z = z.astype(int); t = z == 1
        ps = causal.propensity_scores(X, z, "logit")
        m = BCFModel()
        with threadpool_limits(limits=th):
            m.sample(X_train=X, Z_train=z.astype(float), y_train=y, propensity_train=ps, X_test=X[t], Z_test=np.ones(t.sum()),
                     propensity_test=ps[t], num_gfr=0, num_burnin=burn, num_mcmc=mcmc,
                     general_params={"random_seed": 0, "num_threads": th}, prognostic_forest_params={"num_trees": ntr})
        est.append(float(m.tau_hat_test.mean()))
    print(label, round(rm(est), 4), f"{(time.time()-t0)/N:.1f}s/dataset", flush=True)
run("th1 b300 m1000", 300, 1000, 1)
run("th2 b300 m1000", 300, 1000, 2)
run("th1 b150 m500", 150, 500, 1)
run("th2 b150 m500", 150, 500, 2)
