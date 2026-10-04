import pickle, time, sys, json, numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "reports/below_peers_rootcause/scripts/f44")
from scilib import causal
from threadpoolctl import threadpool_limits
from stochtree import BARTModel
d = pickle.load(open("/private/tmp/claude-501/sc-scratch/f44/devsrc.pkl", "rb"))
X = causal.design_matrix(d["cov"]); N = 12
base = json.load(open("/private/tmp/claude-501/sc-scratch/f44/devsrc_est.json"))
def rm(est): e = (np.array(est) - d["satt"][:N]) / d["sd_y"][:N]; return float(np.sqrt(np.mean(e**2)))
print("ref first12: xlearner_lgbm", rm(base["xlearner_lgbm"][:N]), "impute_lgbm", rm(base["impute_lgbm"][:N]), "joint_bart100", rm(base["joint_bart"][:N]), flush=True)
def run(label, gfr, burn, mcmc, trees=200, joint=True):
    t0 = time.time(); est = []
    for z, y in zip(d["Z"][:N], d["Y"][:N]):
        z = z.astype(int); t = z == 1
        with threadpool_limits(limits=1):
            m = BARTModel()
            Xa = np.column_stack([X, z]); Xte = np.column_stack([X[t], np.zeros(t.sum())])
            m.sample(X_train=Xa, y_train=y, X_test=Xte, num_gfr=gfr, num_burnin=burn, num_mcmc=mcmc,
                     general_params={"random_seed": 0, "num_threads": 1}, mean_forest_params={"num_trees": trees})
            mu0 = m.y_hat_test.mean(axis=1)
        est.append(float(np.mean(y[t] - mu0)))
    print(label, round(rm(est), 4), f"{time.time()-t0:.0f}s", flush=True)
run("gfr0_burn100_mcmc300", 0, 100, 300)
run("gfr0_burn300_mcmc1000", 0, 300, 1000)
run("gfr20_mcmc500", 20, 0, 500)
