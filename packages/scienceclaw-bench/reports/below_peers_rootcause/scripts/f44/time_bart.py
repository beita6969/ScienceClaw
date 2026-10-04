import pickle, time, sys, numpy as np
sys.path.insert(0, ".")
from scilib import causal
from threadpoolctl import threadpool_limits
from stochtree import BARTModel
d = pickle.load(open("/private/tmp/claude-501/sc-scratch/f44/dev.pkl", "rb"))
X = causal.design_matrix(d["cov"]); print(X.shape)
z, y = d["Z"][0].astype(int), d["Y"][0]
def bart(Xtr, ytr, Xte, gfr=10, mcmc=100, trees=200, seed=0):
    m = BARTModel()
    m.sample(X_train=Xtr, y_train=ytr, X_test=Xte, num_gfr=gfr, num_burnin=0, num_mcmc=mcmc,
             general_params={"random_seed": seed, "num_threads": 1},
             mean_forest_params={"num_trees": trees})
    return m.y_hat_test.mean(axis=1)
for gfr, mcmc in [(10, 100), (20, 200)]:
    t = time.time()
    with threadpool_limits(limits=1):
        mu0 = bart(X[z == 0], y[z == 0], X[z == 1], gfr, mcmc)
    tau = float(np.mean(y[z == 1] - mu0)); print(gfr, mcmc, round(time.time() - t, 1), "s", "tau", tau, "satt", d["satt"][0], "sd", d["sd_y"][0])
