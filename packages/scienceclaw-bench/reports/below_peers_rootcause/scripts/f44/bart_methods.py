"""Candidate BART-based ATT estimators for the FoR44 comparison (stochtree 0.4.x). Each f(X, z, y) -> float."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from scilib import causal
from stochtree import BARTModel, BCFModel


def _bart_pred(Xtr, ytr, Xte, gfr=0, mcmc=1000, trees=200, seed=0, burn=300):
    m = BARTModel()
    m.sample(X_train=Xtr, y_train=ytr, X_test=Xte, num_gfr=gfr, num_burnin=burn, num_mcmc=mcmc,
             general_params={"random_seed": seed, "num_threads": 1}, mean_forest_params={"num_trees": trees})
    return m.y_hat_test.mean(axis=1)


def impute_bart(X, z, y, gfr=0, mcmc=1000, trees=200):
    t, c = z == 1, z == 0
    mu0 = _bart_pred(X[c], y[c], X[t], gfr, mcmc, trees)
    return float(np.mean(y[t] - mu0))


def tlearner_bart(X, z, y, gfr=0, mcmc=1000, trees=200):
    t, c = z == 1, z == 0
    mu0 = _bart_pred(X[c], y[c], X[t], gfr, mcmc, trees)
    mu1 = _bart_pred(X[t], y[t], X[t], gfr, mcmc, trees)
    return float(np.mean(mu1 - mu0))


def joint_bart(X, z, y, gfr=0, mcmc=1000, trees=200):
    """Hill (2011): one BART on (x, z); counterfactual for treated = prediction at z=0."""
    t = z == 1
    Xa = np.column_stack([X, z])
    mu0 = _bart_pred(Xa, y, np.column_stack([X[t], np.zeros(t.sum())]), gfr, mcmc, trees)
    return float(np.mean(y[t] - mu0))


def joint_bart_both(X, z, y, gfr=0, mcmc=1000, trees=200):
    t = z == 1
    Xa = np.column_stack([X, z])
    mu0 = _bart_pred(Xa, y, np.column_stack([X[t], np.zeros(t.sum())]), gfr, mcmc, trees)
    mu1 = _bart_pred(Xa, y, np.column_stack([X[t], np.ones(t.sum())]), gfr, mcmc, trees)
    return float(np.mean(mu1 - mu0))


def bcf(X, z, y, gfr=0, mcmc=1000, burn=300):
    ps = causal.propensity_scores(X, z, "logit")
    t = z == 1
    m = BCFModel()
    m.sample(X_train=X, Z_train=z.astype(float), y_train=y, propensity_train=ps, X_test=X[t], Z_test=np.ones(t.sum()),
             propensity_test=ps[t], num_gfr=gfr, num_burnin=burn, num_mcmc=mcmc,
             general_params={"random_seed": 0, "num_threads": 1})
    return float(np.mean(m.tau_hat_test.mean(axis=1)))


def bcf_short(X, z, y):
    return bcf(X, z, y, 0, 500, 150)


METHODS = {"bcf_short": bcf_short, "impute_bart": impute_bart, "tlearner_bart": tlearner_bart, "joint_bart": joint_bart,
           "joint_bart_both": joint_bart_both, "bcf": bcf}
