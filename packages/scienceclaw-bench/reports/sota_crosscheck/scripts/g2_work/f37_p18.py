exec(open("f37_base.py").read().split("# ---- harmonic")[0])
t0=np.arange(0,i19-4); r=per_rmse(X[t0],X[t0+4]); print("persist24 2018 full-year mean-of-RMSE",round(r.mean(),3),"sqrt-mean-MSE",round(np.sqrt((r**2).mean()),3))
