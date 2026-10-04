def run(inputs: dict, config: dict) -> dict:
    import numpy as np
    import warnings
    warnings.filterwarnings('ignore')

    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import RobustScaler, StandardScaler
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
    from sklearn.linear_model import RidgeCV, HuberRegressor, ElasticNetCV, LinearRegression
    from sklearn.svm import SVR
    from sklearn.compose import TransformedTargetRegressor

    Xb = np.asarray(inputs['X_train_base'], dtype=float)
    Xbe = np.asarray(inputs['X_eval_base'], dtype=float)
    Xm = np.asarray(inputs['X_train_mlip'], dtype=float)
    Xme = np.asarray(inputs['X_eval_mlip'], dtype=float)
    y = np.asarray(inputs['y_train'], dtype=float)

    def safe(A):
        A = np.asarray(A, dtype=float)
        return np.where(np.isfinite(A), A, np.nan)
    Xb, Xbe, Xm, Xme = map(safe, (Xb, Xbe, Xm, Xme))

    eps = 1e-6
    def div(a, b):
        return a / np.where(np.abs(b) > eps, b, np.nan)

    def augment(B, M):
        cols = []
        cols.append(B)
        cols.append(M)
        Mp = np.where(np.isfinite(M), np.maximum(M, 0.0), np.nan)
        cols.append(np.log1p(Mp))
        cols.append(np.sqrt(Mp))
        if M.shape[1] >= 21:
            fmax = M[:, [0]]
            gamma = M[:, [19]]
            topmean = M[:, [7]]
            nsites = M[:, [20]]
            # physically meaningful ratios: calibrated MLIP frequency scales and DOS peak positions
            for j in [1,2,3,4,5,7,8,9,10,13,14,15,16,17,18,19]:
                cols.append(div(M[:, [j]], fmax))
                cols.append(div(M[:, [j]], gamma))
                cols.append(div(M[:, [j]], topmean))
            # dispersion / smearing differences
            for a,b in [(16,17),(17,18),(16,18),(13,16),(14,17),(15,18),
                        (0,16),(0,17),(0,18),(19,16),(7,16),(1,16),(2,17),
                        (3,18),(0,19),(0,7),(7,10),(1,3),(3,5),(13,14),(14,15)]:
                cols.append(M[:, [a]] - M[:, [b]])
            cols.append(np.log1p(np.maximum(nsites, 0)))
            # composition/geometry-dependent softening corrections scaled by upper phonon scales
            base_idx = [0,1,2,3,4,5,6,7,8,9,10,11,12,16,17,18,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34]
            for bi in base_idx:
                z = B[:, [bi]]
                cols.append(div(z, fmax))
                cols.append(z * div(M[:, [16]], fmax))
                cols.append(z * div(M[:, [17]], gamma))
        X = np.hstack(cols)
        X = np.where(np.isfinite(X), X, np.nan)
        return X

    X = augment(Xb, Xm)
    Xe = augment(Xbe, Xme)
    print('augmented shapes', X.shape, Xe.shape)

    def make_pipe(est, scale=False):
        steps = [SimpleImputer(strategy='median'), VarianceThreshold(threshold=0.0)]
        if scale:
            steps.append(RobustScaler(quantile_range=(10, 90)))
        steps.append(est)
        return make_pipeline(*steps)

    models = []
    # Tree models capture nonlinear chemistry/geometry corrections to the MLIP phonon estimates.
    models.append(('et1', make_pipe(ExtraTreesRegressor(n_estimators=700, random_state=101, max_features=0.80, min_samples_leaf=1, n_jobs=-1), False), 1.35))
    models.append(('et2', make_pipe(ExtraTreesRegressor(n_estimators=650, random_state=102, max_features=0.55, min_samples_leaf=2, n_jobs=-1), False), 1.10))
    models.append(('rf', make_pipe(RandomForestRegressor(n_estimators=450, random_state=103, max_features=0.65, min_samples_leaf=2, n_jobs=-1), False), 0.75))
    models.append(('gbr_l1', make_pipe(GradientBoostingRegressor(loss='absolute_error', n_estimators=450, learning_rate=0.030, max_depth=3, subsample=0.78, random_state=104), False), 0.80))
    models.append(('hgb_l1', make_pipe(HistGradientBoostingRegressor(loss='absolute_error', max_iter=320, learning_rate=0.040, l2_regularization=0.035, max_leaf_nodes=23, random_state=105), False), 0.80))
    # Smooth calibrated models stabilize extrapolation for the small evaluation set.
    models.append(('ridge', make_pipe(RidgeCV(alphas=np.logspace(-3, 5, 25)), True), 0.75))
    models.append(('log_ridge', TransformedTargetRegressor(regressor=make_pipe(RidgeCV(alphas=np.logspace(-3, 5, 25)), True), func=np.log1p, inverse_func=np.expm1), 0.65))
    models.append(('huber', make_pipe(HuberRegressor(alpha=2e-4, epsilon=1.30, max_iter=400), True), 0.35))
    models.append(('svr', make_pipe(SVR(C=18.0, epsilon=18.0, gamma='scale'), True), 0.35))
    # Direct linear calibration on only the raw MLIP/descriptor features; useful if engineered ratios overfit.
    Xraw = np.hstack([Xb, Xm, np.log1p(np.maximum(Xm, 0))])
    Xeraw = np.hstack([Xbe, Xme, np.log1p(np.maximum(Xme, 0))])
    Xraw = np.where(np.isfinite(Xraw), Xraw, np.nan)
    Xeraw = np.where(np.isfinite(Xeraw), Xeraw, np.nan)

    preds = []
    weights = []
    for name, model, wt in models:
        try:
            model.fit(X, y)
            p = np.asarray(model.predict(Xe), dtype=float)
            if np.all(np.isfinite(p)):
                p = np.clip(p, 10.0, 5000.0)
                preds.append(p); weights.append(float(wt))
                print(name, 'pred range', float(np.min(p)), float(np.max(p)), 'mean', float(np.mean(p)))
            else:
                print(name, 'nonfinite prediction skipped')
        except Exception as e:
            print(name, 'failed', repr(e))

    for name, model, wt in [
        ('raw_ridge', make_pipe(RidgeCV(alphas=np.logspace(-3,5,25)), True), 0.45),
        ('raw_et', make_pipe(ExtraTreesRegressor(n_estimators=450, random_state=120, max_features=0.75, min_samples_leaf=1, n_jobs=-1), False), 0.55),
        ('raw_hgb', make_pipe(HistGradientBoostingRegressor(loss='squared_error', max_iter=260, learning_rate=0.045, l2_regularization=0.04, max_leaf_nodes=25, random_state=121), False), 0.45),
    ]:
        try:
            model.fit(Xraw, y)
            p = np.asarray(model.predict(Xeraw), dtype=float)
            if np.all(np.isfinite(p)):
                p = np.clip(p, 10.0, 5000.0)
                preds.append(p); weights.append(float(wt))
                print(name, 'pred range', float(np.min(p)), float(np.max(p)), 'mean', float(np.mean(p)))
        except Exception as e:
            print(name, 'failed', repr(e))

    if not preds:
        # Very conservative fallback: calibrated median of MLIP last-DOS-peak feature if all regressors fail.
        if Xm.shape[1] >= 18 and np.any(np.isfinite(Xm[:,16])):
            tr = Xm[:,16]
            scale = np.nanmedian(y) / max(np.nanmedian(np.maximum(tr, 1.0)), 1.0)
            p = scale * Xme[:,16]
            p = np.where(np.isfinite(p), p, np.nanmedian(y))
        else:
            p = np.full(len(Xme), np.median(y))
        pred = np.clip(p, 10.0, 5000.0)
    else:
        P = np.vstack(preds)
        w = np.asarray(weights, dtype=float)
        w = w / np.sum(w)
        mean_pred = np.average(P, axis=0, weights=w)
        med_pred = np.median(P, axis=0)
        # Blend weighted mean and median to reduce sensitivity to any unstable nonlinear model.
        pred = 0.72 * mean_pred + 0.28 * med_pred
        pred = np.clip(pred, 10.0, 5000.0)

    print('FINAL eval predictions', [float(x) for x in pred])
    return {'y': np.asarray(pred, dtype=float)}
