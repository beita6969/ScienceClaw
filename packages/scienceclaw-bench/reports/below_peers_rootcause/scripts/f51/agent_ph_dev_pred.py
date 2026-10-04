def run(inputs, config):
    import numpy as np
    import warnings
    warnings.filterwarnings('ignore')
    from sklearn.model_selection import KFold, RepeatedKFold
    from sklearn.metrics import mean_absolute_error
    from sklearn.base import clone
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures, QuantileTransformer
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
    from sklearn.linear_model import RidgeCV, HuberRegressor, ElasticNetCV, LassoCV, BayesianRidge, TheilSenRegressor
    from sklearn.svm import SVR, NuSVR
    from sklearn.neighbors import KNeighborsRegressor
    try:
        from lightgbm import LGBMRegressor
        HAS_LGB=True
    except Exception:
        HAS_LGB=False
    try:
        from xgboost import XGBRegressor
        HAS_XGB=True
    except Exception:
        HAS_XGB=False

    Xc = np.asarray(inputs['X_train'], float)
    Xpc = np.asarray(inputs['X_dev'], float)
    Xph = np.asarray(inputs['X_train_ph'], float)
    Xpph = np.asarray(inputs['X_dev_ph'], float)
    y = np.asarray(inputs['y_train'], float)

    def clean_pair(A, B):
        A=A.copy(); B=B.copy()
        both = np.vstack([A,B])
        med = np.nanmedian(np.where(np.isfinite(both), both, np.nan), axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        # cap wild nonfinite only; tree models can handle broad range once imputed
        A = np.where(np.isfinite(A), A, med)
        B = np.where(np.isfinite(B), B, med)
        return A, B
    Xph_c, Xpph_c = clean_pair(Xph, Xpph)
    Xc_c, Xpc_c = clean_pair(Xc, Xpc)

    def augment_comp(C):
        B=np.asarray(C,float)
        cols=[B, np.log1p(np.maximum(B,0))]
        # known featurizer layout: mean/min/max/std for Z,mass,en,radius,group,period,outer; then nsites, nelems, vol, density, packing, nnd min/mean
        try:
            extras=[]
            for off in [0,4,8,12,16,20,24]:
                mean, mn, mx, std = B[:,off],B[:,off+1],B[:,off+2],B[:,off+3]
                extras += [mx-mn, std/(np.abs(mean)+1e-6), mx/(np.abs(mn)+1e-6)]
            mass_mean=B[:,4]; rad_mean=B[:,12]; volpa=B[:,30]; dens=B[:,31]; nnd=B[:,33]; nndm=B[:,34]
            extras += [1/np.sqrt(np.maximum(mass_mean,1e-6)), 1/np.sqrt(np.maximum(rad_mean,1e-6)),
                       1/np.sqrt(np.maximum(volpa,1e-6)), np.sqrt(np.maximum(dens,0)),
                       nndm/(nnd+1e-6)]
            cols.append(np.vstack(extras).T)
        except Exception:
            pass
        return np.hstack(cols)

    def augment_ph(P):
        B=np.asarray(P,float)
        cols=[B]
        # phonon columns: frequency statistics, dos peak features, gamma, nsites
        try:
            fmax=B[:,0]; fp99=B[:,1]; fp95=B[:,2]; fp90=B[:,3]; fmean=B[:,4]; fmed=B[:,5]; fp25=B[:,6]
            topmean=B[:,7]; topmed=B[:,8]; topp10=B[:,9]; topmin=B[:,10]
            imag=B[:,11]; fmin=B[:,12]
            dos_top01=B[:,13]; dos_top025=B[:,14]; dos_top05=B[:,15]
            dos_last01=B[:,16]; dos_last025=B[:,17]; dos_last05=B[:,18]
            gamma=B[:,19]
            extras=np.vstack([
                fmax-fp99, fp99-fp95, fp95-fp90, fmean-fmed, fmed-fp25,
                topmean-topmin, topmed-topp10, fmax-gamma,
                dos_last01-dos_top01, dos_last025-dos_top025, dos_last05-dos_top05,
                dos_last01-dos_last025, dos_last025-dos_last05,
                dos_top01-dos_top025, dos_top025-dos_top05,
                np.maximum(fmax,1)/(np.maximum(dos_last025,1)),
                np.maximum(gamma,1)/(np.maximum(fmax,1)),
                imag, np.minimum(fmin,0), np.maximum(fmin,0)
            ]).T
            cols.append(extras)
            cols.append(np.log1p(np.maximum(B,0)))
        except Exception:
            cols.append(np.log1p(np.maximum(B,0)))
        return np.hstack(cols)

    Xa = np.hstack([augment_comp(Xc_c), augment_ph(Xph_c)])
    Xpa = np.hstack([augment_comp(Xpc_c), augment_ph(Xpph_c)])
    print('Xa', Xa.shape, 'Xpa', Xpa.shape)

    models=[]
    models.append(('et_deep', ExtraTreesRegressor(n_estimators=900, random_state=20, max_features=0.65, min_samples_leaf=1, n_jobs=-1)))
    models.append(('et_leaf2', ExtraTreesRegressor(n_estimators=800, random_state=21, max_features=0.45, min_samples_leaf=2, n_jobs=-1)))
    models.append(('et_leaf3', ExtraTreesRegressor(n_estimators=700, random_state=22, max_features=0.9, min_samples_leaf=3, n_jobs=-1)))
    models.append(('rf', RandomForestRegressor(n_estimators=700, random_state=23, max_features=0.55, min_samples_leaf=1, n_jobs=-1)))
    models.append(('gbr_abs', GradientBoostingRegressor(loss='absolute_error', n_estimators=600, learning_rate=0.025, max_depth=3, subsample=0.78, random_state=24)))
    models.append(('gbr_huber', GradientBoostingRegressor(loss='huber', alpha=0.9, n_estimators=650, learning_rate=0.025, max_depth=2, subsample=0.85, random_state=25)))
    models.append(('hgb_abs', HistGradientBoostingRegressor(loss='absolute_error', max_iter=500, learning_rate=0.035, max_leaf_nodes=17, l2_regularization=0.02, random_state=26)))
    models.append(('hgb_sq', HistGradientBoostingRegressor(loss='squared_error', max_iter=450, learning_rate=0.035, max_leaf_nodes=15, l2_regularization=0.08, random_state=27)))
    models.append(('svr_rbf', make_pipeline(SimpleImputer(), StandardScaler(), SVR(C=80, gamma='scale', epsilon=8))))
    models.append(('nusvr', make_pipeline(SimpleImputer(), StandardScaler(), NuSVR(C=60, gamma='scale', nu=0.45))))
    models.append(('knn', make_pipeline(SimpleImputer(), StandardScaler(), KNeighborsRegressor(n_neighbors=5, weights='distance', p=1))))
    models.append(('ridge_poly', make_pipeline(SimpleImputer(), StandardScaler(), PolynomialFeatures(degree=2, include_bias=False), RidgeCV(alphas=np.logspace(-3,5,20)))))
    models.append(('bayes', make_pipeline(SimpleImputer(), StandardScaler(), BayesianRidge())))
    if HAS_LGB:
        models.append(('lgb_l1', LGBMRegressor(objective='mae', n_estimators=650, learning_rate=0.025, num_leaves=15, min_child_samples=5, subsample=0.85, colsample_bytree=0.75, reg_lambda=0.05, random_state=28, verbose=-1)))
        models.append(('lgb_l2', LGBMRegressor(objective='regression', n_estimators=600, learning_rate=0.025, num_leaves=11, min_child_samples=4, subsample=0.9, colsample_bytree=0.8, reg_lambda=0.1, random_state=29, verbose=-1)))
    if HAS_XGB:
        models.append(('xgb', XGBRegressor(n_estimators=500, learning_rate=0.025, max_depth=3, subsample=0.85, colsample_bytree=0.8, reg_lambda=1.0, objective='reg:squarederror', random_state=30, n_jobs=2)))

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oofs=[]; preds=[]; names=[]
    for name, model in models:
        transforms = ['raw','log']
        # phonon target roughly proportional; also try sqrt for tree/linear diversity
        if name in ['et_deep','et_leaf2','gbr_abs','hgb_abs','lgb_l1']:
            transforms += ['sqrt']
        for trf in transforms:
            oof=np.zeros(len(y)); fold_preds=[]
            for tr, va in kf.split(Xa):
                m=clone(model)
                yt=y[tr]
                if trf=='log': yt=np.log1p(yt)
                elif trf=='sqrt': yt=np.sqrt(yt)
                m.fit(Xa[tr], yt)
                pv=m.predict(Xa[va]); pp=m.predict(Xpa)
                if trf=='log': pv=np.expm1(pv); pp=np.expm1(pp)
                elif trf=='sqrt': pv=np.square(np.maximum(pv,0)); pp=np.square(np.maximum(pp,0))
                oof[va]=pv; fold_preds.append(pp)
            oof=np.clip(oof,10,5000); pred=np.clip(np.mean(fold_preds,axis=0),10,5000)
            mae=mean_absolute_error(y,oof)
            oofs.append(oof); preds.append(pred); names.append(name+'_'+trf)
            print(names[-1], 'cv', round(mae,3), 'predmean', round(float(pred.mean()),2))

    P=np.vstack(oofs).T; T=np.vstack(preds).T
    maes=np.array([mean_absolute_error(y,P[:,i]) for i in range(P.shape[1])])
    order=np.argsort(maes)
    print('best individual', names[int(order[0])], float(maes[order[0]]))
    # Greedy averaging with replacement on OOF predictions.
    chosen=[int(order[0])]
    current=P[:,chosen[0]].copy(); best=float(maes[chosen[0]])
    for it in range(120):
        best_tuple=None
        for j in range(P.shape[1]):
            cand=(len(chosen)*current + P[:,j])/(len(chosen)+1)
            mae=mean_absolute_error(y,cand)
            if best_tuple is None or mae < best_tuple[0]:
                best_tuple=(mae,j,cand)
        if best_tuple[0] < best - 1e-7:
            best,j,current=best_tuple
            chosen.append(int(j))
        else:
            break
    print('greedy cv', best, 'nchosen', len(chosen), 'chosen', [names[i] for i in chosen[:30]])
    pred=np.mean(T[:,chosen],axis=1)
    # Blend a little with robust median of top models to avoid overfitting OOF weights.
    top=order[:min(12,len(order))]
    pred2=np.median(T[:,top],axis=1)
    oof2=np.median(P[:,top],axis=1)
    for w in [0,0.15,0.25,0.35,0.5]:
        cv=mean_absolute_error(y, (1-w)*current + w*oof2)
        print('blend_w',w,'cv',cv)
    pred=np.clip(0.85*pred + 0.15*pred2,10,5000).astype(float)
    print('final pred range mean', float(pred.min()), float(pred.max()), float(pred.mean()), 'first', pred[:10])
    return {'pred': pred}
