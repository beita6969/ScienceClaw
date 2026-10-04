"""Typed operators for the protein, molecule, materials, causal-inference and clinical modules of the tool library."""
from __future__ import annotations

from typing import Any

from scienceclaw.program.specs._common import p

_VARIANTS = "variant table: columns position (1-based index into the wild-type sequence), wt_aa and mut_aa (one-letter codes)"
_LABELLED = _VARIANTS + " and DMS_score (measured fitness, higher = fitter)"
_ESM_COLUMNS = ("esm_llr_masked", "esm_llr_wt", "esm_logp_mut_masked", "esm_logp_wt_masked", "esm_entropy_masked", "esm_site_mean_llr")
_KIND = "feature kind: morgan_counts, atompair_counts, maccs, descriptors (217 RDKit 2D descriptors) or concat (all four)"

# id, tool, description, inputs, outputs, body of run(), pre, post, applicability tags
SPECS: list[dict[str, Any]] = [
    # ------------------------------------------------------------------------------------------ proteins
    dict(id="dms_variant_effect_supervised", tool="proteinfit.fit_predict",
         description=("Predict the fitness of single amino-acid substitutions of one protein from a labelled sample of its variants "
                      "(deep mutational scanning, ProteinGym-style assay): BLOSUM62 and amino-acid property features, wild-type sequence "
                      "context and position statistics of the training scores feed ridge regression, LightGBM and extra trees whose "
                      "predictions are rank-averaged. Scores order the query variants within the assay."),
         inputs={"train": p("table", _LABELLED), "wild_type": p("text", "wild-type amino-acid sequence of the assay (single-letter codes)"),
                 "query": p("table", _VARIANTS)},
         outputs={"scores": p("array", "predicted fitness rank-score per query row, higher = fitter (comparable within the assay only)",
                              shape=("n_query",), dtype="float")},
         code=("from scilib import proteinfit\n"
               "return {'scores': proteinfit.fit_predict(inputs['train'], inputs['wild_type'], inputs['query'])}"),
         pre=[{"port": "train", "check": "nonempty"}, {"port": "query", "check": "nonempty"}], post=[{"port": "scores", "check": "finite"}],
         tags=["protein", "deep mutational scanning", "variant effect", "fitness", "proteingym", "regression", "substitution"]),
    dict(id="dms_variant_effect_supervised_with_features", tool="proteinfit.fit_predict",
         description=("Supervised fitness prediction for single amino-acid substitutions of one protein (deep mutational scanning) that "
                      "adds precomputed per-variant feature columns, for example the protein-language-model columns of "
                      "protein_variant_effect_esm2, to the sequence, BLOSUM62 and training-score position features of "
                      "dms_variant_effect_supervised; ridge, LightGBM and extra trees are rank-averaged."),
         inputs={"train": p("table", _LABELLED), "wild_type": p("text", "wild-type amino-acid sequence of the assay (single-letter codes)"),
                 "query": p("table", _VARIANTS),
                 "train_features": p("table", "extra numeric feature columns, one row per row of train (same row order)"),
                 "query_features": p("table", "the same extra feature columns, one row per row of query (same row order)")},
         outputs={"scores": p("array", "predicted fitness rank-score per query row, higher = fitter (comparable within the assay only)",
                              shape=("n_query",), dtype="float")},
         code=("from scilib import proteinfit\n"
               "out = proteinfit.fit_predict(inputs['train'], inputs['wild_type'], inputs['query'], "
               "extra_train=inputs['train_features'], extra_query=inputs['query_features'])\n"
               "return {'scores': out}"),
         pre=[{"port": "train", "check": "nonempty"}, {"port": "query", "check": "nonempty"}], post=[{"port": "scores", "check": "finite"}],
         tags=["protein", "deep mutational scanning", "variant effect", "fitness", "language model", "feature fusion", "regression"]),
    dict(id="dms_cross_validated_spearman", tool="proteinfit.cross_validate",
         description=("Estimate how well the supervised variant-effect model (ridge, LightGBM, extra trees on sequence and position "
                      "features) ranks unseen substitutions of one protein: 5-fold cross-validation over the labelled variants of a "
                      "deep mutational scanning assay, Spearman rank correlation per fold; hold out random variants or whole positions."),
         inputs={"train": p("table", _LABELLED), "wild_type": p("text", "wild-type amino-acid sequence of the assay (single-letter codes)"),
                 "by": p("text", "what the folds hold out: 'variant' (random substitutions) or 'position' (whole sequence positions)")},
         outputs={"spearman": p("number", "mean Spearman rank correlation of predicted and measured fitness over the folds", unit="1"),
                  "per_fold": p("list", "Spearman correlation of each fold")},
         code=("from scilib import proteinfit\n"
               "out = proteinfit.cross_validate(inputs['train'], inputs['wild_type'], n_folds=5, by=inputs['by'])\n"
               "return {'spearman': out['spearman'], 'per_fold': out['per_fold']}"),
         pre=[{"port": "train", "check": "nonempty"}], post=[{"port": "spearman", "check": "range", "value": [-1.0, 1.0]}],
         tags=["protein", "deep mutational scanning", "cross-validation", "spearman", "model selection", "proteingym"]),
    dict(id="dms_variant_features", tool="proteinfit.variant_features",
         description=("Model inputs for each single amino-acid substitution of a protein: BLOSUM62 score, amino-acid property values and "
                      "differences, residue-class indicators, wild-type sequence context, and statistics (count, mean, median, extremes, "
                      "shrunk and kernel-weighted means) of the labelled training variants at the same and at neighbouring positions."),
         inputs={"variants": p("table", _VARIANTS), "wild_type": p("text", "wild-type amino-acid sequence (single-letter codes)"),
                 "train": p("table", _LABELLED)},
         outputs={"features": p("table", "one row per variant, numeric columns (NaN where a position has no training variant)",
                                shape=("n_variants", "n_features"))},
         code=("from scilib import proteinfit\n"
               "return {'features': proteinfit.variant_features(inputs['variants'], inputs['wild_type'], inputs['train'])}"),
         pre=[{"port": "variants", "check": "nonempty"}, {"port": "train", "check": "nonempty"}], post=[],
         tags=["protein", "deep mutational scanning", "feature engineering", "blosum62", "amino acid", "substitution"]),
    dict(id="protein_sequence_context", tool="proteinfit.sequence_context",
         description=("Per-position descriptors of a protein sequence computed from the sequence alone: wild-type residue properties "
                      "(hydropathy, volume, charge, helix/sheet/turn propensity), windowed means, helical and strand hydrophobic moments, "
                      "glycine and proline content, relative position and distance to the termini."),
         inputs={"wild_type": p("text", "amino-acid sequence (single-letter codes)")},
         outputs={"context": p("table", "one row per residue (index 1..length), numeric columns", shape=("length", "n_descriptors"))},
         code="from scilib import proteinfit\nreturn {'context': proteinfit.sequence_context(inputs['wild_type'])}",
         pre=[{"port": "wild_type", "check": "nonempty"}], post=[],
         tags=["protein", "sequence", "hydropathy", "secondary structure propensity", "feature engineering"]),
    dict(id="dms_spearman_correlation", tool="proteinfit.spearman",
         description="Spearman rank correlation between predicted and measured variant fitness, the ProteinGym deep-mutational-scanning metric (0.0 if either side is constant).",
         inputs={"predictions": p("array", "predicted fitness per variant", shape=("n",), dtype="float"),
                 "measured": p("array", "measured fitness per variant, same order", shape=("n",), dtype="float")},
         outputs={"rho": p("number", "Spearman rank correlation in [-1, 1]", unit="1")},
         code="from scilib import proteinfit\nreturn {'rho': proteinfit.spearman(inputs['predictions'], inputs['measured'])}",
         pre=[{"port": "predictions", "check": "finite"}, {"port": "measured", "check": "finite"}],
         post=[{"port": "rho", "check": "range", "value": [-1.0, 1.0]}],
         tags=["protein", "metric", "spearman", "rank correlation", "deep mutational scanning"]),
    dict(id="protein_variant_effect_esm2", tool="proteinplm.plm_features",
         description=("Zero-shot effect of single amino-acid substitutions from the pretrained ESM-2 650M protein language model: "
                      "log-likelihood ratio of the mutant against the wild-type residue (masked marginal and wild-type marginal), the two "
                      "log-probabilities, the entropy of the site and the mean log-ratio of all substitutions at the site. Higher log-ratio = "
                      "the mutation is better tolerated. CPU cost grows with sequence length (one masked forward pass per mutated position)."),
         inputs={"variants": p("table", _VARIANTS), "wild_type": p("text", "wild-type amino-acid sequence (single-letter codes)")},
         outputs={"features": p("table", "columns " + ", ".join(_ESM_COLUMNS) + ", one row per variant (natural-log units)",
                                shape=("n_variants", len(_ESM_COLUMNS)))},
         code="from scilib import proteinplm\nreturn {'features': proteinplm.plm_features(inputs['variants'], inputs['wild_type'], model='esm2_650m')}",
         pre=[{"port": "variants", "check": "nonempty"}], post=[{"port": "features", "check": "finite"}],
         tags=["protein", "variant effect", "language model", "zero-shot", "pretrained", "esm2", "deep mutational scanning"]),
    # ------------------------------------------------------------------------------------------ molecules
    dict(id="molecule_features_rdkit", tool="molecules.featurize",
         description=("RDKit features of molecules given as SMILES: Morgan (ECFP-like) atom-environment counts, atom-pair counts, the 167 MACCS "
                      "keys or all 217 RDKit 2D physico-chemical descriptors (molecular weight, logP, TPSA, ring counts). "
                      "SMILES that RDKit cannot parse give a zero row and valid = False."),
         inputs={"smiles": p("list", "list of SMILES strings"), "kind": p("text", _KIND)},
         outputs={"features": p("array", "finite feature matrix, one row per molecule", shape=("n_molecules", "n_features"), dtype="float"),
                  "valid": p("array", "True where RDKit could build the molecule", shape=("n_molecules",), dtype="bool"),
                  "names": p("list", "column names of the feature matrix")},
         code=("from scilib import molecules\n"
               "X, valid, names = molecules.featurize(inputs['smiles'], kinds=inputs['kind'])\n"
               "return {'features': X, 'valid': valid, 'names': names}"),
         pre=[{"port": "smiles", "check": "nonempty"}], post=[{"port": "features", "check": "finite"}],
         tags=["chemistry", "molecule", "smiles", "fingerprint", "descriptors", "rdkit", "morgan", "featurization"]),
    dict(id="molecule_scaffold_ids", tool="molecules.scaffold_groups",
         description=("Bemis-Murcko scaffold group of each molecule given as SMILES: molecules with the same ring-and-linker framework "
                      "share an integer id (rings-free molecules share one id, unparsable SMILES get their own); use the ids as groups for "
                      "scaffold-disjoint train/test splits."),
         inputs={"smiles": p("list", "list of SMILES strings")},
         outputs={"groups": p("array", "integer scaffold id per molecule, numbered by first appearance", shape=("n_molecules",), dtype="int")},
         code="from scilib import molecules\nreturn {'groups': molecules.scaffold_groups(inputs['smiles'])}",
         pre=[{"port": "smiles", "check": "nonempty"}], post=[{"port": "groups", "check": "finite"}],
         tags=["chemistry", "molecule", "scaffold", "bemis-murcko", "data splitting", "rdkit"]),
    dict(id="molecule_activity_classifier", tool="molecules.fit_predict",
         description=("Train a class-balanced random forest and extra-trees ensemble on RDKit features of labelled molecules (SMILES, binary "
                      "activity such as HIV inhibition, BACE inhibition or blood-brain-barrier penetration) and score new molecules; the "
                      "score is the mean predicted probability of the active class, for ROC-AUC ranking."),
         inputs={"train_smiles": p("list", "SMILES of the labelled molecules"),
                 "labels": p("array", "0/1 activity label per training molecule", shape=("n_train",), dtype="int"),
                 "query_smiles": p("list", "SMILES of the molecules to score"), "kind": p("text", _KIND)},
         outputs={"scores": p("array", "probability-like score of label 1 per query molecule (unparsable SMILES get the median)",
                              shape=("n_query",), dtype="prob")},
         code=("from scilib import molecules\n"
               "out = molecules.fit_predict(inputs['train_smiles'], inputs['labels'], inputs['query_smiles'], kinds=inputs['kind'], n_trees=100)\n"
               "return {'scores': out}"),
         pre=[{"port": "train_smiles", "check": "nonempty"}, {"port": "query_smiles", "check": "nonempty"}],
         post=[{"port": "scores", "check": "finite"}, {"port": "scores", "check": "range", "value": [0.0, 1.0]}],
         tags=["chemistry", "molecule", "classification", "roc-auc", "random forest", "activity", "drug discovery", "qsar"]),
    dict(id="molecule_scaffold_cv_auc", tool="molecules.grouped_cv_auc",
         description=("Out-of-fold ROC-AUC of the fingerprint tree-ensemble classifier on labelled molecules with 5-fold cross-validation whose "
                      "folds never split a Bemis-Murcko scaffold, an estimate of how the model generalises to new chemotypes."),
         inputs={"smiles": p("list", "SMILES of the labelled molecules"),
                 "labels": p("array", "0/1 label per molecule", shape=("n_molecules",), dtype="int"), "kind": p("text", _KIND)},
         outputs={"auc": p("number", "pooled out-of-fold ROC-AUC", unit="1"), "fold_aucs": p("list", "ROC-AUC of each fold")},
         code=("from scilib import molecules\n"
               "out = molecules.grouped_cv_auc(inputs['smiles'], inputs['labels'], groups='scaffold', n_splits=5, kinds=inputs['kind'], n_trees=60)\n"
               "return {'auc': out['auc'], 'fold_aucs': out['fold_aucs']}"),
         pre=[{"port": "smiles", "check": "nonempty"}], post=[{"port": "auc", "check": "range", "value": [0.0, 1.0]}],
         tags=["chemistry", "molecule", "cross-validation", "scaffold split", "roc-auc", "model selection"]),
    # ------------------------------------------------------------------------------------------ materials
    dict(id="phonon_feature_matrix_mlip", tool="matphonon_mlip.phonon_features",
         description=("Harmonic phonon-spectrum feature matrix of crystal structures from a pretrained universal interatomic potential "
                      "(SevenNet-l3i5 or CHGNet) with phonopy: frequency percentiles, highest-branch statistics, density-of-states peaks, "
                      "zone-centre maximum and fraction of imaginary modes, all in cm^-1; structures are used unrelaxed, rows of failed "
                      "structures are NaN. Universal potentials soften frequencies, so use the columns as regression inputs."),
         inputs={"structures": p("list", "crystal structures, each a dict with lattice (3x3 rows, angstrom), species (element symbols) and frac_coords"),
                 "model": p("text", "'sevennet' (SevenNet-l3i5) or 'chgnet' (CHGNet 0.3.0)")},
         outputs={"X": p("array", "phonon features per structure (frequencies in cm^-1, shares dimensionless); NaN for failed structures",
                         shape=("n_structures", 21), dtype="float"),
                  "names": p("list", "column names of X")},
         code=("from scilib import matphonon_mlip\n"
               "out = matphonon_mlip.phonon_features(inputs['structures'], model=inputs['model'])\n"
               "return {'X': out['X'], 'names': out['names']}"),
         pre=[{"port": "structures", "check": "nonempty"}], post=[],
         tags=["materials", "phonons", "interatomic potential", "pretrained", "sevennet", "chgnet", "crystal", "feature matrix"]),
    dict(id="phonon_frequency_regression", tool="matphonon_regression.fit_predict",
         description=("Predict positive phonon frequencies (for example the last phonon density-of-states peak of Matbench phonons) of crystals "
                      "from structure descriptors or MLIP phonon features with an ExtraTrees ensemble fitted on the log of the frequency "
                      "(three seeds averaged); returns cm^-1."),
         inputs={"X_train": p("array", "finite feature matrix of the labelled crystals", shape=("n_train", "n_features"), dtype="float"),
                 "y_train": p("array", "positive target frequencies of the labelled crystals", shape=("n_train",), unit="cm^-1", dtype="float"),
                 "X_eval": p("array", "feature matrix of the crystals to predict, same columns as X_train", shape=("n_eval", "n_features"), dtype="float")},
         outputs={"predictions": p("array", "predicted frequency per evaluation crystal, clipped to 0..5000", shape=("n_eval",), unit="cm^-1", dtype="float")},
         code=("from scilib import matphonon_regression\n"
               "return {'predictions': matphonon_regression.fit_predict(inputs['X_train'], inputs['y_train'], inputs['X_eval'])}"),
         pre=[{"port": "X_train", "check": "finite"}, {"port": "X_eval", "check": "finite"}, {"port": "y_train", "check": "range", "value": [0.0, None]}],
         post=[{"port": "predictions", "check": "finite"}],
         tags=["materials", "phonons", "regression", "extra trees", "matbench", "crystal", "frequency"]),
    dict(id="phonon_frequency_regression_boosted", tool="matphonon_regression.fit_hist_predict",
         description=("Predict positive phonon frequencies of crystals from structure descriptors or MLIP phonon features with histogram "
                      "gradient boosting on the log of the frequency (two seeds averaged); an alternative to the extra-trees regressor when "
                      "that over-smooths; returns cm^-1."),
         inputs={"X_train": p("array", "finite feature matrix of the labelled crystals", shape=("n_train", "n_features"), dtype="float"),
                 "y_train": p("array", "positive target frequencies of the labelled crystals", shape=("n_train",), unit="cm^-1", dtype="float"),
                 "X_eval": p("array", "feature matrix of the crystals to predict, same columns as X_train", shape=("n_eval", "n_features"), dtype="float")},
         outputs={"predictions": p("array", "predicted frequency per evaluation crystal", shape=("n_eval",), unit="cm^-1", dtype="float")},
         code=("from scilib import matphonon_regression\n"
               "return {'predictions': matphonon_regression.fit_hist_predict(inputs['X_train'], inputs['y_train'], inputs['X_eval'])}"),
         pre=[{"port": "X_train", "check": "finite"}, {"port": "X_eval", "check": "finite"}, {"port": "y_train", "check": "range", "value": [0.0, None]}],
         post=[{"port": "predictions", "check": "finite"}],
         tags=["materials", "phonons", "regression", "gradient boosting", "matbench", "crystal", "frequency"]),
    dict(id="phonon_regression_cv_mae", tool="matphonon_regression.cross_validate",
         description="Shuffled 5-fold cross-validated mean absolute error, in cm^-1, of the log-target ExtraTrees phonon-frequency regression on labelled crystals.",
         inputs={"X": p("array", "finite feature matrix of the labelled crystals", shape=("n_train", "n_features"), dtype="float"),
                 "y": p("array", "positive target frequencies", shape=("n_train",), unit="cm^-1", dtype="float")},
         outputs={"mae": p("number", "cross-validated mean absolute error", unit="cm^-1")},
         code=("from scilib import matphonon_regression\n"
               "return {'mae': matphonon_regression.cross_validate(inputs['X'], inputs['y'], folds=5, n_estimators=100)['mae']}"),
         pre=[{"port": "X", "check": "finite"}, {"port": "y", "check": "range", "value": [0.0, None]}],
         post=[{"port": "mae", "check": "range", "value": [0.0, None]}],
         tags=["materials", "phonons", "cross-validation", "mae", "matbench", "model selection"]),
    # ------------------------------------------------------------------------------------------ causal inference
    dict(id="causal_effect_estimate", tool="causal.get_method",
         description=("Average treatment effect of a binary treatment from observational covariates, treatment and outcome (ACIC-2016 style, "
                      "LaLonde job training): regression adjustment, outcome imputation, T-learner, X-learner, inverse-propensity weighting "
                      "or cross-fitted doubly robust AIPW with LightGBM, boosted trees, ridge or random-forest learners. Name a method such "
                      "as regression_adjustment, impute_lgbm, xlearner_lgbm, aipw_hgb3, ipw_logit or diff_means; estimand att = effect on the treated."),
         inputs={"X": p("array", "covariate matrix, standardised numeric columns", shape=("n", "p"), dtype="float"),
                 "z": p("array", "0/1 treatment indicator", shape=("n",), dtype="int"),
                 "y": p("array", "observed outcome", shape=("n",), dtype="float"),
                 "method": p("text", "estimator: diff_means, regression_adjustment, ipw_logit, ipw_hgb or <impute|tlearner|xlearner|aipw>_<ridge|hgb|hgb3|lgbm|rf|extra|ridge_hgb>"),
                 "estimand": p("text", "'att' (mean effect on treated units) or 'ate' (mean effect over all units)")},
         outputs={"tau": p("number", "estimated mean treatment effect, in the units of the outcome"),
                  "se": p("number", "standard error of the estimate (NaN when the method has none)")},
         code=("from scilib import causal\n"
               "r = causal.get_method(inputs['method'], inputs['estimand'])(inputs['X'], inputs['z'], inputs['y'])\n"
               "return {'tau': float(r.tau), 'se': float(r.se)}"),
         pre=[{"port": "X", "check": "finite"}, {"port": "y", "check": "finite"}], post=[{"port": "tau", "check": "finite"}],
         tags=["causal inference", "treatment effect", "att", "ate", "observational", "propensity", "doubly robust", "meta-learner", "acic"]),
    dict(id="causal_effect_bayesian_forest", tool="causal.bart_effect",
         description=("Bayesian estimate of the average treatment effect with posterior uncertainty from observational data: BART imputation "
                      "(Hill 2011, Bayesian additive regression trees) or Bayesian causal forests (BCF, Hahn-Murray-Carvalho) sampled "
                      "by MCMC with the stochtree package; the standard error is the posterior standard deviation. Slower than "
                      "the meta-learners (tens of seconds for thousands of units)."),
         inputs={"X": p("array", "covariate matrix, standardised numeric columns", shape=("n", "p"), dtype="float"),
                 "z": p("array", "0/1 treatment indicator", shape=("n",), dtype="int"),
                 "y": p("array", "observed outcome", shape=("n",), dtype="float"),
                 "method": p("text", "'bart' (outcome model with treatment as a covariate) or 'bcf' (separate prognostic and effect forests)"),
                 "estimand": p("text", "'att' (mean effect on treated units) or 'ate' (mean effect over all units)")},
         outputs={"tau": p("number", "posterior-mean treatment effect, in the units of the outcome"),
                  "se": p("number", "posterior standard deviation of the mean effect")},
         code=("from scilib import causal\n"
               "r = causal.get_method(inputs['method'], inputs['estimand'], burn_in=150, draws=300)(inputs['X'], inputs['z'], inputs['y'])\n"
               "return {'tau': float(r.tau), 'se': float(r.se)}"),
         pre=[{"port": "X", "check": "finite"}, {"port": "y", "check": "finite"}], post=[{"port": "tau", "check": "finite"}],
         tags=["causal inference", "treatment effect", "bart", "bcf", "bayesian", "posterior", "stochtree", "uncertainty"]),
    dict(id="causal_propensity_overlap_report", tool="causal.overlap_report",
         description=("Check overlap and covariate balance before estimating a treatment effect: cross-fitted logistic propensity-score "
                      "quantiles of treated and control units, share of treated units beyond the controls' propensity range, effective "
                      "sample size of the ATT weights and the largest standardised mean difference of any covariate."),
         inputs={"X": p("array", "covariate matrix, standardised numeric columns", shape=("n", "p"), dtype="float"),
                 "z": p("array", "0/1 treatment indicator", shape=("n",), dtype="int")},
         outputs={"report": p("dict", "n, n_treated, treated_share, propensity quantiles per arm, frac_treated_above_control_q99, ess_att_weights, max_abs_smd")},
         code="from scilib import causal\nreturn {'report': causal.overlap_report(inputs['X'], inputs['z'])}",
         pre=[{"port": "X", "check": "finite"}], post=[],
         tags=["causal inference", "propensity score", "overlap", "positivity", "covariate balance", "diagnostics"]),
    dict(id="causal_design_matrix", tool="causal.design_matrix",
         description=("Turn a mixed covariate table (numbers, strings, categories, booleans, missing values) into a standardised numeric "
                      "matrix for treatment-effect estimators: one-hot coding of non-numeric columns, median fill of missing values, constant "
                      "columns dropped, z-scored."),
         inputs={"covariates": p("table", "covariate table, one row per unit")},
         outputs={"X": p("array", "z-scored float design matrix", shape=("n", "p"), dtype="float")},
         code="from scilib import causal\nreturn {'X': causal.design_matrix(inputs['covariates'])}",
         pre=[{"port": "covariates", "check": "nonempty"}], post=[{"port": "X", "check": "finite"}],
         tags=["causal inference", "preprocessing", "one-hot", "standardise", "covariates", "design matrix"]),
    dict(id="causal_method_semi_synthetic_check", tool="causal.semi_synthetic_check",
         description=("Rank treatment-effect estimators on the visible data without hidden truth: outcome surfaces are fitted to the observed "
                      "arms, noise is resampled from residuals, so the true effect of the simulated outcomes is known; reports each "
                      "method's RMSE and bias in units of the outcome standard deviation. Favours methods close to the surface learner."),
         inputs={"X": p("array", "covariate matrix, standardised numeric columns", shape=("n", "p"), dtype="float"),
                 "z": p("array", "0/1 treatment indicator", shape=("n",), dtype="int"),
                 "y": p("array", "observed outcome", shape=("n",), dtype="float"),
                 "methods": p("list", "method names, for example ['regression_adjustment', 'impute_lgbm', 'aipw_hgb']")},
         outputs={"scores": p("table", "one row per method with rmse_sd and bias_sd (units of the outcome standard deviation)")},
         code=("from scilib import causal\n"
               "out = causal.semi_synthetic_check(inputs['X'], inputs['z'], inputs['y'], methods=tuple(inputs['methods']), n_rep=2)\n"
               "return {'scores': out.drop(columns='seconds')}"),
         pre=[{"port": "X", "check": "finite"}, {"port": "methods", "check": "nonempty"}], post=[],
         tags=["causal inference", "model selection", "simulation", "benchmark", "treatment effect", "validation"]),
    # ------------------------------------------------------------------------------------------ clinical time series
    dict(id="sepsis_normalized_utility", tool="sepsis.normalized_utility",
         description=("PhysioNet/CinC 2019 normalized clinical utility of hourly sepsis alarms: early true alarms earn up to +1 between 12 h "
                      "before and 3 h after the sepsis time (onset label + 6 h), late misses cost up to -2, false alarms -0.05 per hour; "
                      "(U_observed - U_inaction) / (U_best - U_inaction) over all stays, 1 = optimal and 0 = never alarming."),
         inputs={"labels": p("list", "one array of hourly SepsisLabel (0/1) per stay"),
                 "alarms": p("list", "one array of hourly 0/1 alarm predictions per stay, same lengths as labels")},
         outputs={"utility": p("number", "normalized utility (NaN when no stay is septic)", unit="1")},
         code="from scilib import sepsis\nreturn {'utility': sepsis.normalized_utility(inputs['labels'], inputs['alarms'])}",
         pre=[{"port": "labels", "check": "nonempty"}], post=[{"port": "utility", "check": "finite"}],
         tags=["clinical", "sepsis", "icu", "physionet", "metric", "utility", "early warning"]),
    dict(id="sepsis_hourly_features", tool="sepsis.build_features",
         description=("Causal per-hour features of ICU stays for sepsis early warning: last-observation-carried-forward vitals and labs, rolling "
                      "6/12/24 h mean, minimum, maximum and slope of the vitals, running extremes and 12 h changes of key labs, shock index, "
                      "SIRS and qSOFA-style counts, organ-dysfunction count, static variables; every row sees only the same stay's earlier rows."),
         inputs={"table": p("table", "long table, one row per patient-hour: patient_id, hour and the 40 PhysioNet/CinC 2019 variables (HR, O2Sat, Temp, SBP, MAP, DBP, Resp, ..., Age, Gender, Unit1, Unit2, HospAdmTime, ICULOS)")},
         outputs={"features": p("table", "causal feature columns, one row per input row (same row order)", shape=("n_rows", "n_features"))},
         code="from scilib import sepsis\nreturn {'features': sepsis.build_features(inputs['table'])}",
         pre=[{"port": "table", "check": "nonempty"}], post=[],
         tags=["clinical", "sepsis", "icu", "feature engineering", "time series", "vital signs", "physionet", "early warning"]),
    dict(id="sepsis_early_warning_alarms", tool="sepsis.fit_predict",
         description=("Train an hourly sepsis early-warning model on labelled ICU stays (shallow LightGBM on causal vital-sign and lab features "
                      "plus logistic regression; alarm threshold chosen to maximise the normalized clinical utility of out-of-fold alarms) and "
                      "return 0/1 alarms per hour for new stays; the alarm at hour t uses only rows up to hour t of that stay."),
         inputs={"train": p("table", "labelled long table: patient_id, hour, the 40 PhysioNet/CinC 2019 variables and SepsisLabel"),
                 "stays": p("table", "long table of the stays to score, same columns without SepsisLabel")},
         outputs={"alarms": p("list", "one 0/1 integer array per stay, in order of first appearance in stays, as long as the stay"),
                  "info": p("dict", "threshold, oof_utility (normalized utility of out-of-fold alarms on train), oof_alarm_rate, n_positive_hours")},
         code=("from scilib import sepsis\n"
               "preds, info = sepsis.fit_predict(inputs['train'], [inputs['stays']])\n"
               "return {'alarms': preds[0], 'info': info}"),
         pre=[{"port": "train", "check": "nonempty"}, {"port": "stays", "check": "nonempty"}], post=[{"port": "alarms", "check": "nonempty"}],
         tags=["clinical", "sepsis", "icu", "early warning", "lightgbm", "physionet", "prediction", "classification"]),
]
