---
name: scienceclaw-benchmark-for34
description: "Use when predicting a binary molecular property such as HIV replication inhibition (OGB ogbg-molhiv, MoleculeNet-style activity) from SMILES strings and labelled molecules, delivering one probability-like score per query molecule evaluated by ROC-AUC under a scaffold split. Corresponds to FoR34 of the companion ScienceClaw-Eval benchmark."
metadata: { "openclaw": { "emoji": "📊" } }
---

# Molecular activity classification (ogbg-molhiv)

**Task.** Input: SMILES of labelled molecules (1 = active, 0 = inactive; actives are a few percent) and SMILES of query molecules. Deliverable: a 1-D float array, one score in [0, 1] per query molecule in query order (higher = more likely active); only the ordering matters.

**Quality.** ROC-AUC, higher is better; rank-based and insensitive to prevalence. OGB splits by Bemis-Murcko scaffold, so evaluation molecules have scaffolds absent from training.

**Library** (needs RDKit; check `scienceclaw_tools(operation=show, target=molecules)`):
- `scilib.molecules.featurize(smiles, kinds=...)` returns `(X, valid, names)`; kinds `morgan_counts`, `atompair_counts`, `maccs`, `descriptors` (RDKit 2D) or `concat`; unparsable SMILES give a zero row and `valid` False.
- `scaffold_groups(smiles)` for grouped splits; `fit_predict(train, y, *queries)`: class-balanced RandomForest + ExtraTrees (`TreeEnsemble`), finite scores in [0, 1], median score for unparsable query SMILES; `grouped_cv_auc(train, y, groups="scaffold")`: out-of-fold AUC with scaffold-disjoint folds.
- Operators: `molecule_features_rdkit`, `molecule_scaffold_ids`, `molecule_activity_classifier`, `molecule_scaffold_cv_auc`.
- No pretrained molecular weights ship with the library; the route is classical featurisation plus tree ensembles.

**Routes.** Baseline: class-balanced logistic regression on a few trivial counts (atoms, bonds, rings, element fractions). Default: `molecules.fit_predict` on `concat` features. Compare feature blocks, tree counts and rank-averaged blends by pooled out-of-fold AUC from `grouped_cv_auc`.

**Rules.**
- Validate with scaffold-grouped folds; random folds overestimate AUC under a scaffold shift.
- AUC on a small query set with few actives rests on a handful of positive-negative pairs and is very noisy: do not choose models on the query set or read small differences as real.
- Never use query labels or label-derived features. Keep every query row (invalid SMILES still need a finite score) and its order.
- Output finite, within [0, 1], one value per query molecule.
