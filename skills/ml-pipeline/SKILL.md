---
name: ml-pipeline
description: Machine learning pipeline for scientific research including data preprocessing, feature engineering, model selection, training, evaluation, and interpretation. Covers supervised/unsupervised learning, deep learning, cross-validation, hyperparameter tuning, and model explainability. Use when user asks to build a predictive model, classify data, cluster samples, do feature selection, or apply ML to research data. Triggers on "machine learning", "classification", "clustering", "random forest", "neural network", "deep learning", "predict", "feature selection", "cross-validation", "train model".
---

# ML Pipeline

Machine learning for scientific research. Venv: `source /Users/zhangmingda/clawd/.venv/bin/activate`

## Pipeline Overview

```
Data → Clean → Features → Split → Train → Evaluate → Interpret → Report
```

## Model Selection Guide

| Task | Data Size | Interpretability Need | Recommended |
|------|-----------|----------------------|-------------|
| Classification (small) | < 10K | High | Logistic Regression, Decision Tree |
| Classification (medium) | 10K-100K | Medium | Random Forest, XGBoost |
| Classification (large) | > 100K | Low OK | Neural Network, XGBoost |
| Regression (linear) | Any | High | Linear/Ridge/Lasso |
| Regression (nonlinear) | Medium+ | Medium | Random Forest, Gradient Boosting |
| Clustering | Any | Medium | K-Means, DBSCAN, Hierarchical |
| Dimensionality reduction | Any | Medium | PCA, t-SNE, UMAP |
| Anomaly detection | Any | Medium | Isolation Forest, LOF |
| Time series | Any | Varies | ARIMA, Prophet, LSTM |

## Standard Pipeline

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# 1. Preprocessing
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. Pipeline with scaling
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(random_state=42))
])

# 3. Cross-validation
scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='roc_auc')
print(f"CV AUC: {scores.mean():.3f} ± {scores.std():.3f}")

# 4. Hyperparameter tuning
param_grid = {
    'model__n_estimators': [100, 300, 500],
    'model__max_depth': [5, 10, None],
    'model__min_samples_leaf': [1, 5, 10]
}
grid = GridSearchCV(pipe, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid.fit(X_train, y_train)

# 5. Evaluation
y_pred = grid.predict(X_test)
print(classification_report(y_test, y_pred))
print(f"Test AUC: {roc_auc_score(y_test, grid.predict_proba(X_test)[:,1]):.3f}")
```

## Feature Importance & Explainability

```python
# Built-in importance (tree models)
importances = grid.best_estimator_.named_steps['model'].feature_importances_
feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)

# SHAP values (model-agnostic)
# pip install shap
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
```

## Unsupervised Learning

```python
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# PCA
pca = PCA(n_components=0.95)  # retain 95% variance
X_pca = pca.fit_transform(X_scaled)
print(f"Components: {pca.n_components_}, Explained variance: {pca.explained_variance_ratio_.cumsum()[-1]:.3f}")

# K-Means with elbow method
inertias = [KMeans(n_clusters=k, random_state=42).fit(X_scaled).inertia_ for k in range(2, 11)]

# t-SNE visualization
X_tsne = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(X_scaled)
```

## Reporting ML Results in Papers

Always include:
1. Dataset description (size, features, class balance)
2. Preprocessing steps
3. Model selection rationale
4. Cross-validation strategy (k-fold, stratified, leave-one-out)
5. Hyperparameter search space and method
6. Multiple metrics (accuracy, precision, recall, F1, AUC)
7. Comparison with baselines
8. Feature importance / model interpretation
9. Confidence intervals or statistical tests on performance
10. Code/data availability statement

## Tips
- Always use stratified splits for imbalanced data
- Report multiple metrics, not just accuracy
- Compare against simple baselines (majority class, mean prediction)
- Use nested CV for unbiased performance estimation
- Check for data leakage (especially with time series)
- Document random seeds for reproducibility
