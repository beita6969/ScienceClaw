---
name: scikit-learn-ml
description: "Machine learning with scikit-learn. Use when: classification, regression, clustering, dimensionality reduction, model evaluation, feature engineering. NOT for: deep learning (use transformers/pytorch), time series forecasting (use statsmodels), big data (use spark)."
metadata: { "openclaw": { "emoji": "🤖", "requires": { "bins": ["python3"] }, "install": [{ "id": "uv-scikit-learn", "kind": "uv", "package": "scikit-learn" }, { "id": "uv-pandas", "kind": "uv", "package": "pandas" }, { "id": "uv-numpy", "kind": "uv", "package": "numpy" }] } }
---

# Scikit-Learn Machine Learning

Classification, regression, clustering, dimensionality reduction, and model evaluation.

## When to Use / When NOT to Use

**Use when:** classification, regression, clustering, dimensionality reduction, model evaluation, feature engineering, hyperparameter tuning, pipeline construction.

**NOT for:** deep learning (use transformers/pytorch), time series forecasting (use statsmodels), big data that doesn't fit in memory (use spark), GPU-accelerated training.

## Data Preparation and Splitting

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

df = pd.read_csv('data.csv')
X = df.drop(columns=['target'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()                      # zero mean, unit variance
X_train_scaled = scaler.fit_transform(X_train) # fit on train only
X_test_scaled = scaler.transform(X_test)       # transform test with train stats

le = LabelEncoder()
y_encoded = le.fit_transform(y)                # string labels to integers
```

## Classification

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
y_proba = clf.predict_proba(X_test_scaled)[:, 1]
print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")

# Alternatives
gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1)
svc = SVC(kernel='rbf', probability=True)
```

## Regression

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

reg = Ridge(alpha=1.0)
reg.fit(X_train_scaled, y_train)
y_pred = reg.predict(X_test_scaled)
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"R2:   {r2_score(y_test, y_pred):.4f}")

# ElasticNet combines L1+L2: ElasticNet(alpha=1.0, l1_ratio=0.5)
```

## Clustering

```python
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score

km = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = km.fit_predict(X_scaled)
print(f"Silhouette: {silhouette_score(X_scaled, labels):.4f}")

db = DBSCAN(eps=0.5, min_samples=5)           # density-based, no k needed
agg = AgglomerativeClustering(n_clusters=3)    # hierarchical
```

## Dimensionality Reduction

```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(f"Explained variance: {pca.explained_variance_ratio_.sum():.2%}")

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)         # for visualization only
```

## Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

param_grid = {'n_estimators': [100, 200], 'max_depth': [5, 10, None]}
grid = GridSearchCV(RandomForestClassifier(random_state=42),
                    param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
grid.fit(X_train_scaled, y_train)
print(f"Best params: {grid.best_params_}")
print(f"Best score:  {grid.best_score_:.4f}")

# Cross-validation shortcut
scores = cross_val_score(clf, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"CV Accuracy: {scores.mean():.4f} +/- {scores.std():.4f}")
```

## Best Practices

1. Always split data before any preprocessing; fit scalers on train set only.
2. Use `stratify=y` in `train_test_split` for imbalanced classification.
3. Set `random_state` for reproducibility in models, splits, and clustering.
4. Use `cross_val_score` or `GridSearchCV` instead of single train/test evaluation.
5. Check `feature_importances_` (tree models) or `coef_` (linear) for interpretability.
6. Use `n_jobs=-1` to parallelize grid search and ensemble models.
7. For high-dimensional sparse data, prefer `LinearSVC` or `SGDClassifier` over kernel SVM.
