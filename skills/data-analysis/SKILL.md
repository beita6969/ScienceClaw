---
name: data-analysis
description: Scientific data analysis including data cleaning, exploratory data analysis (EDA), statistical testing, regression, and reporting. Uses Python with pandas, scipy, statsmodels, scikit-learn. Use when user asks to analyze data, clean a dataset, run statistics, do EDA, fit a model, or process CSV/Excel files. Triggers on "analyze this data", "clean my dataset", "run regression", "EDA", "descriptive statistics", "data processing", "correlation analysis".
---

# Data Analysis

Scientific data analysis with Python. All scripts use the venv at `/Users/zhangmingda/clawd/.venv`.

## Setup

```bash
source /Users/zhangmingda/clawd/.venv/bin/activate
```

## Workflow

### 1. Data Loading

```python
import pandas as pd
import numpy as np

# CSV
df = pd.read_csv('data.csv')
# Excel
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')
# JSON
df = pd.read_json('data.json')
# Clipboard (from user paste)
# Save user's data to a temp file first, then read

# Quick inspection
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(df.dtypes)
print(df.describe())
print(f"Missing values:\n{df.isnull().sum()}")
```

### 2. Data Cleaning

```python
# Missing values
df.dropna(subset=['critical_column'])
df['col'].fillna(df['col'].median(), inplace=True)

# Duplicates
df.drop_duplicates(inplace=True)

# Outliers (IQR method)
Q1, Q3 = df['col'].quantile([0.25, 0.75])
IQR = Q3 - Q1
mask = (df['col'] >= Q1 - 1.5*IQR) & (df['col'] <= Q3 + 1.5*IQR)
df_clean = df[mask]

# Type conversion
df['date'] = pd.to_datetime(df['date'])
df['category'] = df['category'].astype('category')
```

### 3. Exploratory Data Analysis

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Distribution
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for i, col in enumerate(numeric_cols[:4]):
    ax = axes[i//2, i%2]
    sns.histplot(df[col], kde=True, ax=ax)
    ax.set_title(col)
plt.tight_layout()
plt.savefig('distributions.png', dpi=150)

# Correlation matrix
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0, fmt='.2f')
plt.savefig('correlation.png', dpi=150)

# Pairplot for key variables
sns.pairplot(df[key_cols], hue='group')
plt.savefig('pairplot.png', dpi=150)
```

### 4. Statistical Tests

Choose test based on:
- **Data type**: continuous vs categorical
- **Distribution**: normal vs non-normal (Shapiro-Wilk test)
- **Groups**: 2 vs 3+ groups
- **Pairing**: independent vs paired/repeated

| Scenario | Normal | Non-normal |
|----------|--------|------------|
| 2 independent groups | Independent t-test | Mann-Whitney U |
| 2 paired groups | Paired t-test | Wilcoxon signed-rank |
| 3+ independent groups | One-way ANOVA | Kruskal-Wallis |
| 3+ paired groups | Repeated measures ANOVA | Friedman |
| Association (continuous) | Pearson r | Spearman ρ |
| Association (categorical) | Chi-square | Fisher's exact |

```python
from scipy import stats

# Normality test
stat, p = stats.shapiro(df['col'])
print(f"Shapiro-Wilk: W={stat:.4f}, p={p:.4f}")

# t-test
t, p = stats.ttest_ind(group1, group2)
# Effect size (Cohen's d)
d = (group1.mean() - group2.mean()) / np.sqrt((group1.std()**2 + group2.std()**2) / 2)

# ANOVA
f, p = stats.f_oneway(g1, g2, g3)

# Chi-square
chi2, p, dof, expected = stats.chi2_contingency(pd.crosstab(df['a'], df['b']))

# Correlation
r, p = stats.pearsonr(df['x'], df['y'])
```

### 5. Regression

```python
import statsmodels.api as sm
import statsmodels.formula.api as smf

# OLS
model = smf.ols('y ~ x1 + x2 + C(group)', data=df).fit()
print(model.summary())

# Logistic
model = smf.logit('outcome ~ x1 + x2', data=df).fit()
print(model.summary())

# Mixed effects
model = smf.mixedlm('y ~ x1 + x2', data=df, groups=df['subject']).fit()
```

### 6. Reporting

Always report:
- Sample size (N) and any exclusions
- Descriptive statistics (M, SD or Median, IQR)
- Test statistic, degrees of freedom, p-value
- Effect size with confidence interval
- Assumptions checked (normality, homogeneity of variance)

Format: "A significant difference was found between groups, t(48) = 2.31, p = .025, Cohen's d = 0.65, 95% CI [0.08, 1.22]."

## Tips
- Always check assumptions before parametric tests
- Report effect sizes, not just p-values
- Use Bonferroni or FDR correction for multiple comparisons
- Visualize data before and after analysis
- Save all outputs as files the user can download
