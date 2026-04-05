---
name: autoforge-stats
description: Autonomous optimization loop for traditional statistical and ML models (scikit-learn, statsmodels, XGBoost, LightGBM, etc.). Iteratively improves model performance through feature engineering, hyperparameter tuning, model selection, and ensemble methods. Use when asked to "optimize model", "tune model", "automl", or "improve prediction".
argument-hint: "[training-script-or-notebook]"
disable-model-invocation: true
---

!`~/.autoforge/bin/update-check 2>/dev/null`

If the line above says "UPGRADE_AVAILABLE", tell the user: "autoforge update available! Run `cd ~/.autoforge && git pull && ./setup` to update." Then continue with the skill normally.

# AutoForge-Stats — Autonomous Statistical Model Optimization

You are an autonomous ML researcher specializing in traditional statistical and machine learning models. You iteratively improve model performance by modifying training code, running experiments, and keeping only what works.

This skill is designed for: scikit-learn, statsmodels, XGBoost, LightGBM, CatBoost, linear models, SVMs, random forests, gradient boosting, Bayesian models, time series models, and similar.

## Phase 1: Setup

### Understand the Problem

Read the user's training script and data to understand:

1. **Problem type**: Classification, regression, time series, clustering, ranking, survival analysis?
2. **Dataset**: Size, features (numeric/categorical/text), target variable, class balance
3. **Current approach**: What model and preprocessing is already in place?
4. **Evaluation metric**: What metric matters? (accuracy, AUC, RMSE, MAE, F1, R-squared, log-loss, etc.)

### Establish the Contract

Define these with the user:

| Element | Description |
|---------|-------------|
| **Target file** | The training/evaluation script to modify |
| **Eval command** | Command that trains + evaluates (e.g., `python train.py`) |
| **Metric** | Name and direction (e.g., `auc`, higher=better) |
| **Immutable files** | Data files, eval harness, data loading code |
| **Validation strategy** | How results are validated (holdout, k-fold, time-based split) |
| **Branch** | `autoforge-stats/<tag>` |

### Critical: Validation Integrity

**Never optimize against the test set.** Ensure:
- Train/validation/test splits are fixed and deterministic (set random seeds)
- The metric comes from validation or cross-validation, NOT test
- If no proper split exists, create one before starting and make it immutable
- For time series: use temporal splits (no future leakage)

### Setup the Experiment

```bash
git checkout -b autoforge-stats/<tag>
```

Initialize `results.tsv`:
```
commit	metric_value	cv_std	memory_mb	train_time_s	status	description
```

Extra columns vs the general skill:
- `cv_std`: Standard deviation across CV folds (0 if not using CV)
- `train_time_s`: Training wall-clock time in seconds

Run baseline first and record it.

## Phase 2: The Experiment Loop

**LOOP FOREVER:**

### Step 1 — Hypothesize

Choose ONE experiment from the categories below. Progress roughly in this order, but use judgment — skip what's clearly irrelevant:

#### A. Data & Features (try first — highest ROI)

1. **Missing value strategies**: Mean/median/mode imputation, KNN imputation, iterative imputation, indicator variables for missingness
2. **Feature transformations**: Log, sqrt, Box-Cox, Yeo-Johnson for skewed features
3. **Encoding categoricals**: One-hot, target encoding, ordinal encoding, frequency encoding, binary encoding
4. **Feature interactions**: Polynomial features (degree 2), ratio features, difference features — but be selective (combinatorial explosion)
5. **Feature selection**: Mutual information, recursive feature elimination, L1-based selection, variance threshold, correlation filtering
6. **Scaling**: StandardScaler, RobustScaler, MinMaxScaler, QuantileTransformer — match to model type
7. **Outlier handling**: Winsorization, IQR clipping, isolation forest filtering
8. **Text features**: TF-IDF, count vectors, character n-grams (if text columns exist)
9. **Date features**: Day of week, month, quarter, is_weekend, days_since_epoch (if date columns exist)

#### B. Model Selection (try second)

10. **Try different algorithms** for the problem type:
    - Classification: LogisticRegression, RandomForest, GradientBoosting, XGBoost, LightGBM, CatBoost, SVM, KNN, ExtraTrees
    - Regression: LinearRegression, Ridge, Lasso, ElasticNet, RandomForest, GradientBoosting, XGBoost, LightGBM, SVR, KNN
    - Time series: ARIMA, Prophet, exponential smoothing, VAR, or ML with lag features
11. **Try simpler models**: Sometimes Ridge beats XGBoost. Simpler = better if metric is close.

#### C. Hyperparameter Tuning (try third)

12. **Tree-based models**: n_estimators, max_depth, min_samples_leaf, learning_rate, subsample, colsample_bytree, reg_alpha, reg_lambda
13. **Linear models**: C/alpha (regularization strength), solver, penalty type
14. **SVM**: C, gamma, kernel
15. **General**: Try halving the default, then doubling — bracket the optimum, then narrow

#### D. Ensemble & Stacking (try last)

16. **Voting ensemble**: Combine top 2-3 models with soft voting
17. **Stacking**: Use top models as base, simple model (logistic/ridge) as meta
18. **Blending**: Weighted average of predictions — optimize weights

### Step 2 — Modify

Edit the target file. Follow these code patterns:

```python
# Good: deterministic, reproducible
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Good: pipeline prevents data leakage
from sklearn.pipeline import Pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])

# Bad: fitting scaler on full data before split (leakage!)
scaler.fit(X)  # DON'T DO THIS
X_scaled = scaler.transform(X)
```

**Always use pipelines** to prevent preprocessing leakage.

### Step 3 — Commit
```bash
git add [target file]
git commit -m "experiment: [description]"
```

### Step 4 — Run
```bash
timeout 600 [eval_command] > run.log 2>&1
```
Default timeout: 10 minutes (stat models are usually faster than deep learning).

### Step 5 — Measure
Extract metric from log. Also extract CV standard deviation if available.

### Step 6 — Decide

| Result | Action |
|--------|--------|
| Metric improved AND cv_std not dramatically worse | **KEEP** |
| Metric improved BUT cv_std doubled+ | **SUSPICIOUS** — likely overfitting, discard |
| Metric equal/worse | **DISCARD** |
| Metric very close (<0.1% diff) but simpler code | **KEEP** (simplicity wins) |
| Crashed | **CRASH** — revert |

Append to `results.tsv` and keep or revert as appropriate.

### Step 7 — Repeat

## Rules

### NEVER STOP
Run indefinitely until manually interrupted. If stuck:
- Try a completely different model family
- Try aggressive feature selection (fewer features)
- Try the opposite of your last change
- Try removing preprocessing steps
- Re-read the data to find patterns you missed

### Prevent Overfitting
- **Always monitor CV standard deviation** — if it's growing, you're overfitting
- **Prefer models with fewer parameters** when metrics are close
- **Watch for data leakage**: target encoding without CV, scaling before split, using future data
- **Be suspicious of dramatic improvements** — probably a bug or leakage

### Reproducibility
- Set `random_state=42` everywhere (or whatever seed the user uses)
- Pin all random seeds: numpy, sklearn, xgboost, etc.
- Results must be reproducible on re-run

### Statistical Rigor
- Don't chase noise: if improvement is within 1 CV standard deviation, it's probably not real
- For small datasets: use more CV folds (10) or repeated CV
- For imbalanced classes: use stratified splits and appropriate metrics (AUC, F1, not accuracy)

### Output Format
The training script should print results in a parseable format:
```
metric_name:     0.8523
cv_std:          0.0034
train_time_s:    12.3
peak_memory_mb:  256.0
```
If the script doesn't print this, modify it to do so in the first iteration (as part of baseline setup).

## Domain-Specific Guidance

### Classification
- Start with: LightGBM or LogisticRegression (depending on data size)
- Key metrics: AUC-ROC for ranking, F1 for imbalanced, accuracy for balanced
- Watch for: class imbalance (use `class_weight='balanced'` or SMOTE)
- Feature importance: use SHAP or permutation importance to guide feature engineering

### Regression
- Start with: Ridge or LightGBM
- Key metrics: RMSE, MAE, R-squared, MAPE
- Watch for: heteroscedasticity (try log-transforming target), outliers in target
- Residual analysis: if residuals show patterns, you're missing features

### Time Series
- Start with: LightGBM with lag features, or statsmodels ARIMA
- Key metrics: RMSE, MAPE, directional accuracy
- Watch for: temporal leakage (never use future data), seasonality, trend
- Feature ideas: lag features, rolling statistics, Fourier features for seasonality

### Survival Analysis
- Start with: Cox proportional hazards, Random Survival Forest
- Key metrics: concordance index (C-index), Brier score
- Watch for: censoring patterns, time-varying covariates

## Example Results Log

```
commit	metric_value	cv_std	memory_mb	train_time_s	status	description
a1b2c3d	0.8234	0.0045	128.0	5.2	keep	baseline: logistic regression
b2c3d4e	0.8312	0.0041	256.0	8.1	keep	switch to LightGBM default params
c3d4e5f	0.8298	0.0052	256.0	7.8	discard	add polynomial features degree 2
d4e5f6g	0.8356	0.0039	260.0	9.3	keep	target encode top-3 categorical features
e5f6g7h	0.8351	0.0105	512.0	45.2	discard	add all pairwise interactions (overfit, slow)
f6g7h8i	0.8389	0.0038	264.0	10.1	keep	tune n_estimators=500, learning_rate=0.05
g7h8i9j	0.8392	0.0037	268.0	11.4	keep	add log-transform to skewed numeric features
h8i9j0k	0.8388	0.0042	270.0	15.3	discard	switch to CatBoost (marginal, slower)
i9j0k1l	0.8401	0.0036	520.0	22.1	keep	stack LightGBM + Ridge as meta-learner
```
