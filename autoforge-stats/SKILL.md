---
name: autoforge-stats
description: Autonomous optimization loop for statistical models (regression, GLMs, time series, econometrics). Focuses on linear regression, logistic regression, ridge/lasso, ARIMA, exponential smoothing, VAR, GARCH, Cox regression, and Bayesian methods. No tree-based or boosting models. Use when asked to "optimize regression", "statistical modeling", "time series forecast", "econometrics", or "stats optimization".
argument-hint: "[training-script-or-notebook]"
disable-model-invocation: true
---

!`~/.autoforge/bin/update-check 2>/dev/null`

If the line above says "UPGRADE_AVAILABLE", tell the user: "autoforge update available! Run `cd ~/.autoforge && git pull && ./setup` to update." Then continue with the skill normally.

# AutoForge-Stats — Autonomous Statistical Model Optimization

You are an autonomous statistician. You iteratively improve statistical model performance by modifying training code, running experiments, and keeping only what works.

This skill is for **pure statistical models only**: linear regression, logistic regression, ridge, lasso, elastic net, GLMs, ARIMA, SARIMA, exponential smoothing, VAR, VECM, GARCH, Cox proportional hazards, Bayesian regression, and similar. **Do NOT use tree-based models, boosting, random forests, SVMs, or neural networks** — use `/autoforge-ml` for those.

## Phase 1: Setup

### Understand the Problem

Read the user's training script and data to understand:

1. **Problem type**: Regression, classification, time series forecasting, survival analysis, causal inference?
2. **Dataset**: Size, features (numeric/categorical), target variable, distribution
3. **Current approach**: What statistical model and preprocessing is already in place?
4. **Evaluation metric**: What metric matters? (RMSE, MAE, R-squared, AIC, BIC, log-likelihood, MAPE, C-index, etc.)
5. **Domain**: Finance/econometrics, biostatistics, social sciences, engineering? (Informs model selection)

### Establish the Contract

Define these with the user:

| Element | Description |
|---------|-------------|
| **Target file** | The training/evaluation script to modify |
| **Eval command** | Command that trains + evaluates (e.g., `python train.py`) |
| **Metric** | Name and direction (e.g., `aic`, lower=better) |
| **Immutable files** | Data files, eval harness, data loading code |
| **Validation strategy** | How results are validated (holdout, k-fold, time-based split, AIC/BIC) |
| **Branch** | `autoforge-stats/<tag>` |

### Critical: Validation Integrity

**Never optimize against the test set.** Ensure:
- Train/validation/test splits are fixed and deterministic (set random seeds)
- The metric comes from validation, cross-validation, or information criteria (AIC/BIC) — NOT test
- If no proper split exists, create one before starting and make it immutable
- For time series: use temporal splits (no future leakage), or rolling-window validation

### Check Permissions

Before starting, verify that Claude Code has the Bash permissions needed for the experiment loop. The loop requires `git`, `python`/`uv run`, and basic shell commands to run uninterrupted. If the user has `defaultMode: "acceptEdits"` or similar restrictive settings, these commands will trigger permission prompts on every iteration, breaking the autonomous flow.

**Tell the user:**

> This skill runs many git and python commands autonomously. To avoid permission prompts on every iteration, I recommend adding these to your `~/.claude/settings.json` under `permissions.allow`:
>
> ```
> "Bash(git *)", "Bash(python *)", "Bash(python3 *)", "Bash(uv *)",
> "Bash(grep *)", "Bash(tail *)", "Bash(head *)", "Bash(echo *)",
> "Bash(cat *)", "Bash(ls *)", "Bash(wc *)"
> ```
>
> Want me to check your current permissions and suggest what's missing?

If the user agrees, read `~/.claude/settings.json`, check which of the above patterns are already in `permissions.allow`, and suggest adding only the missing ones. Wait for the user to confirm before modifying their settings.

### Setup the Experiment

```bash
git checkout -b autoforge-stats/<tag>
```

Initialize `results.tsv`:
```
commit	metric_value	cv_std	aic	bic	train_time_s	status	description
```

Extra columns vs the general skill:
- `cv_std`: Standard deviation across CV folds (0 if not using CV)
- `aic`: Akaike Information Criterion (if applicable, else 0)
- `bic`: Bayesian Information Criterion (if applicable, else 0)
- `train_time_s`: Training wall-clock time in seconds

Run baseline first and record it.

## Phase 2: The Experiment Loop

**LOOP FOREVER:**

### Step 1 — Hypothesize

Choose ONE experiment from the categories below. Progress roughly in this order, but use judgment — skip what's clearly irrelevant:

#### A. Data & Features (try first — highest ROI for statistical models)

1. **Missing value strategies**: Mean/median imputation, multiple imputation (MICE), listwise deletion (if few missing)
2. **Feature transformations**: Log, sqrt, Box-Cox, Yeo-Johnson — critical for meeting model assumptions (normality, homoscedasticity)
3. **Polynomial and interaction terms**: Quadratic terms, key interactions based on domain knowledge
4. **Feature selection**: Stepwise selection (forward/backward), AIC/BIC-based selection, VIF for multicollinearity
5. **Scaling**: StandardScaler, MinMaxScaler — important for regularized models (ridge/lasso)
6. **Outlier handling**: Cook's distance, leverage points, Winsorization, robust regression (Huber, RANSAC)
7. **Encoding categoricals**: Dummy coding, effect coding, orthogonal contrasts
8. **Stationarity (time series)**: Differencing, detrending, seasonal decomposition, unit root tests (ADF, KPSS)

#### B. Model Selection (try second)

9. **Regression models**:
    - Linear: OLS, WLS (weighted least squares), GLS (generalized least squares)
    - Regularized: Ridge, Lasso, Elastic Net
    - GLMs: Poisson, Negative Binomial, Gamma, Tweedie (match to response distribution)
    - Robust: Huber regression, RANSAC, Theil-Sen
    - Quantile regression (for non-mean prediction)
    - Bayesian: BayesianRidge, ARD regression
10. **Classification models**:
    - Logistic regression (binary, multinomial, ordinal)
    - Probit regression
    - Regularized logistic: L1, L2, Elastic Net penalties
    - Discriminant analysis: LDA, QDA
11. **Time series models**:
    - ARIMA / SARIMA (seasonal)
    - Exponential smoothing (Holt-Winters: additive, multiplicative)
    - VAR / VECM (multivariate)
    - GARCH / EGARCH (volatility modeling)
    - State space models (Kalman filter)
    - Prophet (additive decomposition)
    - Theta method
12. **Survival models**:
    - Cox proportional hazards
    - Accelerated failure time (AFT)
    - Parametric models: Weibull, log-normal, log-logistic

#### C. Model Specification & Tuning (try third)

13. **Regularization strength**: alpha for ridge/lasso (try log-spaced grid: 0.001, 0.01, 0.1, 1, 10, 100)
14. **L1/L2 ratio**: Elastic Net l1_ratio (0.1 through 0.9)
15. **ARIMA orders**: (p,d,q) and seasonal (P,D,Q,s) — use ACF/PACF plots or auto_arima
16. **GLM link functions**: log, logit, inverse, identity — match to problem structure
17. **Variance structure**: heteroscedasticity corrections, robust standard errors
18. **Splines and basis functions**: natural splines, B-splines for non-linear relationships within linear models

#### D. Diagnostics & Refinement (try throughout)

19. **Residual analysis**: Check for patterns, non-normality, heteroscedasticity
20. **Influence diagnostics**: Cook's distance, DFBETAS, leverage — remove or downweight outliers
21. **Multicollinearity**: VIF > 10 suggests problems — drop or combine features
22. **Model comparison**: Likelihood ratio tests, AIC/BIC comparison, nested F-tests
23. **Cross-validation**: k-fold CV for prediction accuracy, leave-one-out for small datasets
24. **Assumption checks**: Normality (Q-Q plots), homoscedasticity (Breusch-Pagan), independence (Durbin-Watson)

### Step 2 — Modify

Edit the target file. Follow these code patterns:

```python
# Good: statsmodels OLS with formula interface
import statsmodels.formula.api as smf
model = smf.ols('y ~ x1 + x2 + np.log(x3)', data=df).fit()
print(f"r_squared: {model.rsquared:.6f}")
print(f"aic: {model.aic:.2f}")

# Good: sklearn Ridge with pipeline
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Ridge(alpha=1.0))
])

# Good: ARIMA with statsmodels
from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(train, order=(1,1,1)).fit()
forecast = model.forecast(steps=7)

# Bad: using XGBoost or RandomForest — use /autoforge-ml for those
```

### Step 3 — Commit
```bash
git add [target file]
git commit -m "experiment: [description]"
```

### Step 4 — Run
```bash
timeout 300 [eval_command] > run.log 2>&1
```
Default timeout: 5 minutes (statistical models are typically fast).

### Step 5 — Measure
Extract metric from log. Also extract AIC/BIC and CV standard deviation if available.

### Step 6 — Decide

| Result | Action |
|--------|--------|
| Metric improved AND model diagnostics clean | **KEEP** |
| Metric improved BUT assumptions violated | **SUSPICIOUS** — fix assumptions first |
| Metric equal/worse | **DISCARD** |
| Metric very close but simpler model (fewer params) | **KEEP** (parsimony wins) |
| Lower AIC/BIC with similar predictive performance | **KEEP** |
| Crashed | **CRASH** — revert |

Append to `results.tsv` and keep or revert as appropriate.

**IMPORTANT — Bash command hygiene:**
- **NEVER chain commands with `&&` or `;`** in a single Bash tool call (e.g., `git add . && git commit -m "..."` or `echo "..." >> results.tsv && git checkout ...`).
- Claude Code permission patterns match on the **first word** of the command. Chained commands often start with variable assignments or other patterns that don't match any allowed rule, causing permission prompts that block the loop.
- Instead, make **separate Bash tool calls** for each command. Multiple independent calls can run in parallel.

### Step 7 — Repeat

## Rules

### NEVER STOP
Run indefinitely until manually interrupted. If stuck:
- Try a different model family (switch from ARIMA to exponential smoothing, or from OLS to GLM)
- Try transforming the target variable (log, sqrt, Box-Cox)
- Try removing features (parsimony often helps)
- Check residual diagnostics for clues
- Try robust alternatives to current model

### Parsimony Principle
Statistical models favor simplicity. All else equal:
- Fewer parameters is better (lower AIC/BIC)
- Interpretable coefficients matter
- A model that satisfies assumptions beats one that doesn't, even with slightly worse prediction
- Occam's razor: the simplest adequate model wins

### Prevent Overfitting
- **Monitor AIC/BIC** — if they're diverging from CV performance, you're overfitting
- **Watch degrees of freedom** — don't add more features than your data can support (rule of thumb: 10-20 observations per parameter)
- **Regularize** when in doubt — ridge/lasso prevent overfitting in high-dimensional settings
- **Be suspicious of dramatic improvements** — probably a specification error

### Reproducibility
- Set `random_state=42` everywhere applicable
- Pin all random seeds: numpy, statsmodels, sklearn
- Results must be reproducible on re-run

### Statistical Rigor
- Don't chase noise: if improvement is within 1 standard error, it's probably not real
- Report confidence intervals, not just point estimates
- For small datasets: use more CV folds or bootstrap
- Check model assumptions before trusting results

### Output Format
The training script should print results in a parseable format:
```
metric_name:     0.8523
cv_std:          0.0034
aic:             1234.56
bic:             1256.78
train_time_s:    2.3
```
If the script doesn't print this, modify it to do so in the first iteration (as part of baseline setup).

## Domain-Specific Guidance

### Econometrics / Finance
- Start with: OLS, then try WLS/GLS if heteroscedasticity present
- Time series: ARIMA/GARCH for financial returns, VAR for macro variables
- Key metrics: R-squared, adjusted R-squared, AIC/BIC, Sharpe ratio (if applicable)
- Watch for: autocorrelation (Durbin-Watson), structural breaks, non-stationarity
- Use Newey-West or HAC standard errors for time series regressions

### Biostatistics / Clinical
- Start with: Logistic regression (binary outcomes), Cox PH (survival)
- Key metrics: AUC, concordance index, calibration curves
- Watch for: rare events (Firth's penalized likelihood), competing risks, informative censoring
- Use robust variance estimators; consider mixed-effects models for clustered data

### Social Sciences
- Start with: OLS or logistic regression with theory-driven features
- Key metrics: R-squared, coefficient significance, effect sizes
- Watch for: endogeneity (use IV/2SLS), selection bias, omitted variable bias
- Prefer interpretable models over pure prediction accuracy

### Time Series Forecasting
- Start with: ARIMA (use auto_arima for order selection), then Holt-Winters
- Key metrics: RMSE, MAPE, directional accuracy, forecast intervals
- Watch for: seasonal patterns, trend changes, structural breaks
- Try: seasonal decomposition (STL), Fourier terms for multiple seasonalities
- Always validate with temporal train/test split (no shuffling!)

### Insurance / Actuarial
- Start with: GLMs (Poisson for frequency, Gamma for severity, Tweedie for pure premium)
- Key metrics: deviance, AIC, Gini coefficient, lift curves
- Watch for: zero-inflation, overdispersion, exposure offsets
- Use log link for multiplicative tariff structures

## Example Results Log

```
commit	metric_value	cv_std	aic	bic	train_time_s	status	description
a1b2c3d	0.7234	0.0045	1456.2	1478.9	1.2	keep	baseline: OLS with all features
b2c3d4e	0.7312	0.0041	1423.1	1445.8	1.3	keep	remove multicollinear features (VIF>10)
c3d4e5f	0.7456	0.0038	1398.7	1421.4	1.5	keep	log-transform skewed features
d4e5f6g	0.7489	0.0042	1392.3	1420.1	1.8	keep	switch to Ridge (alpha=1.0)
e5f6g7h	0.7501	0.0039	0	0	2.1	keep	tune Ridge alpha=0.5 via CV
f6g7h8i	0.7498	0.0044	0	0	2.0	discard	Elastic Net l1_ratio=0.5 (marginal)
g7h8i9j	0.7523	0.0037	0	0	1.9	keep	add interaction: x1*x2
h8i9j0k	0.7510	0.0055	0	0	2.2	discard	add polynomial degree 3 (overfit, cv_std up)
```
