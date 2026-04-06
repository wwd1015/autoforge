# S&P 500 7-Day Forecast — autoforge-stats example

Autonomous optimization of an S&P 500 directional prediction model using `/autoforge-stats`. The agent iteratively improves model performance by modifying `sp500_train.py` — trying different models, features, hyperparameters, and ensembles — while keeping only changes that improve the score.

## What it does

Predicts, for each of the next 7 trading days:
- **Direction**: will the S&P 500 close higher or lower than the previous day?
- **% change**: cumulative percentage move from day 0

The metric is `combined_score = directional_accuracy - 0.01 * mean_absolute_error`. Higher is better. Random baseline is ~0.50.

## Prerequisites

1. **autoforge** installed ([instructions](../../README.md))
2. **Python 3.11+** with [uv](https://docs.astral.sh/uv/)
3. **Claude Code** CLI

## Setup (5 minutes)

### 1. Copy the example into its own project directory

```bash
cp -r ~/.autoforge/examples/sp500-forecast ~/sp500-forecast
cd ~/sp500-forecast
```

### 2. Initialize git and install dependencies

```bash
git init
git add -A
git commit -m "initial: baseline GradientBoosting model"
uv sync
```

### 3. Verify the baseline runs

```bash
uv run sp500_train.py
```

First run downloads ~10 years of market data from Yahoo Finance (cached for 12 hours). You should see output like:

```
Fetching data...
  S&P 500: downloading from Yahoo Finance...
  VIX: downloading from Yahoo Finance...
  ...

Dataset ready:
  Features:    64 columns
  Train:       2182 days (2016-xx-xx to 2025-xx-xx)
  Test:        126 days (2025-xx-xx to 2025-xx-xx)
  Forecast:    7 days ahead

Selecting top features...
  Selected 35 features from 64

Training direction models...
  Day 1... train_acc=0.6xxx
  ...

---
combined_score:         0.5xxxxx
avg_direction_accuracy: 0.5xxxxx
avg_pct_mae:            1.xxxx
---
train_time_s:           xx.x
```

Note the `combined_score` — this is your baseline. It should be somewhere around 0.52-0.54 with the default GradientBoosting setup.

### 4. Configure Claude Code permissions (recommended)

Create `.claude/settings.local.json` to auto-approve safe operations:

```bash
mkdir -p .claude
cat > .claude/settings.local.json << 'EOF'
{
  "permissions": {
    "allow": [
      "Bash(uv sync:*)",
      "Bash(uv run:*)",
      "Bash(git add:*)",
      "Bash(git commit:*)",
      "Bash(git checkout:*)",
      "Bash(git reset:*)",
      "Bash(git branch:*)",
      "Bash(grep:*)",
      "Bash(tail:*)",
      "Bash(head:*)",
      "Bash(timeout:*)"
    ]
  }
}
EOF
```

## Running autoforge-stats (20-30 rounds)

### Start Claude Code and invoke the skill

```bash
cd ~/sp500-forecast
claude
```

Then in the Claude Code prompt, type:

```
/autoforge-stats sp500_train.py
```

The agent will:

1. **Read** `sp500_train.py` and `sp500_prepare.py` to understand the problem
2. **Establish the contract** — confirm the setup:
   - Target file: `sp500_train.py`
   - Eval command: `uv run sp500_train.py`
   - Metric: `combined_score` (higher is better)
   - Immutable: `sp500_prepare.py`
3. **Create a branch**: `autoforge-stats/sp500`
4. **Initialize** `results.tsv` with the header
5. **Run baseline** and record the starting score
6. **Begin the experiment loop** — runs indefinitely until you stop it

### What happens during each round (~30-60 seconds)

```
Round N:
  1. Hypothesize  → "Try RandomForest instead of GradientBoosting"
  2. Edit          → Modify sp500_train.py
  3. Commit        → git commit -m "experiment: switch to RandomForest"
  4. Run           → uv run sp500_train.py > run.log 2>&1
  5. Measure       → grep "^combined_score:" run.log
  6. Decide        → Score improved? KEEP. Worse? git reset --hard HEAD~1
  7. Log           → Append row to results.tsv
  8. Repeat
```

### Typical experiment progression (20-30 rounds)

Based on our experience running this exact setup, here's what the agent typically tries:

| Rounds | Phase | Typical experiments |
|--------|-------|-------------------|
| 1-5 | **Model selection** | Switch from GradientBoosting to RandomForest, try different tree depths |
| 6-10 | **Feature engineering** | Add interaction features, try aggregated feature importance selection |
| 11-15 | **Hyperparameter tuning** | n_estimators, max_depth, min_samples_leaf, max_features |
| 16-20 | **Training strategy** | Sample weighting (recency bias), per-day prediction shrinkage |
| 21-25 | **Regularization** | Tune leaf sizes, remove unnecessary preprocessing (StandardScaler) |
| 26-30 | **Refinement** | Fine-tune parameters, try ensembles, simplify code |

**Expected results after 20-30 rounds:**
- Baseline: ~0.52-0.54 (GradientBoosting with default params)
- After optimization: ~0.54-0.56 (RandomForest with tuned params + feature engineering)
- Keep rate: ~5-15% of experiments (most changes don't help — that's normal!)

### Stopping the agent

Press `Ctrl+C` to stop Claude Code at any time. Your progress is saved:
- All successful improvements are committed to git
- `results.tsv` contains the full experiment log
- Failed experiments are already reverted

### Reviewing results

After the run, check the experiment log:

```bash
cat results.tsv
```

See the git history of successful improvements:

```bash
git log --oneline
```

Compare your final model against the baseline:

```bash
git diff main..HEAD -- sp500_train.py
```

## File structure

```
sp500-forecast/
├── sp500_prepare.py    # Data pipeline & evaluation (DO NOT MODIFY)
├── sp500_train.py      # Model training (autoforge-stats modifies this)
├── pyproject.toml      # Python dependencies
├── results.tsv         # Experiment log (created by autoforge-stats)
├── run.log             # Latest experiment output (created during runs)
└── README.md           # This file
```

## How the baseline works

**Data sources** (fetched automatically via Yahoo Finance):
- S&P 500 daily OHLCV
- VIX, 10Y/5Y Treasury yields, US Dollar Index, Gold, Crude Oil
- Sector ETFs (Tech, Financials, Energy, Healthcare, Industrials)
- Bitcoin, EUR/USD, Russell 2000, High Yield Bonds

**Features** (64 total, built by `sp500_prepare.py`):
- Returns at multiple lags (1, 2, 3, 5, 10, 21 days)
- Moving average ratios (5 to 200-day)
- Volatility measures, RSI, MACD, Bollinger Bands, ATR
- External indicator levels and 5-day changes
- Sector relative strength, cross-asset momentum
- Calendar features (day of week, month)

**Baseline model** (`sp500_train.py`):
- GradientBoosting with default params (n_estimators=100, max_depth=3)
- Top 35 features selected by Random Forest importance
- StandardScaler preprocessing
- One model per forecast day (7 classifiers + 7 regressors = 14 models)
- No sample weighting, no prediction shrinkage

The baseline is intentionally simple to give autoforge-stats room to improve.

## What autoforge-stats is allowed to change

Everything in `sp500_train.py` is fair game:
- **Model type**: RandomForest, XGBoost, neural nets (torch), SVM, ensembles
- **Hyperparameters**: tree depth, learning rate, estimators, regularization
- **Feature engineering**: interaction features, transformations, selection strategy
- **Training strategy**: sample weighting, cross-validation, shrinkage
- **Ensemble methods**: stacking, blending, voting across model types
- **Any creative approach** using packages in `pyproject.toml`

What it **cannot** change:
- `sp500_prepare.py` (data pipeline and evaluation are immutable)
- `pyproject.toml` (no new dependencies)
- The evaluation metric formula
