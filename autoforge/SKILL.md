---
name: autoforge
description: Autonomous optimization loop. Iteratively improves any target (model, strategy, config, skill, prompt) by modifying code, running experiments, measuring a metric, and keeping or discarding changes. Use when asked to "optimize", "autoforge", "autoresearch", "experiment loop", or "improve this autonomously".
argument-hint: "[setup-file-or-description]"
disable-model-invocation: true
---

!`~/.autoforge/bin/update-check 2>/dev/null`

If the line above says "UPGRADE_AVAILABLE", tell the user: "autoforge update available! Run `cd ~/.autoforge && git pull && ./setup` to update." Then continue with the skill normally.

# AutoForge — Autonomous Experimentation Loop

You are an autonomous optimizer. You modify code, run experiments, measure results, keep improvements, discard regressions, and repeat — indefinitely, without human intervention.

This skill is generalized: it works on ML models, statistical models, prompts, configs, strategies, or any system with a measurable metric.

## Phase 1: Setup

Before starting the loop, establish the experiment contract with the user. You need to define **five things**:

### 1. The Target File(s)
What file(s) will you modify each iteration?
- Could be a training script, a config, a prompt template, a strategy file, etc.
- Ask the user: "Which file(s) should I experiment with?"

### 2. The Eval Command
How do you measure success? This must be a shell command that produces a numeric metric.
- Examples: `python train.py`, `pytest --benchmark`, `python eval.py`, `bash run_backtest.sh`
- Ask the user: "What command runs an experiment and produces a result?"

### 3. The Metric
What number do you extract from the eval output, and is lower or higher better?
- Examples: `val_bpb` (lower=better), `accuracy` (higher=better), `sharpe_ratio` (higher=better)
- Ask the user: "What metric should I optimize, and is lower or higher better?"
- Define a grep/parse pattern to extract it from output

### 4. The Constraints
What are the hard limits?
- **Time budget**: Max wall-clock per experiment (default: 5 minutes)
- **Resource limits**: Memory, API costs, etc.
- **Immutable files**: Files you must NOT modify (like eval harnesses, data loaders)
- Ask the user: "Any constraints I should respect?"

### 5. The Branch
Create an isolated git branch for experiments.
- `git checkout -b autoforge/<tag>` from current HEAD
- Ask the user for a tag name or suggest one based on today's date

### Setup Checklist

Once you have all five, confirm with the user:

```
Target file(s):  [list]
Eval command:    [command]
Metric:          [name] ([lower/higher] is better)
Constraints:     [list]
Branch:          autoforge/[tag]
```

Initialize a `results.tsv` with header:
```
commit	metric_value	status	description
```

Then run the baseline (unmodified) and record it. This is experiment #0.

## Phase 2: The Experiment Loop

**LOOP FOREVER** (until manually stopped):

### Step 1 — Hypothesize
Based on what you know about the system, form a hypothesis:
- What change might improve the metric?
- Why do you think it will work?
- Keep a mental model of what has been tried (check results.tsv)

### Step 2 — Modify
Edit the target file(s) with your experimental change.
- Make ONE focused change per experiment (isolate variables)
- Use the Edit tool, not manual rewrites

### Step 3 — Commit
```bash
git add [target files]
git commit -m "experiment: [brief description of change]"
```

### Step 4 — Run
```bash
[eval_command] > run.log 2>&1
```
- Redirect ALL output to `run.log` — never let it flood your context
- If the run exceeds 2x the time budget, kill it (`timeout` command)

### Step 5 — Measure
Extract the metric:
```bash
grep "^[metric_name]:" run.log
```
If grep returns empty, the run crashed. Run `tail -n 50 run.log` to diagnose.

### Step 6 — Decide

| Result | Action |
|--------|--------|
| Metric improved | **KEEP** — branch advances, log `keep` in results.tsv |
| Metric equal or worse | **DISCARD** — `git reset --hard HEAD~1`, log `discard` |
| Crashed | **CRASH** — `git reset --hard HEAD~1`, log `crash` with metric_value=0 |

Append result to `results.tsv`:
```
[commit]	[metric_value]	[status]	[description]
```

### Step 7 — Repeat
Go to Step 1. Do NOT ask the human if you should continue.

## Rules

### NEVER STOP
Once the loop begins, do NOT pause to ask the human anything. The human may be away. You run until manually interrupted. If you run out of ideas:
- Re-read the target file for new angles
- Try combining previous near-misses
- Try the opposite of what worked
- Try radical structural changes
- Try simplification (removing code/config)

### Simplicity Criterion
All else equal, simpler is better:
- A tiny improvement that adds ugly complexity? Probably not worth it.
- Removing something and maintaining performance? Definitely keep.
- Weigh improvement magnitude against complexity cost.

### Crash Recovery
- If crash is a typo/import error: fix and re-run (don't count as separate experiment)
- If the idea itself is broken: log crash, revert, move on

### One Change at a Time
Each experiment should test ONE hypothesis. Don't bundle multiple changes — you won't know which one helped.

### Context Management
- Always redirect experiment output to `run.log`
- Read logs with `grep` or `tail`, never cat the full log
- Keep your context clean for long autonomous runs

## Optimization Strategies by Domain

### For ML/Neural Network Training
- Architecture: depth, width, attention heads, activation functions
- Optimizer: learning rate, schedule, momentum, weight decay
- Training: batch size, gradient accumulation, data augmentation
- Regularization: dropout, label smoothing, weight tying

### For Statistical Models
- Feature engineering: transformations, interactions, selections
- Model selection: algorithm choice, ensemble methods
- Hyperparameters: regularization strength, kernel parameters, tree depth
- Validation: cross-validation folds, stratification

### For Prompts/Skills
- Structure: ordering, formatting, emphasis
- Instructions: specificity, examples, constraints
- Evaluation: A/B test on held-out cases

### For Strategies/Configs
- Parameters: thresholds, weights, limits
- Logic: rule ordering, fallback behavior
- Trade-offs: speed vs accuracy, cost vs quality
