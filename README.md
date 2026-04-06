# autoforge

Autonomous optimization skills for [Claude Code](https://claude.ai/code). Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch).

Iteratively modifies code, runs experiments, measures a metric, keeps improvements, discards regressions — and repeats indefinitely without human intervention.

## Skills

- **`/autoforge`** — General-purpose. Optimizes anything with a measurable metric: ML models, configs, prompts, strategies, even other skills.
- **`/autoforge-ml`** — Machine learning models: XGBoost, LightGBM, CatBoost, random forests, SVMs, neural nets, scikit-learn. Feature engineering, hyperparameter tuning, ensembling.
- **`/autoforge-stats`** — Pure statistical models: linear/logistic regression, ridge/lasso, GLMs, ARIMA, exponential smoothing, VAR, GARCH, Cox PH. Econometrics, biostatistics, time series.

## Install

```bash
git clone https://github.com/wwd1015/autoforge.git ~/.autoforge && ~/.autoforge/setup
```

## Update

```bash
cd ~/.autoforge && git pull && ./setup
```

Updates are also checked automatically once per day when you invoke either skill.

## Uninstall

```bash
rm -rf ~/.autoforge
rm -f ~/.claude/skills/autoforge ~/.claude/skills/autoforge-ml ~/.claude/skills/autoforge-stats
```

## License

MIT
