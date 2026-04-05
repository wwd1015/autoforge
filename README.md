# autoforge

Autonomous optimization skills for [Claude Code](https://claude.ai/code). Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch).

Iteratively modifies code, runs experiments, measures a metric, keeps improvements, discards regressions — and repeats indefinitely without human intervention.

## Skills

- **`/autoforge`** — General-purpose. Optimizes anything with a measurable metric: ML models, configs, prompts, strategies, even other skills.
- **`/autoforge-stats`** — Specialized for traditional statistical and ML models (scikit-learn, XGBoost, LightGBM, statsmodels, etc.). Adds CV tracking, leakage prevention, and domain-specific experiment ordering.

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
rm -f ~/.claude/skills/autoforge ~/.claude/skills/autoforge-stats
```

## License

MIT
