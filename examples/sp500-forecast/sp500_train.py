"""
S&P 500 7-Day Forecast — Training Script
This is the file the autoforge-stats agent modifies to improve predictions.

Predicts for each of the next 7 days:
  - Direction (up/down from previous day)
  - % change from day 0 (cumulative)

Usage: uv run sp500_train.py
"""

import time
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

from sp500_prepare import (
    FORECAST_HORIZON, load_all_data, evaluate_forecast, print_results
)

# ---------------------------------------------------------------------------
# Model Configuration
# ---------------------------------------------------------------------------

DIRECTION_PARAMS = dict(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    random_state=42,
)

PCT_PARAMS = dict(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    random_state=42,
)

TOP_K_FEATURES = 35   # use top K features by importance

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_and_evaluate():
    start_time = time.time()

    # Load data
    X_train, y_train, X_test, y_test, sp500 = load_all_data()

    # Separate direction and pct targets
    dir_cols = [f"direction_day{d}" for d in range(1, FORECAST_HORIZON + 1)]
    pct_cols = [f"pct_day{d}" for d in range(1, FORECAST_HORIZON + 1)]

    y_train_dir = y_train[dir_cols].values
    y_train_pct = y_train[pct_cols].values
    y_test_dir = y_test[dir_cols].values
    y_test_pct = y_test[pct_cols].values

    # Replace inf with NaN, then fill NaN with 0
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Feature selection: use a quick RF to pick top features
    print("\nSelecting top features...")
    from sklearn.ensemble import RandomForestClassifier
    selector = RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=42, n_jobs=-1
    )
    selector.fit(X_train, y_train_dir[:, 0])
    importances = selector.feature_importances_
    top_idx = np.argsort(importances)[::-1][:TOP_K_FEATURES]
    top_features = X_train.columns[top_idx]
    print(f"  Selected {TOP_K_FEATURES} features from {len(X_train.columns)}")

    X_train = X_train[top_features]
    X_test = X_test[top_features]

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train one model per forecast day for direction
    print("\nTraining direction models...")
    pred_dir = np.zeros_like(y_test_dir)
    for d in range(FORECAST_HORIZON):
        print(f"  Day {d+1}...", end=" ", flush=True)
        model = GradientBoostingClassifier(**DIRECTION_PARAMS)
        model.fit(X_train_scaled, y_train_dir[:, d])
        pred_dir[:, d] = model.predict(X_test_scaled)
        train_acc = model.score(X_train_scaled, y_train_dir[:, d])
        print(f"train_acc={train_acc:.4f}")

    # Train one model per forecast day for % change
    print("\nTraining % change models...")
    pred_pct = np.zeros_like(y_test_pct)
    for d in range(FORECAST_HORIZON):
        print(f"  Day {d+1}...", end=" ", flush=True)
        model = GradientBoostingRegressor(**PCT_PARAMS)
        model.fit(X_train_scaled, y_train_pct[:, d])
        pred_pct[:, d] = model.predict(X_test_scaled)
        train_mae = np.mean(np.abs(model.predict(X_train_scaled) - y_train_pct[:, d]))
        print(f"train_mae={train_mae:.4f}%")

    # Evaluate
    metrics = evaluate_forecast(y_test_dir, pred_dir, y_test_pct, pred_pct)

    # Add timing info
    elapsed = time.time() - start_time
    print_results(metrics)
    print(f"train_time_s:           {elapsed:.1f}")

    # Show a sample forecast (most recent test date)
    last_idx = X_test.index[-1]
    last_close = sp500.loc[last_idx, "Close"]
    if hasattr(last_close, 'item'):
        last_close = last_close.item()
    print(f"\nSample forecast from {last_idx.date()} (close: ${last_close:,.2f}):")
    for d in range(FORECAST_HORIZON):
        direction = "UP" if pred_dir[-1, d] == 1 else "DOWN"
        arrow = "^" if pred_dir[-1, d] == 1 else "v"
        pct = pred_pct[-1, d]
        price = last_close * (1 + pct / 100)
        print(f"  Day {d+1}: {direction} {arrow}  {pct:+.2f}%  (${price:,.2f})")

if __name__ == "__main__":
    train_and_evaluate()
