"""
S&P 500 7-Day Forecast — Data Preparation & Evaluation
Fetches free data sources, builds features, and provides evaluation utilities.

DO NOT MODIFY this file. The autoforge-stats agent only modifies sp500_train.py.

Usage: uv run sp500_prepare.py
"""

import os
import pickle
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CACHE_DIR = Path.home() / ".cache" / "sp500_forecast"
FORECAST_HORIZON = 7          # predict 7 days ahead
TEST_MONTHS = 6               # last 6 months held out for testing
LOOKBACK_YEARS = 10           # how far back to fetch history

# ---------------------------------------------------------------------------
# Data fetching with caching
# ---------------------------------------------------------------------------

def _cache_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.pkl"

def _save_cache(name: str, data):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(_cache_path(name), "wb") as f:
        pickle.dump({"date": datetime.now(), "data": data}, f)

def _load_cache(name: str, max_age_hours: int = 12):
    path = _cache_path(name)
    if not path.exists():
        return None
    with open(path, "rb") as f:
        cached = pickle.load(f)
    if datetime.now() - cached["date"] > timedelta(hours=max_age_hours):
        return None
    return cached["data"]

def _fetch_series(ticker: str, cache_name: str, label: str) -> pd.Series:
    """Generic helper to fetch a single Close series from Yahoo Finance."""
    cached = _load_cache(cache_name)
    if cached is not None:
        print(f"  {label}: loaded from cache")
        return cached
    print(f"  {label}: downloading from Yahoo Finance...")
    end = datetime.now()
    start = end - timedelta(days=LOOKBACK_YEARS * 365)
    df = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                     end=end.strftime("%Y-%m-%d"), auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    series = df["Close"].squeeze()
    series.index = pd.to_datetime(series.index)
    series = series.sort_index()
    _save_cache(cache_name, series)
    return series

def fetch_sp500() -> pd.DataFrame:
    """Fetch S&P 500 daily OHLCV data."""
    cached = _load_cache("sp500")
    if cached is not None:
        print("  S&P 500: loaded from cache")
        return cached
    print("  S&P 500: downloading from Yahoo Finance...")
    end = datetime.now()
    start = end - timedelta(days=LOOKBACK_YEARS * 365)
    df = yf.download("^GSPC", start=start.strftime("%Y-%m-%d"),
                     end=end.strftime("%Y-%m-%d"), auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    _save_cache("sp500", df)
    return df

def fetch_vix() -> pd.Series:
    return _fetch_series("^VIX", "vix", "VIX")

def fetch_treasury_yield() -> pd.Series:
    return _fetch_series("^TNX", "tnx", "10Y Treasury")

def fetch_dollar_index() -> pd.Series:
    return _fetch_series("DX-Y.NYB", "dxy", "US Dollar Index")

def fetch_gold() -> pd.Series:
    return _fetch_series("GC=F", "gold", "Gold")

def fetch_oil() -> pd.Series:
    return _fetch_series("CL=F", "oil", "Crude Oil")

def fetch_sector_etfs() -> dict:
    """Fetch major sector ETFs."""
    sectors = {
        "xlk": ("XLK", "Tech ETF"),
        "xlf": ("XLF", "Financials ETF"),
        "xle": ("XLE", "Energy ETF"),
        "xlv": ("XLV", "Healthcare ETF"),
        "xli": ("XLI", "Industrials ETF"),
    }
    return {key: _fetch_series(ticker, key, label)
            for key, (ticker, label) in sectors.items()}

def fetch_bitcoin() -> pd.Series:
    return _fetch_series("BTC-USD", "btc", "Bitcoin")

def fetch_5y_treasury() -> pd.Series:
    return _fetch_series("^FVX", "fvx", "5Y Treasury")

def fetch_eurusd() -> pd.Series:
    return _fetch_series("EURUSD=X", "eurusd", "EUR/USD")

def fetch_smallcap() -> pd.Series:
    return _fetch_series("IWM", "iwm", "Small Cap (IWM)")

def fetch_high_yield_bond() -> pd.Series:
    return _fetch_series("HYG", "hyg", "High Yield Bond (HYG)")

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_features(sp500: pd.DataFrame, vix: pd.Series, tnx: pd.Series,
                   dxy: pd.Series, gold: pd.Series, oil: pd.Series,
                   sectors: dict = None, btc: pd.Series = None,
                   fvx: pd.Series = None, eurusd: pd.Series = None,
                   iwm: pd.Series = None, hyg: pd.Series = None) -> pd.DataFrame:
    """Build feature matrix from raw data."""
    df = pd.DataFrame(index=sp500.index)

    close = sp500["Close"]
    high = sp500["High"]
    low = sp500["Low"]
    volume = sp500["Volume"]

    # --- Price-based features ---
    for lag in [1, 2, 3, 5, 10, 21]:
        df[f"return_{lag}d"] = close.pct_change(lag)

    # Moving average ratios
    for window in [5, 10, 20, 50, 100, 200]:
        ma = close.rolling(window).mean()
        df[f"ma_ratio_{window}"] = close / ma - 1

    # Volatility
    for window in [5, 10, 21]:
        df[f"volatility_{window}d"] = close.pct_change().rolling(window).std()

    # RSI (14-day)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Bollinger Band position
    bb_ma = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["bb_position"] = (close - bb_ma) / (2 * bb_std)

    # Average True Range
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean() / close

    # Volume features
    vol_ma = volume.rolling(20).mean()
    df["volume_ratio"] = volume / vol_ma
    df["volume_change"] = volume.pct_change()

    # Day of week, month
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month

    # --- External indicators ---
    df["vix"] = vix.reindex(df.index, method="ffill")
    df["vix_change_5d"] = df["vix"].pct_change(5)

    df["tnx"] = tnx.reindex(df.index, method="ffill")
    df["tnx_change_5d"] = df["tnx"].pct_change(5)

    df["dxy"] = dxy.reindex(df.index, method="ffill")
    df["dxy_change_5d"] = df["dxy"].pct_change(5)

    df["gold"] = gold.reindex(df.index, method="ffill")
    df["gold_change_5d"] = df["gold"].pct_change(5)

    df["oil"] = oil.reindex(df.index, method="ffill")
    df["oil_change_5d"] = df["oil"].pct_change(5)

    # --- Sector ETFs (relative strength vs S&P 500) ---
    if sectors:
        for key, series in sectors.items():
            s = series.reindex(df.index, method="ffill")
            df[f"{key}_return_5d"] = s.pct_change(5)
            df[f"{key}_relative"] = s.pct_change(5) - close.pct_change(5)

    # --- Bitcoin (risk sentiment proxy) ---
    if btc is not None:
        b = btc.reindex(df.index, method="ffill")
        df["btc_return_5d"] = b.pct_change(5)
        df["btc_return_21d"] = b.pct_change(21)
        df["btc_volatility_10d"] = b.pct_change().rolling(10).std()

    # --- Yield curve spread (5Y - 10Y) ---
    if fvx is not None:
        fvx_s = fvx.reindex(df.index, method="ffill")
        tnx_s = df["tnx"]
        df["yield_curve_5_10"] = fvx_s - tnx_s
        df["yield_curve_change_5d"] = df["yield_curve_5_10"].diff(5)

    # --- EUR/USD (dollar strength from FX side) ---
    if eurusd is not None:
        eu = eurusd.reindex(df.index, method="ffill")
        df["eurusd_return_5d"] = eu.pct_change(5)
        df["eurusd_return_21d"] = eu.pct_change(21)

    # --- Small cap vs large cap (risk-on/risk-off) ---
    if iwm is not None:
        iw = iwm.reindex(df.index, method="ffill")
        df["smallcap_return_5d"] = iw.pct_change(5)
        df["smallcap_vs_sp500"] = iw.pct_change(5) - close.pct_change(5)

    # --- High yield bond spread proxy (credit risk) ---
    if hyg is not None:
        hy = hyg.reindex(df.index, method="ffill")
        df["hyg_return_5d"] = hy.pct_change(5)
        df["hyg_return_21d"] = hy.pct_change(21)

    # --- Cross-feature interactions ---
    df["rsi_x_vix"] = df.get("rsi_14", 0) * df.get("vix", 0) / 100
    df["return_5d_x_volume"] = df.get("return_5d", 0) * df.get("volume_ratio", 0)

    return df

# ---------------------------------------------------------------------------
# Target construction
# ---------------------------------------------------------------------------

def build_targets(sp500: pd.DataFrame) -> pd.DataFrame:
    """Build 7-day forecast targets.

    For each date (day 0), compute:
    - direction_dayN: 1 if close on day N > close on day N-1, else 0
    - pct_dayN: cumulative % change from day 0 close to day N close
    """
    close = sp500["Close"]
    targets = pd.DataFrame(index=sp500.index)

    for day in range(1, FORECAST_HORIZON + 1):
        future_close = close.shift(-day)
        prev_close = close.shift(-(day - 1)) if day > 1 else close
        targets[f"direction_day{day}"] = (future_close > prev_close).astype(float)
        targets[f"pct_day{day}"] = (future_close / close - 1) * 100

    return targets

# ---------------------------------------------------------------------------
# Train/test split
# ---------------------------------------------------------------------------

def train_test_split(features: pd.DataFrame, targets: pd.DataFrame):
    """Split into train and test. Test = last TEST_MONTHS months."""
    valid = features.dropna().index.intersection(targets.dropna().index)
    features = features.loc[valid]
    targets = targets.loc[valid]

    cutoff = features.index.max() - pd.DateOffset(months=TEST_MONTHS)
    train_mask = features.index <= cutoff
    test_mask = features.index > cutoff

    return (features[train_mask], targets[train_mask],
            features[test_mask], targets[test_mask])

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_forecast(y_true_directions: np.ndarray, y_pred_directions: np.ndarray,
                      y_true_pct: np.ndarray, y_pred_pct: np.ndarray) -> dict:
    """Evaluate forecast quality. Returns dict with metrics."""
    dir_acc_per_day = []
    for d in range(FORECAST_HORIZON):
        acc = np.mean(y_true_directions[:, d] == y_pred_directions[:, d])
        dir_acc_per_day.append(acc)

    avg_dir_accuracy = np.mean(dir_acc_per_day)

    mae_per_day = []
    for d in range(FORECAST_HORIZON):
        mae = np.mean(np.abs(y_true_pct[:, d] - y_pred_pct[:, d]))
        mae_per_day.append(mae)
    avg_mae = np.mean(mae_per_day)

    # Combined score: direction accuracy is primary, MAE is secondary penalty
    combined_score = avg_dir_accuracy - 0.01 * avg_mae

    return {
        "avg_direction_accuracy": avg_dir_accuracy,
        "direction_accuracy_per_day": dir_acc_per_day,
        "avg_pct_mae": avg_mae,
        "pct_mae_per_day": mae_per_day,
        "combined_score": combined_score,
    }

def print_results(metrics: dict):
    """Print evaluation results in a standard format for the agent to parse."""
    print("\n---")
    print(f"combined_score:         {metrics['combined_score']:.6f}")
    print(f"avg_direction_accuracy: {metrics['avg_direction_accuracy']:.6f}")
    print(f"avg_pct_mae:            {metrics['avg_pct_mae']:.4f}")
    print(f"direction_accuracy_per_day:")
    for i, acc in enumerate(metrics['direction_accuracy_per_day']):
        print(f"  day_{i+1}: {acc:.4f}")
    print(f"pct_mae_per_day:")
    for i, mae in enumerate(metrics['pct_mae_per_day']):
        print(f"  day_{i+1}: {mae:.4f}")
    print("---")

# ---------------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------------

def load_all_data():
    """Fetch all data, build features/targets, split."""
    print("\nFetching data...")
    sp500 = fetch_sp500()
    vix = fetch_vix()
    tnx = fetch_treasury_yield()
    dxy = fetch_dollar_index()
    gold = fetch_gold()
    oil = fetch_oil()
    sectors = fetch_sector_etfs()
    btc = fetch_bitcoin()
    fvx = fetch_5y_treasury()
    eurusd = fetch_eurusd()
    iwm = fetch_smallcap()
    hyg = fetch_high_yield_bond()

    print("\nBuilding features...")
    features = build_features(sp500, vix, tnx, dxy, gold, oil,
                              sectors=sectors, btc=btc, fvx=fvx,
                              eurusd=eurusd, iwm=iwm, hyg=hyg)

    print("Building targets...")
    targets = build_targets(sp500)

    print("Splitting train/test...")
    X_train, y_train, X_test, y_test = train_test_split(features, targets)

    print(f"\nDataset ready:")
    print(f"  Features:    {len(features.columns)} columns")
    print(f"  Train:       {len(X_train)} days ({X_train.index.min().date()} to {X_train.index.max().date()})")
    print(f"  Test:        {len(X_test)} days ({X_test.index.min().date()} to {X_test.index.max().date()})")
    print(f"  Forecast:    {FORECAST_HORIZON} days ahead")

    return X_train, y_train, X_test, y_test, sp500

if __name__ == "__main__":
    X_train, y_train, X_test, y_test, sp500 = load_all_data()
    print("\nDone! Data cached at:", CACHE_DIR)
