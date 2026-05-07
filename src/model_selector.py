"""
Model Selector — Evaluate all models and auto-select best per state.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = actual != 0
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def evaluate_all_models(
    state: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_dir: str = "models",
) -> Dict[str, Any]:
    """
    Train all 4 models on train_df, predict val_df length, compute metrics.
    Returns dict: {model_name: {rmse, mae, mape, predictions}}
    """
    from src.models.arima_model import ARIMAForecaster
    from src.models.prophet_model import ProphetForecaster
    from src.models.xgboost_model import XGBoostForecaster
    from src.models.lstm_model import LSTMForecaster

    val_actual = val_df["Total"].values
    steps = len(val_df)
    results = {}

    # ---- ARIMA ----
    try:
        arima = ARIMAForecaster(state=state, seasonal=True, m=52)
        arima.fit(train_df.set_index("Date")["Total"])
        arima_preds = arima.predict(steps)
        results["ARIMA"] = {
            "rmse": rmse(val_actual, arima_preds),
            "mae": mae(val_actual, arima_preds),
            "mape": mape(val_actual, arima_preds),
            "predictions": arima_preds.tolist(),
        }
        arima.save(os.path.join(model_dir, state, "arima.pkl"))
    except Exception as e:
        print(f"  [ARIMA] Failed for {state}: {e}")
        results["ARIMA"] = {"rmse": 1e18, "mae": 1e18, "mape": 1e18, "predictions": []}

    # ---- Prophet ----
    try:
        prophet = ProphetForecaster(state=state)
        prophet.fit(train_df[["Date", "Total"]])
        last_train_date = train_df["Date"].max()
        prophet_preds = prophet.predict(steps, last_train_date)
        results["Prophet"] = {
            "rmse": rmse(val_actual, prophet_preds),
            "mae": mae(val_actual, prophet_preds),
            "mape": mape(val_actual, prophet_preds),
            "predictions": prophet_preds.tolist(),
        }
        prophet.save(os.path.join(model_dir, state, "prophet.pkl"))
    except Exception as e:
        print(f"  [Prophet] Failed for {state}: {e}")
        results["Prophet"] = {"rmse": 1e18, "mae": 1e18, "mape": 1e18, "predictions": []}

    # ---- XGBoost ----
    try:
        xgb = XGBoostForecaster(state=state)
        xgb.fit(train_df)
        xgb_preds = xgb.predict(steps)
        results["XGBoost"] = {
            "rmse": rmse(val_actual, xgb_preds),
            "mae": mae(val_actual, xgb_preds),
            "mape": mape(val_actual, xgb_preds),
            "predictions": xgb_preds.tolist(),
        }
        xgb.save(os.path.join(model_dir, state, "xgboost.pkl"))
    except Exception as e:
        print(f"  [XGBoost] Failed for {state}: {e}")
        results["XGBoost"] = {"rmse": 1e18, "mae": 1e18, "mape": 1e18, "predictions": []}

    # ---- LSTM ----
    try:
        lstm = LSTMForecaster(state=state, lookback=26, epochs=60, batch_size=16)
        lstm.fit(train_df.set_index("Date")["Total"])
        lstm_preds = lstm.predict(steps)
        results["LSTM"] = {
            "rmse": rmse(val_actual, lstm_preds),
            "mae": mae(val_actual, lstm_preds),
            "mape": mape(val_actual, lstm_preds),
            "predictions": lstm_preds.tolist(),
        }
        lstm.save(os.path.join(model_dir, state, "lstm.pkl"))
    except Exception as e:
        print(f"  [LSTM] Failed for {state}: {e}")
        results["LSTM"] = {"rmse": 1e18, "mae": 1e18, "mape": 1e18, "predictions": []}

    # ---- Select best by RMSE ----
    valid = {k: v for k, v in results.items() if v["rmse"] < 1e17}
    best_model = min(valid, key=lambda k: valid[k]["rmse"]) if valid else "Prophet"

    return {
        "state": state,
        "best_model": best_model,
        "metrics": results,
        "val_dates": val_df["Date"].dt.strftime("%Y-%m-%d").tolist(),
        "val_actual": val_actual.tolist(),
    }


def save_registry(registry: Dict, path: str = "models/model_registry.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(registry, f, indent=2)
    print(f"Registry saved to {path}")


def load_registry(path: str = "models/model_registry.json") -> Dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)
