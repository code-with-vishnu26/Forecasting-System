"""
Forecaster — Unified interface: load best model per state and produce 8-week forecast.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any


REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "model_registry.json")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def load_registry() -> dict:
    if not os.path.exists(REGISTRY_PATH):
        return {}
    with open(REGISTRY_PATH) as f:
        return json.load(f)


def forecast_state(state: str, weeks: int = 8) -> Dict[str, Any]:
    """
    Load the best trained model for a state and return 8-week forecast.
    """
    registry = load_registry()
    if state not in registry:
        raise ValueError(f"State '{state}' not found in registry. Run train.py first.")

    best_model_name = registry[state]["best_model"]
    model_path = os.path.join(MODEL_DIR, state, f"{best_model_name.lower()}.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Load appropriate model class
    if best_model_name == "ARIMA":
        from src.models.arima_model import ARIMAForecaster
        model = ARIMAForecaster.load(model_path)
        preds = model.predict(weeks)
        last_date = pd.Timestamp(registry[state].get("last_train_date", "2023-12-01"))
    elif best_model_name == "Prophet":
        from src.models.prophet_model import ProphetForecaster
        model = ProphetForecaster.load(model_path)
        last_date = pd.Timestamp(registry[state].get("last_train_date", "2023-12-01"))
        preds = model.predict(weeks, last_date)
    elif best_model_name == "XGBoost":
        from src.models.xgboost_model import XGBoostForecaster
        model = XGBoostForecaster.load(model_path)
        preds = model.predict(weeks)
        last_date = pd.Timestamp(registry[state].get("last_train_date", "2023-12-01"))
    elif best_model_name == "LSTM":
        from src.models.lstm_model import LSTMForecaster
        model = LSTMForecaster.load(model_path)
        preds = model.predict(weeks)
        last_date = pd.Timestamp(registry[state].get("last_train_date", "2023-12-01"))
    else:
        raise ValueError(f"Unknown model: {best_model_name}")

    # Generate future dates
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(weeks=1),
        periods=weeks,
        freq="W-SAT",
    )

    return {
        "state": state,
        "best_model": best_model_name,
        "forecast_weeks": weeks,
        "forecast": [
            {"date": str(d.date()), "predicted_sales": round(float(p), 2)}
            for d, p in zip(future_dates, preds)
        ],
        "metrics": registry[state].get("metrics", {}),
        "model_comparison": registry[state].get("metrics", {}),
    }


def forecast_all_models(state: str, weeks: int = 8) -> Dict[str, Any]:
    """Return forecasts from ALL trained models for a state."""
    registry = load_registry()
    if state not in registry:
        raise ValueError(f"State '{state}' not found in registry.")

    last_date = pd.Timestamp(registry[state].get("last_train_date", "2023-12-01"))
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(weeks=1),
        periods=weeks,
        freq="W-SAT",
    )
    date_strs = [str(d.date()) for d in future_dates]

    all_forecasts = {}
    model_names = ["ARIMA", "Prophet", "XGBoost", "LSTM"]
    model_loaders = {
        "ARIMA": ("src.models.arima_model", "ARIMAForecaster"),
        "Prophet": ("src.models.prophet_model", "ProphetForecaster"),
        "XGBoost": ("src.models.xgboost_model", "XGBoostForecaster"),
        "LSTM": ("src.models.lstm_model", "LSTMForecaster"),
    }

    for name in model_names:
        model_path = os.path.join(MODEL_DIR, state, f"{name.lower()}.pkl")
        if not os.path.exists(model_path):
            continue
        try:
            mod_path, cls_name = model_loaders[name]
            import importlib
            module = importlib.import_module(mod_path)
            cls = getattr(module, cls_name)
            model = cls.load(model_path)

            if name == "ARIMA":
                preds = model.predict(weeks)
            elif name == "Prophet":
                preds = model.predict(weeks, last_date)
            elif name in ("XGBoost", "LSTM"):
                preds = model.predict(weeks)
            else:
                continue

            all_forecasts[name] = {
                "dates": date_strs,
                "predictions": [round(float(p), 2) for p in preds],
                "metrics": registry[state].get("metrics", {}).get(name, {}),
            }
        except Exception as e:
            all_forecasts[name] = {"error": str(e)}

    return {
        "state": state,
        "best_model": registry[state]["best_model"],
        "forecast_weeks": weeks,
        "all_models": all_forecasts,
        "future_dates": date_strs,
    }


def get_all_states() -> List[str]:
    """Return list of states that have been trained."""
    registry = load_registry()
    return sorted(registry.keys())


def get_performance_summary() -> Dict[str, Any]:
    """Return model performance summary across all states."""
    registry = load_registry()
    summary = []
    for state, info in registry.items():
        best = info.get("best_model", "N/A")
        metrics = info.get("metrics", {})
        row = {"state": state, "best_model": best}
        for model_name, m in metrics.items():
            if isinstance(m, dict):
                row[f"{model_name}_rmse"] = round(m.get("rmse", 0), 2)
                row[f"{model_name}_mape"] = round(m.get("mape", 0), 2)
        summary.append(row)
    return {"states": summary, "total_states": len(summary)}
