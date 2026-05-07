"""
Forecast routes — single state, all models, batch.
"""

import asyncio
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from api.schemas import (
    ForecastResponse, AllModelsResponse,
    BatchForecastRequest, StateListResponse,
)
from src.forecaster import (
    forecast_state, forecast_all_models,
    get_all_states, load_registry,
)

router = APIRouter()


@router.get("/states", response_model=StateListResponse)
async def list_states():
    """Return list of all states with trained models."""
    states = get_all_states()
    if not states:
        raise HTTPException(
            status_code=503,
            detail="No trained models available. Run train.py first.",
        )
    return StateListResponse(states=states, total=len(states))


@router.get("/{state}", response_model=ForecastResponse)
async def get_forecast(
    state: str,
    weeks: int = Query(8, ge=1, le=52, description="Number of weeks to forecast"),
):
    """
    Get 8-week sales forecast for a specific state using the best trained model.
    """
    # Normalize state name
    available = get_all_states()
    matched = next((s for s in available if s.lower() == state.lower()), None)
    if not matched:
        raise HTTPException(
            status_code=404,
            detail=f"State '{state}' not found. Available: {available}",
        )
    try:
        result = forecast_state(matched, weeks=weeks)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{state}/all-models", response_model=AllModelsResponse)
async def get_all_model_forecasts(
    state: str,
    weeks: int = Query(8, ge=1, le=52),
):
    """
    Get forecasts from ALL 4 models for a state, plus model comparison metrics.
    """
    available = get_all_states()
    matched = next((s for s in available if s.lower() == state.lower()), None)
    if not matched:
        raise HTTPException(status_code=404, detail=f"State '{state}' not found.")
    try:
        result = forecast_all_models(matched, weeks=weeks)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch")
async def batch_forecast(request: BatchForecastRequest):
    """
    Batch forecast for multiple states simultaneously.
    """
    available = get_all_states()
    results = {}
    errors = {}

    for state in request.states:
        matched = next((s for s in available if s.lower() == state.lower()), None)
        if not matched:
            errors[state] = f"State '{state}' not found."
            continue
        try:
            results[matched] = forecast_state(matched, weeks=request.weeks)
        except Exception as e:
            errors[state] = str(e)

    return {
        "weeks": request.weeks,
        "forecasts": results,
        "errors": errors,
        "success_count": len(results),
        "error_count": len(errors),
    }


@router.get("/{state}/history")
async def get_state_history(state: str):
    """Return historical data and validation predictions for a state."""
    available = get_all_states()
    matched = next((s for s in available if s.lower() == state.lower()), None)
    if not matched:
        raise HTTPException(status_code=404, detail=f"State '{state}' not found.")

    registry = load_registry()
    info = registry.get(matched, {})
    return {
        "state": matched,
        "best_model": info.get("best_model", "N/A"),
        "last_train_date": info.get("last_train_date"),
        "val_dates": info.get("val_dates", []),
        "val_actual": info.get("val_actual", []),
        "val_predictions": info.get("val_predictions", {}),
        "metrics": {k: {m: v for m, v in metric.items() if m != "predictions"}
                    for k, metric in info.get("metrics", {}).items()},
    }
