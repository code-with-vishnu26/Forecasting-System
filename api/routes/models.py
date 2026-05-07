"""
Model performance routes.
"""

from fastapi import APIRouter, HTTPException
from api.schemas import PerformanceSummaryResponse
from src.forecaster import get_performance_summary, load_registry

router = APIRouter()


@router.get("/performance", response_model=PerformanceSummaryResponse)
async def get_model_performance():
    """Return model performance metrics (RMSE, MAE, MAPE) for all trained states."""
    try:
        summary = get_performance_summary()
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/registry")
async def get_registry():
    """Return raw model registry (best model per state and all metrics)."""
    try:
        registry = load_registry()
        if not registry:
            raise HTTPException(status_code=404, detail="No trained models found. Run train.py first.")
        # Sanitize for JSON (remove large val_predictions arrays)
        clean = {}
        for state, info in registry.items():
            clean[state] = {k: v for k, v in info.items() if k != "val_predictions"}
        return {"registry": clean, "total": len(clean)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
