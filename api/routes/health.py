"""
Health check route.
"""

from fastapi import APIRouter
from api.schemas import HealthResponse
from src.forecaster import get_all_states

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Service health check — returns status and number of trained states."""
    try:
        states = get_all_states()
        trained = len(states)
        status = "healthy" if trained > 0 else "degraded"
        message = f"{trained} states trained and ready." if trained > 0 else "No trained models found. Run train.py first."
    except Exception as e:
        status = "degraded"
        message = str(e)
        trained = 0

    return HealthResponse(
        status=status,
        message=message,
        total_states_trained=trained,
    )
