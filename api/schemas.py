"""
Pydantic schemas for request/response validation.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class ForecastPoint(BaseModel):
    date: str
    predicted_sales: float


class ForecastResponse(BaseModel):
    state: str
    best_model: str
    forecast_weeks: int
    forecast: List[ForecastPoint]
    metrics: Dict[str, Any]
    model_comparison: Dict[str, Any]


class AllModelsResponse(BaseModel):
    state: str
    best_model: str
    forecast_weeks: int
    all_models: Dict[str, Any]
    future_dates: List[str]


class BatchForecastRequest(BaseModel):
    states: List[str] = Field(..., description="List of state names")
    weeks: int = Field(8, ge=1, le=52, description="Number of weeks to forecast")


class HealthResponse(BaseModel):
    status: str
    message: str
    total_states_trained: int
    version: str = "1.0.0"


class StateListResponse(BaseModel):
    states: List[str]
    total: int


class PerformanceSummaryResponse(BaseModel):
    states: List[Dict[str, Any]]
    total_states: int
