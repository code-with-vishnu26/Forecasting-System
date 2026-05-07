"""
FastAPI — Main application entry point.
Run: uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import time

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from api.routes import forecast, models, health

# ─────────────────────────────────────────
app = FastAPI(
    title="Sales Forecasting API",
    description=(
        "Production-ready REST API for state-level sales forecasting. "
        "Uses ARIMA, Prophet, XGBoost, and LSTM models with automatic best-model selection."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow all origins for demo purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Request timing middleware ─────────────
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    response.headers["X-Process-Time"] = f"{time.time() - start:.4f}s"
    return response

# ─── Routers ──────────────────────────────
app.include_router(health.router, tags=["Health"])
app.include_router(forecast.router, prefix="/forecast", tags=["Forecast"])
app.include_router(models.router, prefix="/models", tags=["Models"])

# ─── Serve frontend static files ──────────
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

# ─── Root redirect ─────────────────────────
@app.get("/api", include_in_schema=False)
async def api_info():
    return {
        "service": "Sales Forecasting API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "health": "/health",
            "states": "/forecast/states",
            "forecast": "/forecast/{state}?weeks=8",
            "all_models": "/forecast/{state}/all-models",
            "batch": "/forecast/batch",
            "performance": "/models/performance",
        },
    }
