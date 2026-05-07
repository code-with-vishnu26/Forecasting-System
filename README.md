# SalesCast AI — End-to-End Time Series Forecasting System

> Production-ready REST API + Premium Dashboard for 8-week state-level beverage sales forecasting.

---

## Dataset
- **Source**: `Forecasting Case- Study (1).xlsx`
- **Records**: 8,084 rows | 43 US states | Weekly Beverages sales
- **Date range**: 2019-01-12 → 2023-12-03

---

## Project Structure
```
QuickHyre/
├── src/
│   ├── data_pipeline.py          # Load, clean, feature engineering
│   ├── model_selector.py         # Train all models & auto-select best
│   ├── forecaster.py             # Unified inference interface
│   └── models/
│       ├── arima_model.py        # SARIMA (auto-order via AIC)
│       ├── prophet_model.py      # Facebook Prophet
│       ├── xgboost_model.py      # XGBoost (recursive lag forecast)
│       └── lstm_model.py         # LSTM (sequence-to-sequence)
├── api/
│   ├── main.py                   # FastAPI app entry point
│   ├── schemas.py                # Pydantic request/response models
│   └── routes/
│       ├── forecast.py           # /forecast/* endpoints
│       ├── models.py             # /models/* endpoints
│       └── health.py             # /health endpoint
├── frontend/
│   ├── index.html                # Premium dark dashboard
│   ├── styles.css                # Glassmorphism dark UI
│   └── app.js                    # Chart.js charts + API calls
├── models/                       # Saved model files (.pkl)
├── data/                         # features.csv output
├── train.py                      # Master training script
└── requirements.txt
```

---

## Feature Engineering

| Feature | Description |
|---------|-------------|
| `lag_1`, `lag_7`, `lag_30` | Lagged sales at t-1, t-7, t-30 weeks |
| `rolling_mean_4`, `rolling_std_4` | 4-week rolling mean & std |
| `rolling_mean_12`, `rolling_std_12` | 12-week rolling mean & std |
| `trend_1`, `trend_4` | Week-over-week and 4-week diff |
| `week_of_year`, `month`, `quarter` | Calendar features |
| `is_holiday`, `near_holiday` | US federal holiday flags |

**Train/Validation Split**: Last 16 weeks held out as validation (no leakage).

---

## Models

| Model | Details |
|-------|---------|
| **ARIMA/SARIMA** | Auto-order via AIC grid search, seasonal period m=52 |
| **Facebook Prophet** | Yearly + weekly seasonality, US holidays, multiplicative mode |
| **XGBoost** | Lag + rolling + calendar features, TimeSeriesSplit CV, recursive forecast |
| **LSTM** | 2-layer LSTM, 26-week lookback, EarlyStopping, recursive multi-step |

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train all models (all 43 states)
```bash
python train.py
```
To train specific states (faster for testing):
```bash
python train.py --states California Texas Florida
```

### 3. Start the API server
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Open the Dashboard
Navigate to: **http://localhost:8000**
Or open `frontend/index.html` directly in a browser.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Service health check |
| `GET` | `/forecast/states` | List all trained states |
| `GET` | `/forecast/{state}?weeks=8` | Best-model forecast |
| `GET` | `/forecast/{state}/all-models` | All 4 model forecasts |
| `GET` | `/forecast/{state}/history` | Historical + validation data |
| `POST` | `/forecast/batch` | Batch forecast for multiple states |
| `GET` | `/models/performance` | Validation metrics across all states |
| `GET` | `/models/registry` | Full model registry |

**Interactive docs**: http://localhost:8000/docs

---

## Example API Response

```json
GET /forecast/California?weeks=8

{
  "state": "California",
  "best_model": "Prophet",
  "forecast_weeks": 8,
  "forecast": [
    { "date": "2023-12-09", "predicted_sales": 471234567.89 },
    { "date": "2023-12-16", "predicted_sales": 468901234.56 },
    ...
  ],
  "metrics": {
    "rmse": 12345678.9,
    "mae": 9876543.2,
    "mape": 2.34
  }
}
```

---

## Model Auto-Selection
After training, each state's best model is selected based on **validation RMSE** (lowest wins).
The selection is stored in `models/model_registry.json`.

---

## Dashboard Features
- 📊 Interactive time series charts (historical + forecast)
- 🔄 All-models comparison chart
- 🌡️ MAPE heatmap across all states × models
- 📋 Sortable/filterable performance table
- 🔌 Live API explorer with one-click test buttons
