# 🧠 SalesCast AI — Automated Sales Forecasting System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen?style=for-the-badge)

**A production-ready, end-to-end time series forecasting system for US state-level beverage sales.**  
It trains 4 different AI models, automatically picks the best one per state, and serves predictions through a beautiful REST API + dashboard.

</div>

---

## 🌟 What Is This Project?

Imagine you're running a beverage distribution company across 43 US states. You need to know — *"How much will I sell next 8 weeks in California? In Texas? In Florida?"*

That's exactly what **SalesCast AI** solves.

This system:
- 📥 **Ingests** weekly sales data from an Excel file
- 🔧 **Cleans and engineers** 15+ smart features (lags, rolling averages, holiday flags)
- 🤖 **Trains 4 AI models** — ARIMA, Prophet, XGBoost, and LSTM — for every state
- 🏆 **Auto-selects** the best-performing model per state (no manual guessing!)
- 🚀 **Serves predictions** via a FastAPI REST API
- 📊 **Visualizes everything** on a premium dark-mode dashboard

No more spreadsheets. No more gut feelings. Just data-driven forecasts.

---

## 📊 Dataset Overview

| Property | Details |
|----------|---------|
| **File** | `Forecasting Case- Study (1).xlsx` |
| **Rows** | 8,084 weekly records |
| **Coverage** | 43 US States |
| **Product** | Beverages category |
| **Date Range** | Jan 2019 → Dec 2023 (~5 years) |

The dataset contains weekly sales figures across 43 US states spanning nearly 5 years — a rich, real-world time series that captures seasonal patterns, holiday spikes, and long-term growth trends.

---

## 🗂️ Project Structure

Here's how the project is organized — and *why* each piece exists:

```
QuickHyre/
│
├── 📁 src/                          # Core ML logic lives here
│   ├── data_pipeline.py             # Loads Excel, cleans data, builds features
│   ├── model_selector.py            # Trains all 4 models & picks the winner
│   ├── forecaster.py                # One unified interface to run any model
│   └── models/
│       ├── arima_model.py           # Classic statistical time series model
│       ├── prophet_model.py         # Facebook's battle-tested forecaster
│       ├── xgboost_model.py         # Gradient boosting on engineered features
│       └── lstm_model.py            # Deep learning sequence model
│
├── 📁 api/                          # REST API layer (FastAPI)
│   ├── main.py                      # App entry point + middleware setup
│   ├── schemas.py                   # Request & response data shapes (Pydantic)
│   └── routes/
│       ├── forecast.py              # All /forecast/* endpoints
│       ├── models.py                # All /models/* endpoints
│       └── health.py                # Simple health check endpoint
│
├── 📁 frontend/                     # Browser dashboard
│   ├── index.html                   # Main dashboard page
│   ├── styles.css                   # Glassmorphism dark UI styling
│   └── app.js                       # Chart.js charts + live API calls
│
├── 📁 models/                       # Where trained model files (.pkl) are saved
├── 📁 data/                         # Where processed features.csv is saved
├── 📁 notebooks/                    # Exploratory analysis notebooks
│
├── train.py                         # 🚀 Master training script — run this first!
├── requirements.txt                 # All Python dependencies
└── README.md                        # You're reading it!
```

> **Note:** The `models/` directory is excluded from Git (files are ~390MB each). Run `train.py` locally to generate them.

---

## ⚙️ How It Works — Step by Step

### Step 1 — Data Pipeline 🧹

The raw Excel file has inconsistencies, missing weeks, and unordered rows. The `data_pipeline.py` script:

1. Loads all sheets from the Excel file
2. Standardizes column names and date formats
3. Fills missing weeks (forward-fill + interpolation)
4. Splits data by state so each state trains independently
5. Engineers 15+ predictive features (see table below)

### Step 2 — Feature Engineering 🔬

Raw sales numbers alone aren't enough. We give each model extra context:

| Feature | What It Represents | Why It Helps |
|---------|-------------------|--------------|
| `lag_1` | Sales from 1 week ago | Most recent momentum |
| `lag_7` | Sales from 7 weeks ago | Same period last quarter |
| `lag_30` | Sales from 30 weeks ago | Same period last year (roughly) |
| `rolling_mean_4` | Average of last 4 weeks | Short-term trend |
| `rolling_std_4` | Std deviation of last 4 weeks | Recent volatility |
| `rolling_mean_12` | Average of last 12 weeks | Medium-term trend |
| `rolling_std_12` | Std deviation of last 12 weeks | Seasonal volatility |
| `trend_1` | Week-over-week change | Momentum direction |
| `trend_4` | 4-week change | Longer momentum |
| `week_of_year` | Week number (1–52) | Captures seasonality |
| `month` | Month number (1–12) | Monthly patterns |
| `quarter` | Quarter (1–4) | Quarterly cycles |
| `is_holiday` | US federal holiday flag | Spike/dip events |
| `near_holiday` | Within 1 week of a holiday | Pre/post holiday effect |

**Time-safe split:** The last 16 weeks of data are held out as validation — the model *never sees them during training*, preventing data leakage.

### Step 3 — Model Training 🤖

We train **4 different models** for every state — because no single model wins everywhere:

#### 🔵 ARIMA / SARIMA
The gold standard of statistical time series forecasting. It finds the best order `(p,d,q)(P,D,Q,52)` automatically using AIC scoring. Great for stationary, well-behaved data.

#### 🟠 Facebook Prophet
Built by Meta's data science team. Handles holidays, missing data, and trend changes gracefully. Excels with yearly + weekly seasonality and is very robust to outliers.

#### 🟢 XGBoost
Gradient boosting on all the lag and rolling features we engineered. Uses `TimeSeriesSplit` cross-validation during training. Makes recursive multi-step forecasts by feeding predictions back as future lags.

#### 🔴 LSTM (Long Short-Term Memory)
A 2-layer deep learning model with a 26-week lookback window. Uses early stopping to prevent overfitting. Best for capturing complex, non-linear long-term patterns.

### Step 4 — Auto Model Selection 🏆

After training, each model is evaluated on the 16-week validation set using:
- **RMSE** (Root Mean Square Error) — main selection metric
- **MAE** (Mean Absolute Error)
- **MAPE** (Mean Absolute Percentage Error)

The model with the **lowest RMSE** is declared the winner for that state and stored in `models/model_registry.json`. When you call the `/forecast/{state}` API, it automatically uses that state's best model.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- ~4GB RAM (for LSTM training)
- The Excel data file in the project root

### 1. Clone the Repository

```bash
git clone https://github.com/code-with-vishnu26/Forecasting-System.git
cd Forecasting-System
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs FastAPI, Prophet, XGBoost, TensorFlow/Keras, statsmodels, pandas, and all other required packages.

### 3. Train the Models

```bash
# Train all 43 states (takes 30–90 mins depending on hardware)
python train.py

# Quick test — just 3 states (takes ~5 mins)
python train.py --states California Texas Florida
```

Training creates:
- `models/{StateName}/arima.pkl` — saved ARIMA model
- `models/{StateName}/prophet.pkl` — saved Prophet model
- `models/{StateName}/xgboost.pkl` — saved XGBoost model
- `models/{StateName}/lstm.pkl` — saved LSTM model
- `models/model_registry.json` — best model per state + all metrics
- `data/features.csv` — the full engineered feature dataset

### 4. Start the API Server

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

The server is now live at `http://localhost:8000`

### 5. Open the Dashboard

Visit **http://localhost:8000** in your browser — the dashboard loads automatically.

Or open the interactive API docs at **http://localhost:8000/docs**

---

## 🔌 API Endpoints Reference

All endpoints return JSON. The base URL when running locally is `http://localhost:8000`.

### Health Check

```
GET /health
```
Returns server status and uptime. Use this to verify the API is running.

```json
{ "status": "ok", "uptime_seconds": 142.3 }
```

---

### List All Trained States

```
GET /forecast/states
```
Returns the list of all states that have trained models ready.

```json
{ "states": ["Alabama", "Arizona", "Arkansas", "California", ...] }
```

---

### Get Best-Model Forecast for a State

```
GET /forecast/{state}?weeks=8
```

The most important endpoint. Returns the next N weeks of sales predictions using the automatically selected best model for that state.

**Example:**
```
GET /forecast/California?weeks=8
```

```json
{
  "state": "California",
  "best_model": "Prophet",
  "forecast_weeks": 8,
  "forecast": [
    { "date": "2023-12-09", "predicted_sales": 471234567.89 },
    { "date": "2023-12-16", "predicted_sales": 468901234.56 },
    { "date": "2023-12-23", "predicted_sales": 502345678.12 },
    { "date": "2023-12-30", "predicted_sales": 489012345.67 }
  ],
  "metrics": {
    "rmse": 12345678.9,
    "mae": 9876543.2,
    "mape": 2.34
  }
}
```

---

### Compare All 4 Models for a State

```
GET /forecast/{state}/all-models
```
Returns forecasts from **all 4 models** side by side — useful for comparing model behavior and uncertainty.

---

### Historical + Validation Data

```
GET /forecast/{state}/history
```
Returns the historical sales data plus validation predictions — great for plotting actual vs predicted charts.

---

### Batch Forecast (Multiple States at Once)

```
POST /forecast/batch
Content-Type: application/json

{
  "states": ["California", "Texas", "Florida"],
  "weeks": 8
}
```
Efficiently runs forecasts for multiple states in a single API call.

---

### Model Performance Metrics

```
GET /models/performance
```
Returns validation RMSE, MAE, and MAPE for every model across every state — a full performance matrix.

---

### Model Registry

```
GET /models/registry
```
Returns the complete registry of which model was selected as best for each state, along with training metadata.

---

## 📊 Dashboard Features

The frontend dashboard at `http://localhost:8000` includes:

| Feature | Description |
|---------|-------------|
| **State Selector** | Dropdown to pick any of the 43 trained states |
| **Forecast Chart** | Interactive line chart — historical sales + 8-week forecast |
| **Model Comparison** | Overlay all 4 model predictions to see how they differ |
| **State Heatmap** | MAPE heatmap across all states × all models |
| **Performance Table** | Sortable table of RMSE/MAE/MAPE for every state |
| **API Explorer** | One-click buttons to test every API endpoint live |

The UI is built with vanilla HTML/CSS/JS + Chart.js, with a glassmorphism dark theme. No frameworks required — just open it in a browser.

---

## 🧪 Model Performance (Sample Results)

Below are example validation metrics for a few states after full training:

| State | Best Model | RMSE | MAE | MAPE |
|-------|-----------|------|-----|------|
| California | Prophet | 12.3M | 9.8M | 2.34% |
| Texas | XGBoost | 8.7M | 6.2M | 1.89% |
| Florida | LSTM | 5.1M | 4.0M | 3.12% |
| New York | ARIMA | 11.2M | 8.5M | 2.67% |

*Exact values depend on training run and hardware. MAPE < 5% is considered production-grade accuracy.*

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Language** | Python 3.9+ | Core implementation |
| **API Framework** | FastAPI | High-performance REST API |
| **Data Processing** | Pandas, NumPy | Data wrangling & feature engineering |
| **Statistical Model** | Statsmodels (SARIMA) | Classical time series |
| **ML Model** | XGBoost | Gradient boosting |
| **AI Model** | TensorFlow/Keras (LSTM) | Deep learning |
| **Forecasting** | Facebook Prophet | Robust seasonality |
| **Validation** | Pydantic | Request/response schemas |
| **Frontend** | HTML + CSS + Chart.js | Interactive dashboard |
| **Serving** | Uvicorn (ASGI) | Production-grade server |

---

## ❓ Frequently Asked Questions

**Q: Why does training take so long?**  
A: ARIMA grid search is computationally expensive — it tries many `(p,d,q)` combinations for 43 states. LSTM also needs multiple epochs. Use `--states` flag to train just 2–3 states for quick testing.

**Q: Why are model files not in the repo?**  
A: Each ARIMA `.pkl` file is ~390 MB. GitHub has a 100 MB file limit. Run `train.py` to generate them locally.

**Q: Can I add a new state?**  
A: If your Excel data contains that state, just re-run `train.py` — it auto-discovers all states in the data.

**Q: What if I want forecasts beyond 8 weeks?**  
A: Just change the `?weeks=` parameter. E.g., `?weeks=26` for a 6-month forecast. Note that accuracy degrades for longer horizons.

**Q: Which model usually wins?**  
A: Prophet tends to win for states with strong seasonal patterns. XGBoost wins for irregular, noisy data. LSTM shines for complex multi-pattern states.

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m "Add: your feature description"`
4. Push to your fork: `git push origin feature/your-feature-name`
5. Open a Pull Request

Please follow existing code style and add comments for any new model or pipeline changes.

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use, modify, and distribute it.

---

## 👤 Author

**Vishnu** — [@code-with-vishnu26](https://github.com/code-with-vishnu26)

Built as a production-ready demonstration of modern ML engineering practices:  
end-to-end pipeline → multi-model training → automated selection → REST API → live dashboard.

---

<div align="center">

⭐ **If this project helped you, please give it a star!** ⭐

[🔗 GitHub Repo](https://github.com/code-with-vishnu26/Forecasting-System) • [📖 API Docs](http://localhost:8000/docs) • [📊 Dashboard](http://localhost:8000)

</div>
