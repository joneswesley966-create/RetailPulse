# config/config.py
# RetailPulse – Central Configuration

import os

# ── Paths ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# ── Data ───────────────────────────────────────────────
RAW_FILE = os.path.join(DATA_RAW, "online_retail.csv")
CLEANED_FILE = os.path.join(DATA_PROCESSED, "cleaned_retail.csv")
RFM_FILE = os.path.join(DATA_PROCESSED, "rfm_scores.csv")
FORECAST_FILE = os.path.join(DATA_PROCESSED, "daily_sales.csv")
CHURN_FILE = os.path.join(DATA_PROCESSED, "churn_features.csv")
INVENTORY_FILE = os.path.join(DATA_PROCESSED, "inventory_recommendations.csv")

# ── Segmentation ───────────────────────────────────────
N_CLUSTERS = 6          # K-Means clusters
RANDOM_STATE = 42

# ── Forecasting ────────────────────────────────────────
FORECAST_HORIZON = 30   # days ahead
MAPE_TARGET = 12.0      # % target

# ── Churn ──────────────────────────────────────────────
CHURN_DAYS = 90         # inactive days = churned
AUC_TARGET = 0.88

# ── Inventory ──────────────────────────────────────────
SAFETY_STOCK_MULTIPLIER = 1.5
LEAD_TIME_DAYS = 7

# ── MLflow ─────────────────────────────────────────────
MLFLOW_TRACKING_URI = "mlruns"
EXPERIMENT_NAME = "RetailPulse"