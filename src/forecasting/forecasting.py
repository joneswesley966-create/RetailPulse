# src/forecasting/forecasting.py
# RetailPulse – Demand Forecasting (Prophet + LSTM Ensemble)

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import mlflow
import mlflow.sklearn
import joblib

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from config import *

# Force local MLflow tracking
MLFLOW_URI = "sqlite:///C:/Users/Lenovo/PyCharmMiscProject/mlflow.db"
EXPERIMENT_NAME = "demand_forecasting"
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────────────────────────────────────

def load_daily_sales(filepath: str = FORECAST_FILE) -> pd.DataFrame:
    df = pd.read_csv(filepath, parse_dates=["ds"])
    df = df.sort_values("ds").reset_index(drop=True)
    print(f"[FORECAST] Loaded daily sales: {df.shape}, "
          f"from {df['ds'].min().date()} to {df['ds'].max().date()}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. STATIONARITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

def check_stationarity(series: pd.Series) -> None:
    from statsmodels.tsa.stattools import adfuller
    result = adfuller(series.dropna())
    print(f"\n[STATIONARITY] ADF Test:")
    print(f"  ADF Statistic : {result[0]:.4f}")
    print(f"  p-value       : {result[1]:.4f}")
    print(f"  Stationary    : {'Yes ✓' if result[1] < 0.05 else 'No – differencing needed'}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. PROPHET MODEL
# ─────────────────────────────────────────────────────────────────────────────

def train_prophet(df: pd.DataFrame, horizon: int = FORECAST_HORIZON):
    """Train Facebook Prophet model."""
    from prophet import Prophet
    from prophet.diagnostics import cross_validation, performance_metrics

    print(f"\n[PROPHET] Training Prophet model (horizon={horizon} days) …")

    train = df[["ds", "y"]].copy()

    with mlflow.start_run(run_name="Prophet_Forecast"):
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10,
        )
        model.fit(train)

        # Future dataframe
        future = model.make_future_dataframe(periods=horizon)
        forecast = model.predict(future)

        # Cross-validation (use last 20% as validation)
        cutoff = df["ds"].max() - pd.Timedelta(days=horizon)
        train_cv = train[train["ds"] <= cutoff]
        val_cv   = train[train["ds"] > cutoff]

        if len(val_cv) > 0:
            val_pred = forecast[forecast["ds"].isin(val_cv["ds"])]["yhat"].values
            val_true = val_cv["y"].values[:len(val_pred)]
            mape = np.mean(np.abs((val_true - val_pred) / (val_true + 1e-8))) * 100
        else:
            mape = 999.0

        print(f"  → MAPE: {mape:.2f}% (target ≤ {MAPE_TARGET}%)")
        mlflow.log_param("horizon_days", horizon)
        mlflow.log_metric("MAPE", mape)

    # Save model
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODELS_DIR, "prophet_model.pkl"))

    # Plot
    _plot_forecast(df, forecast, "Prophet", mape)
    return model, forecast, mape


def _plot_forecast(df, forecast, model_name, mape):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df["ds"], df["y"], label="Actual", color="steelblue", linewidth=1)

    future_forecast = forecast[forecast["ds"] > df["ds"].max()]
    ax.plot(forecast["ds"], forecast["yhat"],
            label=f"{model_name} Forecast", color="orange", linewidth=1.5)
    ax.fill_between(forecast["ds"], forecast["yhat_lower"],
                    forecast["yhat_upper"], alpha=0.2, color="orange")
    ax.axvline(df["ds"].max(), color="red", linestyle="--", label="Forecast Start")
    ax.set_title(f"{model_name} Demand Forecast | MAPE: {mape:.2f}%")
    ax.set_xlabel("Date"); ax.set_ylabel("Daily Revenue (£)")
    ax.legend(); plt.tight_layout()
    os.makedirs(REPORTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(REPORTS_DIR, f"{model_name.lower()}_forecast.png"), dpi=150)
    plt.close()
    print(f"  → Forecast plot saved")


# ─────────────────────────────────────────────────────────────────────────────
# 4. LSTM MODEL
# ─────────────────────────────────────────────────────────────────────────────

def build_sequences(series: np.ndarray, seq_len: int = 30):
    X, y = [], []
    for i in range(len(series) - seq_len):
        X.append(series[i:i + seq_len])
        y.append(series[i + seq_len])
    return np.array(X), np.array(y)


def train_lstm(df: pd.DataFrame, seq_len: int = 30, epochs: int = 50,
               horizon: int = FORECAST_HORIZON):
    """Train PyTorch LSTM model."""
    try:
        import torch
        import torch.nn as nn
        from sklearn.preprocessing import MinMaxScaler
    except ImportError:
        print("[LSTM] PyTorch not available – skipping LSTM training")
        return None, None, 999.0

    print(f"\n[LSTM] Training LSTM model (seq_len={seq_len}, epochs={epochs}) …")

    values = df["y"].values.astype(float)
    scaler_lstm = MinMaxScaler()
    values_scaled = scaler_lstm.fit_transform(values.reshape(-1, 1)).flatten()

    # Train/val split (80/20)
    split = int(len(values_scaled) * 0.8)
    train_vals = values_scaled[:split]
    val_vals   = values_scaled[split:]

    X_train, y_train = build_sequences(train_vals, seq_len)
    X_val,   y_val   = build_sequences(val_vals, seq_len)

    if len(X_train) == 0 or len(X_val) == 0:
        print("  [WARN] Not enough data for LSTM – skipping")
        return None, None, 999.0

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).unsqueeze(-1)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t   = torch.FloatTensor(X_val).unsqueeze(-1)

    # ── Model ──────────────────────────────────────────
    class LSTMModel(nn.Module):
        def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                batch_first=True, dropout=dropout)
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :]).squeeze()

    model_lstm = LSTMModel()
    optimizer  = torch.optim.Adam(model_lstm.parameters(), lr=0.001)
    criterion  = nn.MSELoss()

    # ── Training loop ──────────────────────────────────
    model_lstm.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model_lstm(X_train_t)
        loss = criterion(pred, y_train_t)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs} | Loss: {loss.item():.6f}")

    # ── Validation ─────────────────────────────────────
    model_lstm.eval()
    with torch.no_grad():
        val_pred_scaled = model_lstm(X_val_t).numpy()

    val_pred = scaler_lstm.inverse_transform(val_pred_scaled.reshape(-1, 1)).flatten()
    val_true = scaler_lstm.inverse_transform(y_val.reshape(-1, 1)).flatten()
    mape = np.mean(np.abs((val_true - val_pred) / (val_true + 1e-8))) * 100
    print(f"  → LSTM MAPE: {mape:.2f}%")

    # ── Future forecast ────────────────────────────────
    last_seq = torch.FloatTensor(values_scaled[-seq_len:]).unsqueeze(0).unsqueeze(-1)
    future_preds = []
    with torch.no_grad():
        seq = last_seq.clone()
        for _ in range(horizon):
            p = model_lstm(seq).item()
            future_preds.append(p)
            seq = torch.cat([seq[:, 1:, :],
                             torch.FloatTensor([[[p]]])], dim=1)

    future_vals = scaler_lstm.inverse_transform(
        np.array(future_preds).reshape(-1, 1)).flatten()
    future_dates = pd.date_range(df["ds"].max() + pd.Timedelta(days=1), periods=horizon)
    forecast_lstm = pd.DataFrame({"ds": future_dates, "yhat": future_vals})

    # Save
    torch.save(model_lstm.state_dict(), os.path.join(MODELS_DIR, "lstm_model.pt"))
    joblib.dump(scaler_lstm, os.path.join(MODELS_DIR, "lstm_scaler.pkl"))

    with mlflow.start_run(run_name="LSTM_Forecast"):
        mlflow.log_param("seq_len", seq_len)
        mlflow.log_param("epochs", epochs)
        mlflow.log_metric("MAPE", mape)

    return model_lstm, forecast_lstm, mape


# ─────────────────────────────────────────────────────────────────────────────
# 5. ENSEMBLE (Prophet + LSTM)
# ─────────────────────────────────────────────────────────────────────────────

def ensemble_forecast(prophet_fc: pd.DataFrame, lstm_fc,
                      alpha: float = 0.6) -> pd.DataFrame:
    """
    Weighted average ensemble:  alpha * Prophet + (1-alpha) * LSTM
    Default: 60% Prophet, 40% LSTM (Prophet tends to be more reliable on short series)
    """
    if lstm_fc is None:
        print("[ENSEMBLE] LSTM unavailable – using Prophet only")
        return prophet_fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(FORECAST_HORIZON)

    future_prophet = prophet_fc[prophet_fc["ds"].isin(lstm_fc["ds"])].copy()
    merged = future_prophet.merge(lstm_fc.rename(columns={"yhat": "yhat_lstm"}),
                                  on="ds", how="inner")
    merged["yhat_ensemble"] = (alpha * merged["yhat"] +
                                (1 - alpha) * merged["yhat_lstm"])

    print(f"\n[ENSEMBLE] Prophet ({alpha*100:.0f}%) + LSTM ({(1-alpha)*100:.0f}%)")
    print(f"  Forecast range: {merged['ds'].min().date()} → {merged['ds'].max().date()}")
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_daily_sales()
    check_stationarity(df["y"])

    prophet_model, prophet_fc, prophet_mape = train_prophet(df)
    lstm_model, lstm_fc, lstm_mape          = train_lstm(df, epochs=30)

    ensemble = ensemble_forecast(prophet_fc, lstm_fc)

    # Save forecasts
    prophet_fc.to_csv(os.path.join(DATA_PROCESSED, "prophet_forecast.csv"), index=False)
    if lstm_fc is not None:
        lstm_fc.to_csv(os.path.join(DATA_PROCESSED, "lstm_forecast.csv"), index=False)
    ensemble.to_csv(os.path.join(DATA_PROCESSED, "ensemble_forecast.csv"), index=False)

    print(f"\n✅ Forecasting complete!")
    print(f"   Prophet MAPE : {prophet_mape:.2f}%")
    print(f"   LSTM MAPE    : {lstm_mape:.2f}%")
    print(f"   Target MAPE  : ≤ {MAPE_TARGET}%")