# src/ingestion/data_loader.py
# RetailPulse – Data Ingestion, Cleaning & Feature Engineering

import os
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from config import *

# ─────────────────────────────────────────────────────────────────────────────
# 1. DOWNLOAD / LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_raw_data(filepath: str = RAW_FILE) -> pd.DataFrame:
    """
    Load the Online Retail dataset.
    Download from: https://archive.ics.uci.edu/dataset/352/online+retail
    Save as data/raw/online_retail.csv
    """
    print(f"[INFO] Loading raw data from {filepath}")
    df = pd.read_csv(filepath, encoding="ISO-8859-1")
    print(f"[INFO] Raw shape: {df.shape}")
    print(df.head(3))
    return df


def generate_synthetic_data(n_customers=1000, n_transactions=50000,
                             seed=42) -> pd.DataFrame:
    """
    Generate synthetic retail data so you can run everything
    before downloading the real dataset.
    """
    print("[INFO] Generating synthetic retail data …")
    np.random.seed(seed)
    rng = pd.date_range("2022-01-01", "2024-12-31", freq="D")

    invoice_nos = [f"INV{str(i).zfill(6)}" for i in range(1, n_transactions + 1)]
    customer_ids = np.random.randint(10000, 10000 + n_customers, n_transactions)
    stock_codes = [f"SC{str(np.random.randint(1000, 9999))}" for _ in range(n_transactions)]
    descriptions = np.random.choice(
        ["Widget A", "Widget B", "Gadget X", "Tool Y",
         "Product Z", "Item Alpha", "Item Beta"], n_transactions)
    quantities = np.random.randint(1, 50, n_transactions)
    unit_prices = np.round(np.random.uniform(0.5, 50.0, n_transactions), 2)
    invoice_dates = np.random.choice(rng, n_transactions)
    countries = np.random.choice(
        ["United Kingdom", "Germany", "France", "Spain", "Australia"],
        n_transactions, p=[0.7, 0.1, 0.1, 0.05, 0.05])

    df = pd.DataFrame({
        "InvoiceNo": invoice_nos,
        "StockCode": stock_codes,
        "Description": descriptions,
        "Quantity": quantities,
        "InvoiceDate": pd.to_datetime(invoice_dates),
        "UnitPrice": unit_prices,
        "CustomerID": customer_ids.astype(float),
        "Country": countries,
    })

    # Inject ~5% cancellations (negative qty, invoice starts with C)
    cancel_idx = np.random.choice(df.index, int(n_transactions * 0.05), replace=False)
    df.loc[cancel_idx, "InvoiceNo"] = "C" + df.loc[cancel_idx, "InvoiceNo"]
    df.loc[cancel_idx, "Quantity"] = -df.loc[cancel_idx, "Quantity"]

    # Inject ~2% missing CustomerID
    missing_idx = np.random.choice(df.index, int(n_transactions * 0.02), replace=False)
    df.loc[missing_idx, "CustomerID"] = np.nan

    print(f"[INFO] Synthetic data shape: {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. CLEANING
# ─────────────────────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline:
    - Remove cancellations
    - Drop nulls in CustomerID
    - Remove duplicates
    - Remove invalid Quantity / UnitPrice
    - Parse dates
    - Add TotalAmount
    """
    print("\n[CLEAN] Starting cleaning …")
    original_rows = len(df)

    # Ensure date column is datetime
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # Remove cancellations (InvoiceNo starts with 'C')
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    print(f"  → After removing cancellations: {len(df):,} rows")

    # Drop missing CustomerID
    df = df.dropna(subset=["CustomerID"])
    print(f"  → After dropping null CustomerID: {len(df):,} rows")

    # Remove duplicates
    df = df.drop_duplicates()
    print(f"  → After dropping duplicates: {len(df):,} rows")

    # Remove invalid Quantity and UnitPrice
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    print(f"  → After removing invalid Qty/Price: {len(df):,} rows")

    # Cast CustomerID
    df["CustomerID"] = df["CustomerID"].astype(int)

    # Add derived columns
    df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]
    df["Year"] = df["InvoiceDate"].dt.year
    df["Month"] = df["InvoiceDate"].dt.month
    df["DayOfWeek"] = df["InvoiceDate"].dt.dayofweek
    df["Date"] = df["InvoiceDate"].dt.date

    removed = original_rows - len(df)
    print(f"[CLEAN] Done. Removed {removed:,} rows ({removed/original_rows*100:.1f}%)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING – RFM SCORES
# ─────────────────────────────────────────────────────────────────────────────

def build_rfm(df: pd.DataFrame, snapshot_date=None) -> pd.DataFrame:
    """
    Build RFM (Recency, Frequency, Monetary) features per customer.
    """
    print("\n[RFM] Building RFM features …")

    if snapshot_date is None:
        snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalAmount", "sum"),
    ).reset_index()

    # Score 1–5 (higher = better for F and M; lower recency = better)
    rfm["R_Score"] = pd.qcut(rfm["Recency"], q=5,
                              labels=[5, 4, 3, 2, 1], duplicates="drop").astype(int)
    rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), q=5,
                              labels=[1, 2, 3, 4, 5], duplicates="drop").astype(int)
    rfm["M_Score"] = pd.qcut(rfm["Monetary"].rank(method="first"), q=5,
                              labels=[1, 2, 3, 4, 5], duplicates="drop").astype(int)

    rfm["RFM_Score"] = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]

    # Segment label
    def segment(row):
        if row["RFM_Score"] >= 13:
            return "Champions"
        elif row["R_Score"] >= 4 and row["F_Score"] >= 3:
            return "Loyal Customers"
        elif row["R_Score"] >= 3 and row["F_Score"] == 1:
            return "Potential Loyalists"
        elif row["R_Score"] <= 2 and row["F_Score"] >= 3:
            return "At-Risk"
        elif row["R_Score"] == 1 and row["F_Score"] == 1:
            return "Lost"
        else:
            return "Others"

    rfm["Segment"] = rfm.apply(segment, axis=1)
    print(f"[RFM] Segments:\n{rfm['Segment'].value_counts()}")
    return rfm


# ─────────────────────────────────────────────────────────────────────────────
# 4. ROLLING STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

def build_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling sales statistics for time-series modeling.
    """
    print("\n[ROLLING] Building rolling features …")
    daily = (df.groupby("Date")["TotalAmount"]
               .sum()
               .reset_index()
               .rename(columns={"Date": "ds", "TotalAmount": "y"}))
    daily["ds"] = pd.to_datetime(daily["ds"])
    daily = daily.sort_values("ds").reset_index(drop=True)

    daily["rolling_7d_mean"]  = daily["y"].rolling(7, min_periods=1).mean()
    daily["rolling_30d_mean"] = daily["y"].rolling(30, min_periods=1).mean()
    daily["rolling_7d_std"]   = daily["y"].rolling(7, min_periods=1).std().fillna(0)
    daily["lag_1"]  = daily["y"].shift(1).fillna(method="bfill")
    daily["lag_7"]  = daily["y"].shift(7).fillna(method="bfill")
    daily["lag_30"] = daily["y"].shift(30).fillna(method="bfill")

    print(f"[ROLLING] Daily sales shape: {daily.shape}")
    return daily


# ─────────────────────────────────────────────────────────────────────────────
# 5. SAVE
# ─────────────────────────────────────────────────────────────────────────────

def save_processed(df: pd.DataFrame, rfm: pd.DataFrame,
                   daily: pd.DataFrame) -> None:
    os.makedirs(DATA_PROCESSED, exist_ok=True)
    df.to_csv(CLEANED_FILE, index=False)
    rfm.to_csv(RFM_FILE, index=False)
    daily.to_csv(FORECAST_FILE, index=False)
    print(f"\n[SAVE] Files saved to {DATA_PROCESSED}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(DATA_RAW, exist_ok=True)

    # Use synthetic data (swap with load_raw_data() once you have the CSV)
    raw = generate_synthetic_data()

    cleaned = clean_data(raw)
    rfm     = build_rfm(cleaned)
    daily   = build_rolling_features(cleaned)

    save_processed(cleaned, rfm, daily)
    print("\n✅ Data ingestion & feature engineering complete!")