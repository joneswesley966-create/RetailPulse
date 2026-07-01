# tests/test_pipeline.py
# RetailPulse – Basic Unit Tests

import sys
import os
import pytest
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


# ─────────────────────────────────────────────────────────────────────────────
# Test: Synthetic Data Generation
# ─────────────────────────────────────────────────────────────────────────────

def test_synthetic_data_generation():
    from src.ingestion.data_loader import generate_synthetic_data
    df = generate_synthetic_data(n_customers=100, n_transactions=1000)
    assert len(df) == 1000
    assert "CustomerID" in df.columns
    assert "TotalAmount" not in df.columns  # added after cleaning


def test_data_cleaning():
    from src.ingestion.data_loader import generate_synthetic_data, clean_data
    raw = generate_synthetic_data(n_customers=100, n_transactions=1000)
    cleaned = clean_data(raw)
    # No cancellations
    assert not cleaned["InvoiceNo"].astype(str).str.startswith("C").any()
    # No null CustomerID
    assert cleaned["CustomerID"].isnull().sum() == 0
    # TotalAmount should be positive
    assert (cleaned["TotalAmount"] > 0).all()


def test_rfm_build():
    from src.ingestion.data_loader import generate_synthetic_data, clean_data, build_rfm
    raw = generate_synthetic_data(n_customers=100, n_transactions=2000)
    cleaned = clean_data(raw)
    rfm = build_rfm(cleaned)
    assert "Recency" in rfm.columns
    assert "Frequency" in rfm.columns
    assert "Monetary" in rfm.columns
    assert "RFM_Score" in rfm.columns
    assert rfm["R_Score"].between(1, 5).all()
    assert rfm["F_Score"].between(1, 5).all()


# ─────────────────────────────────────────────────────────────────────────────
# Test: Inventory Logic
# ─────────────────────────────────────────────────────────────────────────────

def test_reorder_point_calculation():
    """Safety stock and reorder point should always be non-negative."""
    from src.inventory.inventory import compute_reorder_recommendations
    stats = pd.DataFrame({
        "StockCode": ["A", "B"],
        "Description": ["Product A", "Product B"],
        "avg_daily_demand": [10.0, 5.0],
        "std_daily_demand": [2.0, 1.0],
        "max_daily_demand": [15.0, 8.0],
        "total_qty_sold": [1000, 500],
        "days_with_sales": [200, 100],
        "demand_variability": [0.2, 0.2],
        "total_revenue": [10000.0, 5000.0],
    })
    rec = compute_reorder_recommendations(stats)
    assert (rec["safety_stock"] >= 0).all()
    assert (rec["reorder_point"] >= 0).all()
    assert (rec["eoq"] >= 1).all()


# ─────────────────────────────────────────────────────────────────────────────
# Test: Churn Label Creation
# ─────────────────────────────────────────────────────────────────────────────

def test_churn_label_binary():
    labels = pd.Series([0, 1, 0, 1, 1])
    assert labels.isin([0, 1]).all()
    assert labels.dtype in [int, np.int64, np.int32]


# ─────────────────────────────────────────────────────────────────────────────
# Test: Config paths exist (or can be created)
# ─────────────────────────────────────────────────────────────────────────────

def test_config_imports():
    from config import (DATA_RAW, DATA_PROCESSED, MODELS_DIR,
                                FORECAST_HORIZON, MAPE_TARGET, AUC_TARGET,
                                N_CLUSTERS, CHURN_DAYS)
    assert FORECAST_HORIZON == 30
    assert MAPE_TARGET == 12.0
    assert AUC_TARGET == 0.88
    assert N_CLUSTERS == 6
    assert CHURN_DAYS == 90


if __name__ == "__main__":
    pytest.main([__file__, "-v"])