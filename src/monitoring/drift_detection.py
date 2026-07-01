# src/monitoring/drift_detection.py
# RetailPulse – Drift Detection with Evidently AI

import os
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from config import *


def run_drift_detection():
    """
    Run data drift detection comparing reference vs current data windows.
    Uses Evidently AI if available, else falls back to statistical tests.
    """
    print("[DRIFT] Running drift detection …")

    churn_path = os.path.join(DATA_PROCESSED, "churn_features.csv")
    if not os.path.exists(churn_path):
        print("[DRIFT] churn_features.csv not found – run churn_model.py first")
        return

    df = pd.read_csv(churn_path)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ["CustomerID", "churned"]]

    # Split into reference (first 70%) and current (last 30%)
    split = int(len(df) * 0.7)
    reference = df.iloc[:split][numeric_cols]
    current   = df.iloc[split:][numeric_cols]

    # Try Evidently
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
        from evidently.metrics import DatasetDriftMetric

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference, current_data=current)

        os.makedirs(REPORTS_DIR, exist_ok=True)
        out_html = os.path.join(REPORTS_DIR, "drift_report.html")
        report.save_html(out_html)
        print(f"[DRIFT] Evidently report saved → {out_html}")

        result = report.as_dict()
        drift_share = result["metrics"][0]["result"].get("share_of_drifted_columns", 0)
        print(f"[DRIFT] Drifted features: {drift_share*100:.1f}%")
        return drift_share

    except ImportError:
        print("[DRIFT] Evidently not installed – using KS test fallback")

    # Fallback: KS Test
    from scipy.stats import ks_2samp

    results = []
    for col in numeric_cols:
        stat, pval = ks_2samp(reference[col].dropna(), current[col].dropna())
        drifted = pval < 0.05
        results.append({
            "feature": col,
            "ks_statistic": round(stat, 4),
            "p_value": round(pval, 4),
            "drifted": drifted
        })

    res_df = pd.DataFrame(results)
    drifted_features = res_df[res_df["drifted"]]
    print(f"\n[DRIFT] KS Test Results:")
    print(res_df.to_string(index=False))
    print(f"\n[DRIFT] Drifted features: {len(drifted_features)}/{len(res_df)}")

    # Save
    res_df.to_csv(os.path.join(REPORTS_DIR, "drift_ks_results.csv"), index=False)

    if len(drifted_features) > len(res_df) * 0.3:
        print("\n[ALERT] ⚠️  >30% features drifted – retraining recommended!")
    return len(drifted_features) / len(res_df)


if __name__ == "__main__":
    run_drift_detection()
    print("\n✅ Drift detection complete!")