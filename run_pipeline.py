# run_pipeline.py
# RetailPulse – Master Pipeline Runner
# Runs all steps end-to-end in order

import os
import sys
import time

sys.path.append(os.path.dirname(__file__))

STEPS = [
    ("Data Ingestion & Feature Engineering",
     "src/ingestion/data_loader.py"),
    ("Customer Segmentation",
     "src/segmentation/segmentation.py"),
    ("Demand Forecasting",
     "src/forecasting/forecasting.py"),
    ("Churn Prediction",
     "src/churn/churn_model.py"),
    ("Inventory Optimization",
     "src/inventory/inventory.py"),
    ("Drift Detection",
     "src/monitoring/drift_detection.py"),
]


def run_step(name: str, script: str) -> bool:
    print(f"\n{'='*60}")
    print(f"  ▶  {name}")
    print(f"{'='*60}")
    start = time.time()
    ret = os.system(f"python {script}")
    elapsed = time.time() - start
    if ret == 0:
        print(f"  ✅ Completed in {elapsed:.1f}s")
        return True
    else:
        print(f"  ❌ FAILED (exit code {ret})")
        return False


if __name__ == "__main__":
    print("\n🚀 RetailPulse – Full Pipeline Starting …\n")
    start_total = time.time()
    failed = []

    for name, script in STEPS:
        ok = run_step(name, script)
        if not ok:
            failed.append(name)

    total = time.time() - start_total
    print(f"\n{'='*60}")
    print(f"  Pipeline complete in {total:.1f}s")
    if failed:
        print(f"  ❌ Failed steps: {', '.join(failed)}")
    else:
        print("  ✅ All steps succeeded!")
    print(f"{'='*60}")
    print("\n  Launch dashboard with:")
    print("  streamlit run src/dashboard/app.py\n")