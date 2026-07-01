# src/inventory/inventory.py
# RetailPulse – Inventory Optimization

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from config import *


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    cleaned = pd.read_csv(CLEANED_FILE, parse_dates=["InvoiceDate"])

    forecast_path = os.path.join(DATA_PROCESSED, "ensemble_forecast.csv")
    if not os.path.exists(forecast_path):
        forecast_path = os.path.join(DATA_PROCESSED, "prophet_forecast.csv")

    forecast = pd.read_csv(forecast_path, parse_dates=["ds"])
    print(f"[INV] Cleaned: {cleaned.shape} | Forecast: {forecast.shape}")
    return cleaned, forecast


# ─────────────────────────────────────────────────────────────────────────────
# 2. PER-PRODUCT DEMAND STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_product_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean and std of daily demand per product (StockCode).
    """
    # Group to daily product sales
    daily = (df.groupby(["StockCode", "Description",
                          pd.Grouper(key="InvoiceDate", freq="D")])
               ["Quantity"].sum()
               .reset_index()
               .rename(columns={"Quantity": "daily_qty"}))

    stats = daily.groupby(["StockCode", "Description"]).agg(
        avg_daily_demand=("daily_qty", "mean"),
        std_daily_demand=("daily_qty", "std"),
        max_daily_demand=("daily_qty", "max"),
        total_qty_sold  =("daily_qty", "sum"),
        days_with_sales =("daily_qty", lambda x: (x > 0).sum()),
    ).reset_index().fillna(0)

    stats["demand_variability"] = (stats["std_daily_demand"] /
                                    (stats["avg_daily_demand"] + 1e-8))

    # Revenue per product
    revenue = df.groupby("StockCode")["TotalAmount"].sum().reset_index()
    revenue.columns = ["StockCode", "total_revenue"]
    stats = stats.merge(revenue, on="StockCode", how="left")

    print(f"[INV] Product stats computed: {len(stats)} products")
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# 3. REORDER QUANTITY FORMULA
# ─────────────────────────────────────────────────────────────────────────────

def compute_reorder_recommendations(stats: pd.DataFrame,
                                     lead_time: int = LEAD_TIME_DAYS,
                                     safety_mult: float = SAFETY_STOCK_MULTIPLIER,
                                     service_level_z: float = 1.65  # 95% service level
                                     ) -> pd.DataFrame:
    """
    Economic Reorder Point (ROP) and Reorder Quantity (ROQ):

    Safety Stock  = Z × σ_daily × √lead_time
    ROP           = μ_daily × lead_time + Safety_Stock
    EOQ           = √(2 × D × S / H)  [simplified with S=ordering cost, H=holding cost]
    """
    print(f"\n[INV] Computing reorder recommendations …")
    print(f"  Lead time    : {lead_time} days")
    print(f"  Service level: 95% (Z={service_level_z})")

    rec = stats.copy()

    # Safety stock
    rec["safety_stock"] = (service_level_z *
                            rec["std_daily_demand"] *
                            np.sqrt(lead_time)).round(0)

    # Reorder Point
    rec["reorder_point"] = (rec["avg_daily_demand"] * lead_time +
                             rec["safety_stock"]).round(0)

    # Reorder Quantity (EOQ approximation)
    # Assume ordering cost S=10, holding cost H=0.2 per unit per day
    S = 10; H = 0.20
    D = rec["avg_daily_demand"] * 365   # Annual demand
    rec["eoq"] = np.sqrt(2 * D * S / (H + 1e-8)).round(0)
    rec["eoq"] = rec["eoq"].clip(lower=1)

    # Stock status (simulated current stock = 30-day demand × 1.2 for demo)
    rec["simulated_current_stock"] = (rec["avg_daily_demand"] * 30 * 1.2).round(0)
    rec["stock_status"] = rec.apply(
        lambda r: "🔴 Reorder Now" if r["simulated_current_stock"] <= r["reorder_point"]
                  else ("🟡 Monitor" if r["simulated_current_stock"] <= r["reorder_point"] * 1.5
                        else "🟢 OK"), axis=1)

    # Priority rank by revenue × demand variability
    rec["priority_score"] = (rec["total_revenue"] *
                              (1 + rec["demand_variability"])).rank(ascending=False)
    rec = rec.sort_values("priority_score")

    print(f"\n  Stock Status Summary:")
    print(rec["stock_status"].value_counts().to_string())

    # Save
    out = os.path.join(DATA_PROCESSED, "inventory_recommendations.csv")
    rec.to_csv(out, index=False)
    print(f"\n[INV] Saved → {out}")
    return rec


# ─────────────────────────────────────────────────────────────────────────────
# 4. VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────

def plot_inventory_insights(rec: pd.DataFrame) -> None:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    top20 = rec.head(20)

    # Top 20 products by priority
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Reorder points
    axes[0].barh(top20["Description"].str[:20],
                 top20["reorder_point"], color="steelblue")
    axes[0].set_title("Top 20 Products – Reorder Point")
    axes[0].set_xlabel("Units")
    axes[0].invert_yaxis()

    # Safety stock
    axes[1].barh(top20["Description"].str[:20],
                 top20["safety_stock"], color="coral")
    axes[1].set_title("Top 20 Products – Safety Stock")
    axes[1].set_xlabel("Units")
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "inventory_recommendations.png"), dpi=150)
    plt.close()

    # Stock status pie
    status_counts = rec["stock_status"].value_counts()
    colors = {"🔴 Reorder Now": "#e74c3c",
              "🟡 Monitor": "#f39c12",
              "🟢 OK": "#27ae60"}
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(status_counts.values,
           labels=status_counts.index,
           colors=[colors.get(k, "gray") for k in status_counts.index],
           autopct="%1.1f%%", startangle=140)
    ax.set_title("Inventory Stock Status Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "stock_status_pie.png"), dpi=150)
    plt.close()
    print("[INV] Plots saved")


# ─────────────────────────────────────────────────────────────────────────────
# 5. WHAT-IF ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def what_if_analysis(rec: pd.DataFrame,
                      demand_change_pct: float = 20.0,
                      lead_time_change: int = 3) -> pd.DataFrame:
    """
    Simulate how recommendations change under different demand/lead time scenarios.
    """
    adj = rec.copy()
    multiplier = 1 + demand_change_pct / 100
    new_lead   = LEAD_TIME_DAYS + lead_time_change

    adj["adj_avg_daily"] = adj["avg_daily_demand"] * multiplier
    adj["adj_safety_stock"] = (1.65 * adj["std_daily_demand"] *
                                np.sqrt(new_lead)).round(0)
    adj["adj_reorder_point"] = (adj["adj_avg_daily"] * new_lead +
                                 adj["adj_safety_stock"]).round(0)

    print(f"\n[WHAT-IF] Demand +{demand_change_pct}%, Lead time +{lead_time_change}d:")
    delta = ((adj["adj_reorder_point"] - adj["reorder_point"]) /
              (adj["reorder_point"] + 1e-8) * 100).mean()
    print(f"  Avg reorder point increase: {delta:.1f}%")
    return adj


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cleaned, forecast = load_data()
    stats = compute_product_stats(cleaned)
    rec   = compute_reorder_recommendations(stats)
    plot_inventory_insights(rec)
    what_if = what_if_analysis(rec, demand_change_pct=20, lead_time_change=3)
    print("\n✅ Inventory optimization complete!")