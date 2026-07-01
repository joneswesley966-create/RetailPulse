# src/segmentation/segmentation.py
# RetailPulse – Customer Segmentation (K-Means + DBSCAN)

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import mlflow
import mlflow.sklearn
import joblib

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from config import *

# Force local MLflow tracking (overrides config)
EXPERIMENT_NAME = "customer_segmentation"
mlflow.set_tracking_uri("sqlite:///C:/Users/Lenovo/PyCharmMiscProject/mlflow.db")
mlflow.set_experiment(EXPERIMENT_NAME)


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD & PREPROCESS
# ─────────────────────────────────────────────────────────────────────────────

def load_rfm(filepath: str = RFM_FILE) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    print(f"[SEG] Loaded RFM: {df.shape}")
    return df


def preprocess_rfm(rfm: pd.DataFrame):
    features = ["Recency", "Frequency", "Monetary"]
    X = rfm[features].copy()

    # Log-transform monetary & frequency to reduce skew
    X["Frequency"] = np.log1p(X["Frequency"])
    X["Monetary"]  = np.log1p(X["Monetary"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler, features


# ─────────────────────────────────────────────────────────────────────────────
# 2. ELBOW + SILHOUETTE TO FIND BEST K
# ─────────────────────────────────────────────────────────────────────────────

def find_optimal_k(X_scaled, k_range=range(2, 11)) -> int:
    inertias, silhouettes = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(list(k_range), inertias, "bo-")
    axes[0].set_title("Elbow Method"); axes[0].set_xlabel("K"); axes[0].set_ylabel("Inertia")
    axes[1].plot(list(k_range), silhouettes, "rs-")
    axes[1].set_title("Silhouette Score"); axes[1].set_xlabel("K"); axes[1].set_ylabel("Score")
    plt.tight_layout()
    os.makedirs(REPORTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(REPORTS_DIR, "elbow_silhouette.png"), dpi=150)
    plt.close()

    best_k = list(k_range)[int(np.argmax(silhouettes))]
    print(f"[SEG] Best K by silhouette: {best_k} (score={max(silhouettes):.4f})")
    return best_k


# ─────────────────────────────────────────────────────────────────────────────
# 3. K-MEANS CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────

def run_kmeans(X_scaled, rfm: pd.DataFrame, k: int) -> pd.DataFrame:
    print(f"\n[SEG] Running K-Means with k={k} …")

    with mlflow.start_run(run_name="KMeans_Segmentation"):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10, max_iter=300)
        labels = km.fit_predict(X_scaled)

        sil = silhouette_score(X_scaled, labels)
        mlflow.log_param("n_clusters", k)
        mlflow.log_metric("silhouette_score", sil)
        mlflow.sklearn.log_model(km, "kmeans_model")

        print(f"  → Silhouette Score: {sil:.4f}")

    rfm["KMeans_Cluster"] = labels

    # Save model
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(km, os.path.join(MODELS_DIR, "kmeans.pkl"))
    return rfm


# ─────────────────────────────────────────────────────────────────────────────
# 4. DBSCAN (noise / outlier detection)
# ─────────────────────────────────────────────────────────────────────────────

def run_dbscan(X_scaled, rfm: pd.DataFrame) -> pd.DataFrame:
    print("\n[SEG] Running DBSCAN …")
    db = DBSCAN(eps=0.5, min_samples=5)
    rfm["DBSCAN_Cluster"] = db.fit_predict(X_scaled)
    noise = (rfm["DBSCAN_Cluster"] == -1).sum()
    print(f"  → DBSCAN clusters: {rfm['DBSCAN_Cluster'].nunique() - 1}, "
          f"noise points: {noise}")
    return rfm


# ─────────────────────────────────────────────────────────────────────────────
# 5. PCA VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_clusters(X_scaled, labels, title="KMeans Clusters") -> None:
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)
    plt.figure(figsize=(9, 6))
    scatter = plt.scatter(coords[:, 0], coords[:, 1],
                          c=labels, cmap="tab10", alpha=0.6, s=20)
    plt.colorbar(scatter, label="Cluster")
    plt.title(title)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.tight_layout()
    fname = title.replace(" ", "_").lower() + ".png"
    plt.savefig(os.path.join(REPORTS_DIR, fname), dpi=150)
    plt.close()
    print(f"  → Plot saved: {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. CLUSTER BUSINESS INTERPRETATION
# ─────────────────────────────────────────────────────────────────────────────

def interpret_clusters(rfm: pd.DataFrame) -> pd.DataFrame:
    summary = rfm.groupby("KMeans_Cluster").agg(
        Count=("CustomerID", "count"),
        Avg_Recency=("Recency", "mean"),
        Avg_Frequency=("Frequency", "mean"),
        Avg_Monetary=("Monetary", "mean"),
        Avg_RFM_Score=("RFM_Score", "mean"),
    ).round(2)

    # Auto-label clusters by RFM score
    labels_map = {}
    sorted_clusters = summary["Avg_RFM_Score"].sort_values(ascending=False).index
    names = ["Champions", "Loyal Customers", "At-Risk Customers",
             "Potential Loyalists", "Lost Customers", "Others"]
    for i, cluster in enumerate(sorted_clusters):
        labels_map[cluster] = names[i] if i < len(names) else f"Cluster {cluster}"

    summary["Business_Label"] = summary.index.map(labels_map)
    rfm["Business_Segment"] = rfm["KMeans_Cluster"].map(labels_map)

    print("\n[SEG] Cluster Business Interpretation:")
    print(summary.to_string())
    return rfm, summary


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rfm = load_rfm()
    X_scaled, scaler, features = preprocess_rfm(rfm)

    best_k = find_optimal_k(X_scaled)
    rfm    = run_kmeans(X_scaled, rfm, k=best_k)
    rfm    = run_dbscan(X_scaled, rfm)

    plot_clusters(X_scaled, rfm["KMeans_Cluster"].values, "KMeans Clusters")
    plot_clusters(X_scaled, rfm["DBSCAN_Cluster"].values, "DBSCAN Clusters")

    rfm, cluster_summary = interpret_clusters(rfm)

    # Save
    out = os.path.join(DATA_PROCESSED, "segmented_customers.csv")
    rfm.to_csv(out, index=False)
    cluster_summary.to_csv(os.path.join(DATA_PROCESSED, "cluster_summary.csv"))
    print(f"\n✅ Segmentation complete! Saved to {out}")

