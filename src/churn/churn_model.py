# src/churn/churn_model.py
# RetailPulse – Churn Prediction (XGBoost + SHAP)

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, classification_report,
                             confusion_matrix, precision_recall_curve,
                             RocCurveDisplay)
from sklearn.pipeline import Pipeline
import xgboost as xgb
import shap
import mlflow
import mlflow.xgboost
import joblib
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from config import *

# Force local MLflow tracking
mlflow.set_tracking_uri("sqlite:///C:/Users/Lenovo/PyCharmMiscProject/mlflow.db")
mlflow.set_experiment("churn_prediction")


# ─────────────────────────────────────────────────────────────────────────────
# 1. BUILD CHURN FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def build_churn_features(cleaned_csv: str = CLEANED_FILE,
                          rfm_csv: str = RFM_FILE) -> pd.DataFrame:
    print("[CHURN] Building churn features …")
    df  = pd.read_csv(cleaned_csv, parse_dates=["InvoiceDate"])
    rfm = pd.read_csv(rfm_csv)

    snapshot   = df["InvoiceDate"].max()
    cutoff     = snapshot - pd.Timedelta(days=CHURN_DAYS)
    train_df   = df[df["InvoiceDate"] <= cutoff]
    holdout_df = df[df["InvoiceDate"] > cutoff]

    active_customers = holdout_df["CustomerID"].unique()

    feats = train_df.groupby("CustomerID").agg(
        total_revenue     = ("TotalAmount", "sum"),
        avg_order_value   = ("TotalAmount", "mean"),
        order_count       = ("InvoiceNo", "nunique"),
        unique_products   = ("StockCode", "nunique"),
        avg_qty           = ("Quantity", "mean"),
        std_qty           = ("Quantity", "std"),
        recency_days      = ("InvoiceDate", lambda x: (cutoff - x.max()).days),
        tenure_days       = ("InvoiceDate", lambda x: (x.max() - x.min()).days),
        purchase_freq_30d = ("InvoiceDate",
                             lambda x: ((x >= cutoff - pd.Timedelta(days=30)) &
                                        (x <= cutoff)).sum()),
    ).reset_index().fillna(0)

    feats = feats.merge(rfm[["CustomerID", "R_Score", "F_Score",
                               "M_Score", "RFM_Score"]], on="CustomerID", how="left")

    feats["churned"] = (~feats["CustomerID"].isin(active_customers)).astype(int)

    churn_rate = feats["churned"].mean() * 100
    print(f"  → Customers: {len(feats):,} | Churned: {feats['churned'].sum():,} "
          f"({churn_rate:.1f}%)")

    out_path = os.path.join(DATA_PROCESSED, "churn_features.csv")
    feats.to_csv(out_path, index=False)
    return feats


# ─────────────────────────────────────────────────────────────────────────────
# 2. OPTUNA HYPERPARAMETER TUNING
# ─────────────────────────────────────────────────────────────────────────────

def tune_xgboost(X_train, y_train, n_trials: int = 30) -> dict:
    print(f"\n[CHURN] Tuning XGBoost with Optuna ({n_trials} trials) …")

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
            "use_label_encoder": False,
            "eval_metric": "auc",
            "random_state": RANDOM_STATE,
        }
        model = xgb.XGBClassifier(**params)
        scores = cross_val_score(model, X_train, y_train,
                                  cv=StratifiedKFold(3), scoring="roc_auc")
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"  → Best AUC-ROC (CV): {study.best_value:.4f}")
    print(f"  → Best params: {study.best_params}")
    return study.best_params


# ─────────────────────────────────────────────────────────────────────────────
# 3. TRAIN XGBOOST CHURN MODEL
# ─────────────────────────────────────────────────────────────────────────────

def train_churn_model(feats: pd.DataFrame, tune: bool = False):
    feature_cols = [c for c in feats.columns
                    if c not in ["CustomerID", "churned"]]
    X = feats[feature_cols].fillna(0)
    y = feats["churned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    print(f"\n[CHURN] Training split: {X_train.shape[0]} train | {X_test.shape[0]} test")

    if tune and len(X_train) > 200:
        best_params = tune_xgboost(X_train, y_train, n_trials=20)
    else:
        best_params = {
            "n_estimators": 300, "max_depth": 5, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8,
        }

    with mlflow.start_run(run_name="XGBoost_Churn"):
        model = xgb.XGBClassifier(
            **best_params,
            use_label_encoder=False,
            eval_metric="auc",
            random_state=RANDOM_STATE,
        )
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  verbose=False)

        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred       = (y_pred_proba >= 0.5).astype(int)

        auc  = roc_auc_score(y_test, y_pred_proba)
        prec_at_20 = _precision_at_k(y_test, y_pred_proba, k=0.20)

        print(f"\n[CHURN] Results:")
        print(f"  AUC-ROC        : {auc:.4f}  (target ≥ {AUC_TARGET})")
        print(f"  Precision@20%  : {prec_at_20:.4f}  (target ≥ 0.75)")
        print(f"\n{classification_report(y_test, y_pred, target_names=['Active','Churned'])}")

        mlflow.log_params(best_params)
        mlflow.log_metric("AUC_ROC", auc)
        mlflow.log_metric("Precision_at_20pct", prec_at_20)
        mlflow.xgboost.log_model(model, "xgboost_churn_model")

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODELS_DIR, "churn_model.pkl"))
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "churn_features.pkl"))

    _plot_roc(model, X_test, y_test, auc)
    _plot_shap(model, X_train, feature_cols)
    _plot_confusion(y_test, y_pred)

    return model, feature_cols, auc


def _precision_at_k(y_true, y_proba, k: float = 0.20) -> float:
    n = max(1, int(len(y_true) * k))
    top_idx = np.argsort(y_proba)[::-1][:n]
    return y_true.iloc[top_idx].mean()


def _plot_roc(model, X_test, y_test, auc):
    fig, ax = plt.subplots(figsize=(7, 5))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.set_title(f"ROC Curve | AUC = {auc:.4f}")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "roc_curve.png"), dpi=150)
    plt.close()


def _plot_shap(model, X_train, feature_cols):
    print("\n[CHURN] Computing SHAP values …")
    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(X_train)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals, X_train, feature_names=feature_cols,
                      show=False, plot_size=(10, 6))
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "shap_summary.png"), dpi=150)
    plt.close()
    print("  → SHAP summary plot saved")


def _plot_confusion(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Active", "Churned"])
    ax.set_yticklabels(["Active", "Churned"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 4. SCORE ALL CUSTOMERS
# ─────────────────────────────────────────────────────────────────────────────

def score_all_customers(feats: pd.DataFrame, model, feature_cols) -> pd.DataFrame:
    X = feats[feature_cols].fillna(0)
    feats["churn_probability"] = model.predict_proba(X)[:, 1]
    feats["churn_risk_label"] = pd.cut(feats["churn_probability"],
                                        bins=[0, 0.3, 0.6, 1.0],
                                        labels=["Low", "Medium", "High"])
    out = os.path.join(DATA_PROCESSED, "churn_scores.csv")
    feats[["CustomerID", "churn_probability", "churn_risk_label",
           "churned"]].to_csv(out, index=False)
    print(f"\n[CHURN] Risk distribution:\n{feats['churn_risk_label'].value_counts()}")
    return feats


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    feats = build_churn_features()
    model, feature_cols, auc = train_churn_model(feats, tune=True)
    feats = score_all_customers(feats, model, feature_cols)
    print(f"\n✅ Churn prediction complete! AUC-ROC = {auc:.4f}")