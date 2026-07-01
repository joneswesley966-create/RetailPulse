# 📊 RetailPulse – AI-Powered Customer Analytics & Demand Forecasting

> An end-to-end retail analytics platform built during the Zidio Development Data Science Internship (June–September 2026).

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)
![MLflow](https://img.shields.io/badge/MLflow-2.13-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green)
![Prophet](https://img.shields.io/badge/Prophet-1.1-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🚀 Overview

RetailPulse is a production-ready analytics platform that combines machine learning, time-series forecasting, and interactive dashboards to help retail businesses make data-driven decisions.

### Key Capabilities
- **Customer Segmentation** — RFM + K-Means / DBSCAN clustering
- **Demand Forecasting** — Prophet + LSTM ensemble (30-day horizon)
- **Churn Prediction** — XGBoost + SHAP explainability
- **Inventory Optimization** — EOQ, safety stock, reorder recommendations
- **Drift Detection** — Evidently AI / KS Test monitoring
- **Interactive Dashboard** — 4-panel Streamlit app with What-If controls

---

## 🗂 Project Structure

```
RetailPulse/
│
├── src/
│   ├── ingestion/          # Data loading, cleaning, RFM
│   ├── segmentation/       # K-Means + DBSCAN clustering
│   ├── forecasting/        # Prophet + LSTM forecasting
│   ├── churn/              # XGBoost churn model
│   ├── inventory/          # EOQ inventory optimization
│   ├── monitoring/         # Drift detection
│   └── dashboard/          # Streamlit multi-page app
│
├── config/                 # Global configuration
├── data/
│   ├── raw/                # Raw input data
│   └── processed/          # Model outputs
├── models/                 # Saved ML models (.pkl, .pt)
├── reports/                # Charts and HTML reports
├── tests/                  # Unit tests (pytest)
│
├── run_pipeline.py         # Master pipeline runner
├── test_pipeline.py        # Unit tests
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/retailpulse.git
cd retailpulse
```

### 2. Create virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🏃 Running the Pipeline

### Run all steps end-to-end
```bash
python run_pipeline.py
```

### Run individual modules
```bash
python src/ingestion/data_loader.py
python src/segmentation/segmentation.py
python src/forecasting/forecasting.py
python src/churn/churn_model.py
python src/inventory/inventory.py
python src/monitoring/drift_detection.py
```

### Launch the dashboard
```bash
streamlit run src/dashboard/app.py
```

---

## 🐳 Docker

### Build and run
```bash
docker build -t retailpulse .
docker run -p 8501:8501 retailpulse
```

Then open: http://localhost:8501

---

## 🧪 Testing

```bash
pytest test_pipeline.py -v
```

Expected: **6/6 tests passing**

---

## 📊 Dashboard Pages

| Page | Description |
|------|-------------|
| 💰 Sales Dashboard | Revenue trends, top products, country breakdown |
| 👥 Customer Dashboard | Segmentation, RFM analysis, churn intelligence |
| 📈 Forecast Dashboard | 30-day demand forecast with What-If controls |
| 📦 Inventory Dashboard | Reorder recommendations, safety stock, EOQ |

---

## 📈 Model Performance Targets

| Model | Metric | Target |
|-------|--------|--------|
| K-Means | Silhouette Score | ≥ 0.35 |
| Prophet + LSTM | MAPE | ≤ 12% |
| XGBoost Churn | AUC-ROC | ≥ 0.88 |

---

## 🛠 Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.11 |
| ML | Scikit-learn, XGBoost, SHAP, Optuna |
| Forecasting | Prophet, PyTorch (LSTM) |
| Tracking | MLflow |
| Dashboard | Streamlit, Plotly |
| Drift | Evidently AI, SciPy |
| Testing | Pytest |
| DevOps | Docker, GitHub Actions |

---

## 👤 Author

**Jones Wesley**  
Data Science & Analytics Intern — Zidio Development (2026)  
📧 [LinkedIn](https://linkedin.com/in/joneswesley)

---

## 📄 License

MIT License — free to use and modify.
