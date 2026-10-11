# Enterprise E-Commerce Analytics & MLOps Platform

[![Platform](https://img.shields.io/badge/Platform-Streamlit%20MLOps-blue)](http://localhost:8501)
[![FastAPI](https://img.shields.io/badge/Inference-FastAPI%20REST-green)](http://localhost:8000/docs)
[![MLflow](https://img.shields.io/badge/Registry-MLflow%20Production-purple)](sqlite:///mlflow.db)
[![Data Warehouse](https://img.shields.io/badge/Warehouse-Kimball%20Star%20Schema-orange)](ecommerce.db)
[![Tests](https://img.shields.io/badge/Tests-6%2F6%20Passed-brightgreen)](ml/test_api.py)

---

## 1. Executive Summary & Architecture

This repository hosts a production-grade **E-Commerce Sales Analytics, Data Engineering, and Customer Churn Prediction MLOps Platform**. The system features a **Pipeline Execution Console**, automated ETL pipelines with Zero Data Loss quarantine segregation, Kimball-style dimensional modeling, leakage-free temporal feature engineering, multi-model cross-validation, MLflow model governance, and containerized FastAPI real-time scoring microservices.

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   ENTERPRISE MLOPS ARCHITECTURE                                  │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                 │
│  [UCI Online Retail (541k)]                                                                     │
│             │                                                                                   │
│             ▼                                                                                   │
│  [Ingestion & Validation] ──(Rejections)──► [Quarantine Store (144k Rows)]                     │
│             │                                 • missing_fields.csv (135k)                       │
│             │ (387k Clean Rows)               • cancelled_invoices.csv (8.9k)                   │
│             ▼                                 • duplicates.csv (5.2k)                           │
│  [Kimball Star Schema Warehouse]              • invalid_price.csv (2 rows)                      │
│     ├── fact_sales (387k facts)                                                                 │
│     ├── dim_customer (4,338 entities)                                                           │
│     ├── dim_product (3,665 SKUs)                                                                │
│     ├── dim_date & dim_geography                                                                │
│     └── sales_mart & customer_mart                                                              │
│             │                                                                                   │
│             ▼                                                                                   │
│  [Temporal Feature Store (4,338 Customers)]                                                     │
│     └── 13 Leakage-Free Behavioral Metrics (Recency, Frequency, Spend, AOV, Cancel Rate, Basket)│
│             │                                                                                   │
│             ▼                                                                                   │
│  [Multi-Model Benchmark & 5-Fold Stratified CV]                                                 │
│     ├── Baseline: Logistic Regression (StandardScaler) [CHAMPION: Recall 78.28%, ROC 0.7252]   │
│     ├── Random Forest Classifier (100 Estimators)                                               │
│     └── Gradient Boosting Classifier (GBDT)                                                     │
│             │                                                                                   │
│             ▼                                                                                   │
│  [MLflow Experiment Tracking & Model Registry]                                                  │
│     └── sqlite:///mlflow.db ──► CustomerChurnClassifier (Stage: Production)                    │
│             │                                                                                   │
│             ▼                                                                                   │
│  [Production Serving & Customer Segment Operations]                                             │
│     ├── FastAPI Microservice (http://127.0.0.1:8000)                                            │
│     ├── Customer Batch Scoring & VIP Risk Playbooks (4,338 scored)                              │
│     └── Pipeline Execution Console (Streamlit http://localhost:8501)                           │
│                                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 10-Page MLOps Operations Workspaces

The platform provides an engineering operations experience: a persistent dark navy sidebar (`#0f172a`), top application bar with live system chips, real-time terminal output, and 10 dedicated functional pages:

| Page # | Navigation Workspace | Core Functionality |
| :--- | :--- | :--- |
| **Page 1** | **Overview** | Platform health indicators, warehouse financial KPIs ($8.89M revenue, 18,536 orders, 4,338 customers), Zero Data Loss funnel, active ML model snapshot, customer risk cohort breakdown, and recent run history feed. |
| **Page 2** | **Pipelines** | Pipeline Monitor: Visual sequence across all 9 pipeline stages, target pipeline selector (`run_mlops.py`, `etl/pipeline.py`), concurrency run locking, live Execution Logs with timestamped output, and Pipeline Run History table. |
| **Page 3** | **Data Quality** | Ingestion accounting (Raw 541k, Clean 387k, Quarantined 144k), quarantine classification cards (`missing_fields.csv`, `cancelled_invoices.csv`, etc.), interactive record explorer with search, and one-click CSV export. |
| **Page 4** | **Warehouse** | Dimensional catalog browser (`fact_sales`, `dim_*`, marts), schema inspector, paginated row browser, and Safe Read-Only SQL Query Studio with template queries and export. |
| **Page 5** | **Analytics** | Executive sales analytics: monthly revenue trajectory, country distributions, top 10 products, and customer RFM segmentation with date and country filters. |
| **Page 6** | **Model Evaluation** | Multi-model benchmarking table (Logistic Regression, Random Forest, GBDT), 5-fold CV results, diagnostic artifact plots (ROC, PR, Confusion Matrix, Feature Importance), feature taxonomy, and retrain trigger. |
| **Page 7** | **Model Registry** | Local SQLite MLflow tracking inspector (`sqlite:///mlflow.db`), run parameters, metrics, registered model `CustomerChurnClassifier` in `Production`, and model lifecycle promotion console. |
| **Page 8** | **Churn Prediction** | **Mode A:** Real customer lookup from warehouse with live scoring and prescriptive retention playbook. **Mode B:** 13-feature what-if simulator with sliders, gauge chart, and REST API payload inspector. |
| **Page 9** | **Risk Analytics** | Cohort analysis across all 4,338 customers, risk tier breakdown (High Risk >65%, Moderate 35–65%, Low <35%), total revenue at risk ($1.43M), and filterable VIP retention target table with CSV export. |
| **Page 10** | **System Health** | Diagnostic matrix checking Python environment, dependency versions, SQLite integrity (`PRAGMA integrity_check`), MLflow status, artifact availability, FastAPI health, and automated PyTest runner. |

---

## 3. Quickstart & How to Run

### Prerequisites
- Python 3.10+
- Windows PowerShell or Command Prompt

### Step 1: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 2: Run End-to-End MLOps Pipeline
Executes temporal feature extraction, model cross-validation, MLflow logging, batch customer scoring, and automated tests in sequence:
```powershell
python run_mlops.py
```

### Step 3: Launch the Platform
You can run services individually or use the one-click launcher:

#### Option A: One-Click Windows Launcher
```powershell
.\run_all.bat
```

#### Option B: Manual Service Launch
In Terminal 1 (Inference API):
```powershell
python -m uvicorn ml.api:app --host 127.0.0.1 --port 8000
```

In Terminal 2 (MLOps Platform Dashboard):
```powershell
python -m streamlit run dashboard/app.py --server.port 8501
```

Access the user interfaces:
- **MLOps Platform Dashboard**: `http://localhost:8501`
- **FastAPI Interactive Docs**: `http://127.0.0.1:8000/docs`
- **FastAPI Health Endpoint**: `http://127.0.0.1:8000/health`

---

## 4. Verification & Automated Test Results

### Automated API Validation (`ml/test_api.py`)
```bash
python ml/test_api.py
```
**Results (6/6 passing):**
- `PASS: Root endpoint test` (HTTP 200)
- `PASS: Health endpoint test` (status: HEALTHY, model_loaded: true)
- `PASS: Model info endpoint test` (champion: Logistic_Regression, stage: Production)
- `PASS: Predict endpoint test` (valid probability & retention playbook)
- `PASS: Batch predict endpoint test` (multi-customer scoring)
- `PASS: Customer lookup endpoint test` (Customer 17850 verified against warehouse)

### 10-Page Integration Verification (`test_dashboard_pages.py`)
```bash
python test_dashboard_pages.py
```
**Results (10/10 verified):**
- `Page 1 (Overview) Data: OK`
- `Page 2 (Pipelines & Task Monitor): OK`
- `Page 3 (Data Engineering & Quality): OK`
- `Page 4 (Data Warehouse Explorer): OK`
- `Page 5 (Business Intelligence): OK`
- `Page 6 (Model Training & Evaluation): OK`
- `Page 7 (MLflow Model Registry): OK`
- `Page 8 (Customer Churn Prediction): OK`
- `Page 9 (Customer Risk Analytics): OK`
- `Page 10 (System Health & Settings): OK`

---

## 5. Machine Learning Governance & Methodology

### Time-Split Temporal Cohort (Leakage-Free)
To avoid lookahead bias and data leakage:
- **Cutoff Date:** September 10, 2011
- **Observation Window:** Orders placed before or on cutoff used to engineer 13 features.
- **Target Churn Window:** 90 days following cutoff. Customers with 0 orders in this window are labeled as churned (`churn = 1`).
- **Base Rate:** 43.0% churn across $N=3,370$ cohort customers.

### Multi-Model Benchmark Comparison

| Model Architecture | Test Accuracy | Precision | Recall (Churn) | F1-Score | ROC-AUC | PR-AUC | 5-Fold CV ROC-AUC | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Logistic Regression (L2)** | **66.91%** | **58.66%** | **78.28%** | **0.6706** | **0.7252** | **0.6111** | **0.7453 ± 0.018** | **✔ PRODUCTION CHAMPION** |
| Random Forest (100 Trees) | 65.13% | 57.26% | 74.83% | 0.6487 | 0.7164 | 0.6133 | 0.7512 ± 0.021 | Candidate |
| Gradient Boosting (GBDT) | 62.61% | 56.60% | 56.21% | 0.5640 | 0.6920 | 0.5776 | 0.7390 ± 0.015 | Candidate |

**Why Logistic Regression is the Production Champion:**
In commercial customer retention, failing to identify an at-risk customer (False Negative) results in permanent customer loss. Logistic Regression achieved the highest **Recall (78.28%)** on detecting churners, capturing 227 out of 290 test churners, while maintaining calibrated probabilities and an interpretable linear decision boundary.

---

## 6. Project Directory Layout

```
.
├── dashboard/
│   ├── app.py                      # Main entry point & 10-page router
│   ├── styles.py                   # Enterprise CSS design system & layout
│   ├── utils.py                    # Database connection, safe SQL, inference, MLflow
│   └── views/
│       ├── overview.py             # Page 1: Overview
│       ├── pipeline_monitor.py     # Page 2: Pipelines (Pipeline Monitor)
│       ├── data_quality.py         # Page 3: Data Quality
│       ├── warehouse_explorer.py   # Page 4: Warehouse
│       ├── business_intelligence.py# Page 5: Analytics
│       ├── model_training.py       # Page 6: Model Evaluation
│       ├── mlflow_registry.py      # Page 7: Model Registry
│       ├── churn_prediction.py     # Page 8: Churn Prediction
│       ├── risk_analytics.py       # Page 9: Risk Analytics
│       └── system_health.py        # Page 10: System Health
├── data/
│   ├── raw/                        # Online Retail.xlsx
│   ├── processed/                  # Cleaned fact_sales, features, scored files
│   └── rejected/                   # Zero Data Loss quarantine CSV files
├── etl/                            # Ingestion, validation, transformation, load
├── ml/
│   ├── feature_engineering.py      # Temporal cohort extraction & feature store
│   ├── train.py                    # Multi-model training, 5-fold CV & artifacts
│   ├── mlflow_tracking.py          # MLflow experiment logging & registry
│   ├── predict.py                  # Batch scoring & prescriptive playbooks
│   ├── api.py                      # FastAPI REST microservice
│   ├── test_api.py                 # Automated API test suite
│   └── artifacts/                  # Saved model, metrics JSON & diagnostic plots
├── logs/                           # Pipeline run history & console outputs
├── ecommerce.db                    # SQLite Kimball Star Schema Warehouse (70MB)
├── mlflow.db                       # MLflow SQLite Tracking Store
├── run_mlops.py                    # Master orchestrator script
├── run_all.bat                     # Windows one-click service launcher
├── requirements.txt                # Python dependencies
└── README.md                       # Platform documentation
```
