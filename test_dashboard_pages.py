"""
End-to-End Verification Test for All 10 Dashboard Views
-------------------------------------------------------
Verifies that all 10 page view functions execute their backend queries,
models, and data pipelines without raising exceptions.
"""

import os
import sys
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from dashboard.utils import (
    get_warehouse_kpis, get_quarantine_stats, get_evaluation_metrics,
    get_pipeline_history, get_db_info, check_api_health, get_all_tables,
    get_table_schema, load_table_data, execute_safe_sql, load_quarantine_data,
    get_mlflow_runs, get_mlflow_registered_models, predict_customer
)

def test_page_1_overview_data():
    kpis = get_warehouse_kpis()
    assert kpis['total_revenue'] > 8000000, "KPI revenue invalid"
    assert kpis['total_orders'] > 18000, "KPI orders invalid"
    assert kpis['unique_customers'] > 4300, "KPI customers invalid"
    print("Page 1 (Overview) Data: OK")

def test_page_2_pipeline_monitor():
    history = get_pipeline_history()
    assert len(history) > 0, "No pipeline history found"
    print("Page 2 (Pipelines & Task Monitor): OK")

def test_page_3_data_quality():
    cats, total_q = get_quarantine_stats()
    assert total_q > 140000, f"Quarantine count mismatch: {total_q}"
    sample = load_quarantine_data("missing_fields.csv", limit=5)
    assert not sample.empty, "Failed to load quarantine sample"
    print("Page 3 (Data Engineering & Quality): OK")

def test_page_4_warehouse_explorer():
    tables = get_all_tables()
    assert 'fact_sales' in tables['name'].values, "fact_sales missing"
    schema = get_table_schema('fact_sales')
    assert not schema.empty, "Schema empty"
    df_sql, ms, err = execute_safe_sql("SELECT COUNT(*) AS cnt FROM fact_sales;")
    assert err is None, f"SQL error: {err}"
    assert df_sql['cnt'].iloc[0] >= 380000, f"Row count mismatch in SQL: {df_sql['cnt'].iloc[0]}"
    print("Page 4 (Data Warehouse Explorer): OK")

def test_page_5_bi_queries():
    from dashboard.views.business_intelligence import load_bi_data, load_rfm_data
    sales = load_bi_data()
    assert not sales.empty, "BI sales empty"
    rfm = load_rfm_data()
    assert not rfm.empty, "RFM mart empty"
    print("Page 5 (Business Intelligence): OK")

def test_page_6_model_training():
    metrics = get_evaluation_metrics()
    assert "champion_model" in metrics, "Champion model missing"
    assert "Logistic_Regression" in metrics["models"], "LogReg missing"
    print("Page 6 (Model Training & Evaluation): OK")

def test_page_7_mlflow_registry():
    runs = get_mlflow_runs()
    assert not runs.empty, "MLflow runs empty"
    reg = get_mlflow_registered_models()
    assert not reg.empty, "Registered models empty"
    print("Page 7 (MLflow Model Registry): OK")

def test_page_8_churn_prediction():
    sample_feat = {
        "recency_days": 30, "frequency": 5, "monetary_total": 1500,
        "avg_order_value": 300, "total_quantity": 400, "avg_basket_size": 80,
        "unique_products": 25, "tenure_days": 180, "purchase_span_days": 150,
        "cancellation_count": 0, "cancellation_rate": 0.0, "is_uk": 1, "repeat_customer": 1
    }
    pred = predict_customer(sample_feat)
    assert "churn_probability" in pred, "Churn prob missing"
    assert "risk_tier" in pred, "Risk tier missing"
    assert "retention_playbook" in pred, "Playbook missing"
    print(f"Page 8 (Customer Churn Prediction): OK (Prob: {pred['churn_probability']:.4f}, Tier: {pred['risk_tier']})")

def test_page_9_risk_analytics():
    from dashboard.views.risk_analytics import load_scored_customers
    scored = load_scored_customers()
    assert len(scored) == 4338, f"Expected 4,338 scored customers, got {len(scored)}"
    print("Page 9 (Customer Risk Analytics): OK")

def test_page_10_system_health():
    from dashboard.views.system_health import check_dependencies, check_db_integrity, check_artifacts
    deps = check_dependencies()
    assert (deps['Status'] == 'Installed').all(), "Missing dependencies detected"
    integ, cnt = check_db_integrity()
    assert integ.lower() == "ok", f"DB integrity check failed: {integ}"
    arts = check_artifacts()
    assert (arts['Status'] == 'Available').all(), "Missing model artifacts"
    print("Page 10 (System Health): OK")

if __name__ == "__main__":
    print("=========================================================")
    print(" Running Full Backend & Page Integration Verification   ")
    print("=========================================================")
    test_page_1_overview_data()
    test_page_2_pipeline_monitor()
    test_page_3_data_quality()
    test_page_4_warehouse_explorer()
    test_page_5_bi_queries()
    test_page_6_model_training()
    test_page_7_mlflow_registry()
    test_page_8_churn_prediction()
    test_page_9_risk_analytics()
    test_page_10_system_health()
    print("=========================================================")
    print(" ALL 10 PAGES BACKEND & LOGIC VERIFIED 100% SUCCESSFUL!  ")
    print("=========================================================")
