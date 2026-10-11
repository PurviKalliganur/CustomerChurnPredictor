"""
Unit and Integration Tests for FastAPI Inference Service
--------------------------------------------------------
Tests:
  1. GET  /               (Root endpoint)
  2. GET  /health         (Health check)
  3. GET  /model_info     (Model metadata)
  4. POST /predict        (Single customer prediction)
  5. POST /predict_batch  (Batch predictions)
  6. GET  /customer/{id}  (Database customer scoring)
"""

import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)

from fastapi.testclient import TestClient
from ml.api import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ONLINE"
    assert "documentation" in data
    print("PASS: Root endpoint test")

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert data["model_loaded"] is True
    print("PASS: Health endpoint test")

def test_model_info_endpoint():
    response = client.get("/model_info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "CustomerChurnClassifier"
    assert len(data["features_required"]) > 0
    print("PASS: Model info endpoint test")

def test_predict_endpoint():
    payload = {
        "customer_id": 99999,
        "recency_days": 15.0,
        "frequency": 8,
        "monetary_total": 3500.0,
        "avg_order_value": 437.5,
        "total_quantity": 2100,
        "avg_basket_size": 262.5,
        "unique_products": 45,
        "tenure_days": 210.0,
        "purchase_span_days": 195.0,
        "cancellation_count": 0,
        "cancellation_rate": 0.0,
        "is_uk": 1,
        "repeat_customer": 1,
        "threshold": 0.50
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "churn_probability" in data
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert "risk_tier" in data
    assert "recommended_action" in data
    print(f"PASS: Predict endpoint test (Prob: {data['churn_probability']}, Tier: {data['risk_tier']})")

def test_batch_predict_endpoint():
    batch_payload = [
        {
            "customer_id": 10001,
            "recency_days": 5.0,
            "frequency": 12,
            "monetary_total": 5400.0,
            "avg_order_value": 450.0,
            "total_quantity": 3000,
            "avg_basket_size": 250.0,
            "unique_products": 60,
            "tenure_days": 300.0,
            "purchase_span_days": 295.0,
            "cancellation_count": 1,
            "cancellation_rate": 0.08,
            "is_uk": 1,
            "repeat_customer": 1
        },
        {
            "customer_id": 10002,
            "recency_days": 250.0,
            "frequency": 1,
            "monetary_total": 85.0,
            "avg_order_value": 85.0,
            "total_quantity": 10,
            "avg_basket_size": 10.0,
            "unique_products": 2,
            "tenure_days": 250.0,
            "purchase_span_days": 0.0,
            "cancellation_count": 0,
            "cancellation_rate": 0.0,
            "is_uk": 0,
            "repeat_customer": 0
        }
    ]
    response = client.post("/predict_batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_records"] == 2
    assert len(data["predictions"]) == 2
    print(f"PASS: Batch predict endpoint test (Total: {data['total_records']}, Breakdown: {data['risk_breakdown']})")

def test_customer_lookup_endpoint():
    response = client.get("/customer/17850")
    assert response.status_code == 200
    data = response.json()
    assert data["customer_id"] == 17850
    assert "churn_probability" in data
    assert "risk_tier" in data
    assert "recommended_action" in data
    print(f"PASS: Customer lookup endpoint test (Customer 17850, Prob: {data['churn_probability']}, Tier: {data['risk_tier']})")

def run_all_tests():
    print("\n=========================================================")
    print("       MLOps API Validation & Automated Tests           ")
    print("=========================================================")
    test_root_endpoint()
    test_health_endpoint()
    test_model_info_endpoint()
    test_predict_endpoint()
    test_batch_predict_endpoint()
    test_customer_lookup_endpoint()
    print("=========================================================")
    print("  ALL 6 API TESTS PASSED SUCCESSFULLY!                  ")
    print("=========================================================\n")

if __name__ == '__main__':
    run_all_tests()
