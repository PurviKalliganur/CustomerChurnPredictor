"""
Enterprise FastAPI Model Deployment Microservice
------------------------------------------------
Provides production REST API endpoints for Customer Churn inference:
  - GET  /            : Service overview and interactive docs links
  - GET  /health      : System health check & model readiness status
  - GET  /model_info  : Model architecture, features & evaluation metrics
  - POST /predict     : Real-time single customer inference & retention playbook
  - POST /predict_batch: High-throughput batch inference
  - GET  /customer/{id}: Direct database customer lookup and scoring
"""

import os
import sys

# Setup paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)

import json
import sqlite3
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ml.predict import predict_single, predict_batch, load_production_model, MODEL_PATH
from ml.mlflow_tracking import get_registered_models_summary

logging.basicConfig(
    filename='logs/api.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Application Lifespan Handler (Modern replacement for on_event)
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Validates and pre-loads model artifact into memory upon server startup."""
    try:
        load_production_model()
        logging.info("Champion model loaded into memory on API startup.")
    except Exception as e:
        logging.warning(f"Startup model loading warning: {e}")
    yield

# Initialize FastAPI application
app = FastAPI(
    title="E-Commerce Customer Churn Prediction API",
    description="Production MLOps Inference Microservice for customer churn probability, risk classification, and retention strategies.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Enable CORS for cross-origin dashboard communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Schemas (Pydantic v2 compatible)
class CustomerFeaturesInput(BaseModel):
    customer_id: Optional[int] = Field(None, description="Optional customer ID for tracking")
    recency_days: float = Field(..., description="Days since last purchase")
    frequency: int = Field(..., description="Total unique orders placed")
    monetary_total: float = Field(..., description="Total lifetime spend ($)")
    avg_order_value: float = Field(..., description="Average Order Value ($)")
    total_quantity: int = Field(..., description="Total units purchased")
    avg_basket_size: float = Field(..., description="Average items per order")
    unique_products: int = Field(..., description="Distinct SKUs purchased")
    tenure_days: float = Field(..., description="Days since first purchase")
    purchase_span_days: float = Field(..., description="Days between first and last purchase")
    cancellation_count: int = Field(0, description="Total cancelled invoices")
    cancellation_rate: float = Field(0.0, description="Cancellation ratio")
    is_uk: int = Field(1, description="1 if United Kingdom, 0 otherwise")
    repeat_customer: int = Field(1, description="1 if frequency > 1, 0 otherwise")
    threshold: Optional[float] = Field(0.50, description="Classification decision threshold (default 0.50)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "customer_id": 17850,
                "recency_days": 14.0,
                "frequency": 6,
                "monetary_total": 2450.50,
                "avg_order_value": 408.42,
                "total_quantity": 1350,
                "avg_basket_size": 225.0,
                "unique_products": 42,
                "tenure_days": 180.0,
                "purchase_span_days": 166.0,
                "cancellation_count": 1,
                "cancellation_rate": 0.14,
                "is_uk": 1,
                "repeat_customer": 1,
                "threshold": 0.50
            }
        }
    }

class PredictionOutput(BaseModel):
    customer_id: Optional[int]
    churn_probability: float
    is_churn_predicted: int
    churn_threshold: float
    risk_tier: str
    recommended_action: str
    prediction_timestamp: str

class BatchPredictionResponse(BaseModel):
    total_records: int
    churn_rate_predicted: float
    risk_breakdown: dict
    predictions: List[PredictionOutput]

def extract_model_dict(obj):
    """Safely extracts dictionary from Pydantic model (v1 or v2 compatible)."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return obj.dict()

from fastapi import FastAPI, HTTPException, status, Response

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=status.HTTP_204_NO_CONTENT)

@app.get("/", tags=["General"])
def root():
    return {
        "service": "E-Commerce Customer Churn Prediction API",
        "status": "ONLINE",
        "documentation": "/docs",
        "health_check": "/health",
        "model_info": "/model_info",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health", tags=["Health & Diagnostics"])
def health_check():
    """Health check endpoint for Kubernetes / Docker liveness probes."""
    model_loaded = os.path.exists(MODEL_PATH)
    return {
        "status": "HEALTHY" if model_loaded else "DEGRADED",
        "model_loaded": model_loaded,
        "artifact_path": MODEL_PATH,
        "timestamp": datetime.utcnow().isoformat(),
        "api_version": "1.0.0"
    }

@app.get("/model_info", tags=["Model Metadata"])
def get_model_info():
    """Returns champion model architecture, features, and evaluation metrics."""
    metrics_path = os.path.join(PROJECT_DIR, 'ml', 'artifacts', 'evaluation_metrics.json')
    features_path = os.path.join(PROJECT_DIR, 'ml', 'artifacts', 'feature_columns.json')
    
    if not os.path.exists(metrics_path) or not os.path.exists(features_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model metrics or feature artifacts not found. Please run training pipeline first."
        )
        
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
        
    with open(features_path, 'r') as f:
        features = json.load(f)
        
    mlflow_info = get_registered_models_summary()
    
    return {
        "model_name": "CustomerChurnClassifier",
        "champion_algorithm": metrics.get("champion_model"),
        "features_required": features,
        "evaluation_metrics": metrics.get("models", {}).get(metrics.get("champion_model")),
        "all_candidate_scores": metrics.get("models"),
        "mlflow_registry": mlflow_info
    }

@app.post("/predict", response_model=PredictionOutput, tags=["Inference"])
def predict_churn(customer: CustomerFeaturesInput):
    """
    Real-time inference endpoint. Computes churn probability, risk tier,
    and returns prescriptive business retention action.
    """
    try:
        data_dict = extract_model_dict(customer)
        threshold = float(data_dict.pop('threshold', 0.50))
        cid = data_dict.get('customer_id')
        if cid is not None:
            cid = int(cid)
        
        result = predict_single(data_dict, threshold=threshold)
        
        return PredictionOutput(
            customer_id=cid,
            churn_probability=float(result['churn_probability']),
            is_churn_predicted=int(result['is_churn_predicted']),
            churn_threshold=float(result['churn_threshold']),
            risk_tier=str(result['risk_tier']),
            recommended_action=str(result['recommended_action']),
            prediction_timestamp=datetime.utcnow().isoformat()
        )
    except Exception as e:
        logging.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(e)}"
        )

@app.post("/predict_batch", response_model=BatchPredictionResponse, tags=["Inference"])
def batch_inference(customers: List[CustomerFeaturesInput]):
    """
    Batch inference endpoint for high-throughput customer cohort scoring.
    """
    try:
        if not customers:
            raise HTTPException(status_code=400, detail="Empty customer list provided.")
            
        records = [extract_model_dict(c) for c in customers]
        results = []
        for r in records:
            threshold = float(r.pop('threshold', 0.50))
            cid = r.get('customer_id')
            if cid is not None:
                cid = int(cid)
            res = predict_single(r, threshold=threshold)
            results.append(PredictionOutput(
                customer_id=cid,
                churn_probability=float(res['churn_probability']),
                is_churn_predicted=int(res['is_churn_predicted']),
                churn_threshold=float(res['churn_threshold']),
                risk_tier=str(res['risk_tier']),
                recommended_action=str(res['recommended_action']),
                prediction_timestamp=datetime.utcnow().isoformat()
            ))
            
        total = len(results)
        churn_count = sum(r.is_churn_predicted for r in results)
        churn_rate = round(churn_count / total, 4) if total > 0 else 0.0
        
        risk_breakdown = {}
        for r in results:
            risk_breakdown[r.risk_tier] = risk_breakdown.get(r.risk_tier, 0) + 1
            
        return BatchPredictionResponse(
            total_records=total,
            churn_rate_predicted=churn_rate,
            risk_breakdown=risk_breakdown,
            predictions=results
        )
    except Exception as e:
        logging.error(f"Batch inference error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/customer/{customer_id}", response_model=PredictionOutput, tags=["Customer DB Lookup"])
def score_existing_customer(customer_id: int):
    """
    Fetches customer behavioral profile from data warehouse / feature store
    and generates on-demand real-time prediction.
    """
    db_path = os.path.join(PROJECT_DIR, 'ecommerce.db')
    if not os.path.exists(db_path):
        raise HTTPException(status_code=404, detail="Data warehouse database not found.")
        
    try:
        conn = sqlite3.connect(db_path)
        query = "SELECT * FROM customer_features WHERE customer_id = ?"
        df = pd.read_sql_query(query, conn, params=(customer_id,))
        conn.close()
        
        if df.empty:
            raise HTTPException(status_code=404, detail=f"Customer ID {customer_id} not found in database.")
            
        row_dict = df.iloc[0].to_dict()
        res = predict_single(row_dict)
        
        return PredictionOutput(
            customer_id=int(customer_id),
            churn_probability=float(res['churn_probability']),
            is_churn_predicted=int(res['is_churn_predicted']),
            churn_threshold=float(res['churn_threshold']),
            risk_tier=str(res['risk_tier']),
            recommended_action=str(res['recommended_action']),
            prediction_timestamp=datetime.utcnow().isoformat()
        )
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Customer lookup failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Lookup failed: {str(e)}")

if __name__ == '__main__':
    import uvicorn
    print("Starting E-Commerce Churn Prediction API on http://127.0.0.1:8000 ...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
