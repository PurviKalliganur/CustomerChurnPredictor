"""
Model Inference & Batch Scoring Engine
---------------------------------------
Executes single-customer real-time inference and full database batch scoring.
Generates:
  - Churn Probability (0.00 to 1.00)
  - Churn Prediction (0 or 1)
  - Risk Tier (Low Risk, Moderate Risk, High Risk / Critical)
  - Prescriptive Business Retention Action Playbook
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
import joblib
import numpy as np
import pandas as pd

logging.basicConfig(
    filename='logs/predict.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

MODEL_PATH = os.path.join(PROJECT_DIR, 'ml', 'artifacts', 'best_model.joblib')
FEATURES_PATH = os.path.join(PROJECT_DIR, 'ml', 'artifacts', 'feature_columns.json')

_CACHED_MODEL = None
_CACHED_FEATURES = None

def load_production_model():
    """Loads and caches the champion model pipeline and feature schema."""
    global _CACHED_MODEL, _CACHED_FEATURES
    if _CACHED_MODEL is None or _CACHED_FEATURES is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Run training first.")
        _CACHED_MODEL = joblib.load(MODEL_PATH)
        
        with open(FEATURES_PATH, 'r') as f:
            _CACHED_FEATURES = json.load(f)
            
    return _CACHED_MODEL, _CACHED_FEATURES

def assign_risk_tier_and_action(prob: float, monetary: float = 0.0, frequency: int = 1):
    """
    Assigns business risk category and recommended retention action
    based on predicted churn probability and customer lifetime spend.
    """
    if prob >= 0.65:
        tier = "High Risk / Critical"
        if monetary > 1500 or frequency >= 5:
            action = "[Urgent VIP Outreach] Assign Dedicated Account Rep + Exclusive 25% Comeback Concierge Offer"
        else:
            action = "[Win-Back Campaign] Automated Dynamic Discount Email (15% Off Cart Revival)"
    elif prob >= 0.40:
        tier = "Moderate Risk"
        if monetary > 1000:
            action = "[Loyalty Boost] Double Loyalty Points on Next Purchase + Free Priority Shipping"
        else:
            action = "[Product Recommendation] Send Curated Best-Sellers & New Arrivals Newsletter"
    else:
        tier = "Low Risk / Active"
        if monetary > 2000:
            action = "[VIP Champion] Invite to Platinum Club & Early Access New Product Launches"
        else:
            action = "[Steady Engagement] Standard Post-Purchase Nurture Sequence & Product Reviews"
            
    return tier, action

def predict_single(customer_dict: dict, threshold: float = 0.50):
    """
    Performs real-time inference on a single customer feature vector.
    """
    model, features = load_production_model()
    
    # Build input row matching feature columns
    input_df = pd.DataFrame([{f: customer_dict.get(f, 0.0) for f in features}])
    
    # Predict
    prob = float(model.predict_proba(input_df)[0, 1])
    is_churn = int(prob >= threshold)
    
    monetary = float(customer_dict.get('monetary_total', 0.0))
    frequency = int(customer_dict.get('frequency', 1))
    tier, action = assign_risk_tier_and_action(prob, monetary, frequency)
    
    result = {
        "churn_probability": round(prob, 4),
        "is_churn_predicted": is_churn,
        "churn_threshold": threshold,
        "risk_tier": tier,
        "recommended_action": action,
        "features_evaluated": {f: customer_dict.get(f, 0.0) for f in features}
    }
    return result

def predict_batch(df: pd.DataFrame, threshold: float = 0.50):
    """
    Generates batch predictions for a DataFrame of customers.
    """
    model, features = load_production_model()
    
    missing_cols = [c for c in features if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Batch dataframe missing required columns: {missing_cols}")
        
    X = df[features]
    probs = model.predict_proba(X)[:, 1]
    
    res_df = df.copy()
    res_df['churn_probability'] = np.round(probs, 4)
    res_df['is_churn_predicted'] = (res_df['churn_probability'] >= threshold).astype(int)
    
    tiers_and_actions = [
        assign_risk_tier_and_action(
            row['churn_probability'],
            row.get('monetary_total', 0.0),
            row.get('frequency', 1)
        )
        for _, row in res_df.iterrows()
    ]
    
    res_df['risk_tier'] = [t[0] for t in tiers_and_actions]
    res_df['recommended_action'] = [t[1] for t in tiers_and_actions]
    
    return res_df

def run_batch_scoring(source_csv='data/processed/customer_features.csv',
                      output_csv='data/processed/customer_churn_scored.csv'):
    """
    Scores the entire customer feature store and persists results to CSV and SQLite.
    """
    print("\n=========================================================")
    print("      MLOps Stage: Customer Churn Batch Scoring         ")
    print("=========================================================")
    
    if not os.path.exists(source_csv):
        from ml.feature_engineering import extract_current_customer_features
        print("Customer features not found. Extracting live features...")
        df = extract_current_customer_features()
    else:
        df = pd.read_csv(source_csv)
        
    print(f"Loaded {len(df)} customers for batch scoring...")
    scored_df = predict_batch(df)
    
    # Save to CSV
    scored_df.to_csv(output_csv, index=False)
    print(f"--> Batch scored results saved to {output_csv}")
    
    # Save to SQLite
    db_path = 'ecommerce.db'
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            scored_df.to_sql('customer_churn_scored', conn, if_exists='replace', index=False)
            conn.commit()
            conn.close()
            print(f"--> Updated 'customer_churn_scored' table in SQLite ({db_path})")
        except Exception as e:
            logging.warning(f"Could not write scored data to SQLite: {e}")
            
    # Risk Breakdown Summary
    breakdown = scored_df['risk_tier'].value_counts()
    print("\n--- Churn Risk Breakdown Across Customer Base ---")
    for tier, count in breakdown.items():
        pct = (count / len(scored_df)) * 100
        print(f" - {tier:22s}: {count:5d} customers ({pct:.1f}%)")
        
    print("=========================================================\n")
    return scored_df

if __name__ == '__main__':
    run_batch_scoring()
