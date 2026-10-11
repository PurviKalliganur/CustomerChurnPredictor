"""
Feature Engineering Module for Customer Churn Prediction
---------------------------------------------------------
Extracts RFM, behavioral, basket, tenure, and cancellation metrics
from clean transactional sales and rejected cancellation audit logs.
Generates:
  1. Time-split cohort features for ML model training (Leakage-free)
  2. Full snapshot feature store for real-time inference and batch scoring
  3. Feature storage in SQLite / PostgreSQL
"""

import os
import sqlite3
import logging
import pandas as pd
import numpy as np
from datetime import datetime

# Setup paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_DIR)
os.makedirs('logs', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)
os.makedirs('ml/artifacts', exist_ok=True)

logging.basicConfig(
    filename='logs/feature_engineering.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Core Feature Column Definitions
FEATURE_COLUMNS = [
    'recency_days',
    'frequency',
    'monetary_total',
    'avg_order_value',
    'total_quantity',
    'avg_basket_size',
    'unique_products',
    'tenure_days',
    'purchase_span_days',
    'cancellation_count',
    'cancellation_rate',
    'is_uk',
    'repeat_customer'
]

def load_transaction_data(
    val_csv_path='data/processed/val_online_retail.csv',
    cancelled_csv_path='data/rejected/cancelled_invoices.csv'
):
    """Loads valid transactions and cancellation logs."""
    if not os.path.exists(val_csv_path):
        raise FileNotFoundError(f"Valid transactions file not found at: {val_csv_path}")
        
    df_valid = pd.read_csv(val_csv_path)
    df_valid['InvoiceDate'] = pd.to_datetime(df_valid['InvoiceDate'])
    df_valid['CustomerID'] = df_valid['CustomerID'].astype(int)
    
    df_cancelled = None
    if os.path.exists(cancelled_csv_path):
        try:
            df_cancelled = pd.read_csv(cancelled_csv_path)
            if 'InvoiceDate' in df_cancelled.columns:
                df_cancelled['InvoiceDate'] = pd.to_datetime(df_cancelled['InvoiceDate'])
            df_cancelled['CustomerID'] = pd.to_numeric(df_cancelled['CustomerID'], errors='coerce')
            df_cancelled = df_cancelled.dropna(subset=['CustomerID'])
            df_cancelled['CustomerID'] = df_cancelled['CustomerID'].astype(int)
        except Exception as e:
            logging.warning(f"Could not parse cancelled invoices: {e}")
            df_cancelled = None
            
    return df_valid, df_cancelled

def compute_customer_aggregates(tx_df, ref_date, cancelled_df=None):
    """
    Computes behavioral aggregates per customer given a transaction window
    and a reference snapshot date.
    """
    # Vectorized revenue
    tx_df = tx_df.copy()
    tx_df['Revenue'] = tx_df['Quantity'] * tx_df['UnitPrice']
    grouped = tx_df.groupby('CustomerID')
    
    # Base RFM metrics
    recency = (ref_date - grouped['InvoiceDate'].max()).dt.days
    frequency = grouped['InvoiceNo'].nunique()
    monetary = grouped['Revenue'].sum()
    total_quantity = grouped['Quantity'].sum()
    unique_products = grouped['StockCode'].nunique()
    
    first_purchase = grouped['InvoiceDate'].min()
    last_purchase = grouped['InvoiceDate'].max()
    tenure_days = (ref_date - first_purchase).dt.days
    purchase_span_days = (last_purchase - first_purchase).dt.days
    
    # Geography (Majority country per customer)
    country_series = grouped['Country'].agg(lambda x: x.mode()[0] if not x.empty else 'Unknown')
    is_uk = (country_series == 'United Kingdom').astype(int)
    
    # Create base dataframe
    feats = pd.DataFrame({
        'customer_id': frequency.index,
        'recency_days': recency.values,
        'frequency': frequency.values,
        'monetary_total': monetary.values.round(2),
        'avg_order_value': (monetary / frequency).values.round(2),
        'total_quantity': total_quantity.values,
        'avg_basket_size': (total_quantity / frequency).values.round(2),
        'unique_products': unique_products.values,
        'tenure_days': tenure_days.values,
        'purchase_span_days': purchase_span_days.values,
        'is_uk': is_uk.values,
        'repeat_customer': (frequency > 1).astype(int).values
    })
    
    # Incorporate cancellations
    if cancelled_df is not None and not cancelled_df.empty:
        # Filter cancellations up to ref_date if date exists
        if 'InvoiceDate' in cancelled_df.columns:
            c_sub = cancelled_df[cancelled_df['InvoiceDate'] <= ref_date]
        else:
            c_sub = cancelled_df
            
        canc_counts = c_sub.groupby('CustomerID')['InvoiceNo'].nunique()
        feats = feats.merge(canc_counts.rename('cancellation_count'), left_on='customer_id', right_index=True, how='left')
        feats['cancellation_count'] = feats['cancellation_count'].fillna(0).astype(int)
    else:
        feats['cancellation_count'] = 0
        
    total_tx = feats['frequency'] + feats['cancellation_count']
    feats['cancellation_rate'] = (feats['cancellation_count'] / total_tx.replace(0, 1)).round(4)
    
    return feats

def extract_time_split_features(val_csv_path='data/processed/val_online_retail.csv',
                                cancelled_csv_path='data/rejected/cancelled_invoices.csv',
                                churn_window_days=90):
    """
    Extracts time-split cohort features to prevent target leakage.
    - Observation window: Min Date -> Cutoff Date (Max Date - churn_window_days)
    - Prediction window: Cutoff Date -> Max Date
    Target: churn_label = 1 if no purchases in prediction window, else 0.
    """
    logging.info("Extracting time-split cohort features for ML training...")
    df_valid, df_cancelled = load_transaction_data(val_csv_path, cancelled_csv_path)
    
    max_date = df_valid['InvoiceDate'].max()
    cutoff_date = max_date - pd.Timedelta(days=churn_window_days)
    
    # Split transactions
    obs_tx = df_valid[df_valid['InvoiceDate'] <= cutoff_date].copy()
    pred_tx = df_valid[df_valid['InvoiceDate'] > cutoff_date].copy()
    
    # Customers in observation window
    obs_customers = set(obs_tx['CustomerID'].unique())
    active_in_pred = set(pred_tx['CustomerID'].unique())
    
    # Calculate features strictly within observation window
    feats_df = compute_customer_aggregates(obs_tx, ref_date=cutoff_date, cancelled_df=df_cancelled)
    
    # Target: 1 if customer did NOT purchase in the next 90 days, 0 if active
    feats_df['churn_label'] = feats_df['customer_id'].apply(lambda cid: 0 if cid in active_in_pred else 1)
    
    out_path = 'data/processed/train_features.csv'
    feats_df.to_csv(out_path, index=False)
    logging.info(f"Saved {len(feats_df)} time-split training samples to {out_path} (Churn rate: {feats_df['churn_label'].mean():.2%})")
    print(f"--> Saved training features to {out_path} (Rows: {len(feats_df)}, Churn Rate: {feats_df['churn_label'].mean():.2%})")
    
    return feats_df

def extract_current_customer_features(val_csv_path='data/processed/val_online_retail.csv',
                                      cancelled_csv_path='data/rejected/cancelled_invoices.csv'):
    """
    Extracts current snapshot features for all 4,338 customers across full history.
    Used for production batch scoring and live inference lookups.
    """
    logging.info("Extracting current snapshot features for all customers...")
    df_valid, df_cancelled = load_transaction_data(val_csv_path, cancelled_csv_path)
    ref_date = df_valid['InvoiceDate'].max() + pd.Timedelta(days=1)
    
    feats_df = compute_customer_aggregates(df_valid, ref_date=ref_date, cancelled_df=df_cancelled)
    
    out_path = 'data/processed/customer_features.csv'
    feats_df.to_csv(out_path, index=False)
    logging.info(f"Saved {len(feats_df)} current customer feature vectors to {out_path}")
    print(f"--> Saved live customer feature store to {out_path} (Total Customers: {len(feats_df)})")
    
    # Save to SQLite
    db_path = 'ecommerce.db'
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            feats_df.to_sql('customer_features', conn, if_exists='replace', index=False)
            conn.commit()
            conn.close()
            logging.info("Saved customer_features table to SQLite ecommerce.db")
            print(f"--> Updated 'customer_features' table in SQLite ({db_path})")
        except Exception as e:
            logging.warning(f"Could not write to SQLite: {e}")
            
    return feats_df

def run_feature_engineering():
    print("\n=========================================================")
    print("      MLOps Stage: Customer Feature Engineering          ")
    print("=========================================================")
    t_start = datetime.now()
    
    train_df = extract_time_split_features()
    cust_df = extract_current_customer_features()
    
    duration = (datetime.now() - t_start).total_seconds()
    print(f"--> Feature Engineering completed successfully in {duration:.2f}s!")
    print("=========================================================\n")
    return train_df, cust_df

if __name__ == '__main__':
    run_feature_engineering()
