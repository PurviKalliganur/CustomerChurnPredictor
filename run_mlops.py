"""
End-to-End MLOps Pipeline Master Orchestrator
--------------------------------------------
Executes the complete machine learning lifecycle in automated sequence:
  [Stage 1/5] Feature Engineering & Cohort Extraction (Leakage-free)
  [Stage 2/5] Model Training, Cross-Validation & Metric Evaluation
  [Stage 3/5] MLflow Experiment Tracking & Model Registry Transition
  [Stage 4/5] Enterprise Batch Scoring & Churn Risk Segmentation
  [Stage 5/5] Microservice Validation & Automated Health Checks
"""

import os
import sys
import time
import logging
from datetime import datetime

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)
os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    filename='logs/mlops_pipeline.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

from ml.feature_engineering import run_feature_engineering
from ml.train import train_and_evaluate
from ml.mlflow_tracking import log_experiment_runs
from ml.predict import run_batch_scoring
from ml.test_api import run_all_tests

def run_mlops_pipeline():
    start_total = time.time()
    print("=" * 65)
    print("       E-Commerce Enterprise MLOps Pipeline Orchestrator       ")
    print("=" * 65)
    logging.info("MLOps Pipeline orchestration started.")
    
    try:
        # Stage 1: Feature Engineering
        print("\n[Stage 1/5] Feature Engineering & Feature Store Generation...")
        t0 = time.time()
        train_df, cust_df = run_feature_engineering()
        print(f"--> Stage 1 completed in {time.time() - t0:.2f}s (Train: {len(train_df):,}, Live: {len(cust_df):,})")
        
        # Stage 2: Model Training & Evaluation
        print("\n[Stage 2/5] Training Candidate Models & Evaluating Metrics...")
        t0 = time.time()
        results, champion_name, X_train, X_test, y_train, y_test = train_and_evaluate()
        print(f"--> Stage 2 completed in {time.time() - t0:.2f}s (Champion: {champion_name})")
        
        # Stage 3: MLflow Experiment Tracking & Model Registry
        print("\n[Stage 3/5] MLflow Tracking, Signatures & Model Registry...")
        t0 = time.time()
        logged_runs, reg_champion = log_experiment_runs()
        print(f"--> Stage 3 completed in {time.time() - t0:.2f}s (Logged Runs: {len(logged_runs)})")
        
        # Stage 4: Batch Customer Scoring
        print("\n[Stage 4/5] Full Customer Base Batch Scoring & Risk Tiers...")
        t0 = time.time()
        scored_df = run_batch_scoring()
        print(f"--> Stage 4 completed in {time.time() - t0:.2f}s (Scored: {len(scored_df):,})")
        
        # Stage 5: Microservice Verification & Tests
        print("\n[Stage 5/5] Automated Inference API Validation & Health Check...")
        t0 = time.time()
        run_all_tests()
        print(f"--> Stage 5 completed in {time.time() - t0:.2f}s")
        
        total_duration = time.time() - start_total
        print("=" * 65)
        print(f" [SUCCESS] Full MLOps Pipeline completed in {total_duration:.2f} seconds!")
        print(f" - Feature Store Records     : {len(cust_df):,}")
        print(f" - Training Samples          : {len(train_df):,}")
        print(f" - Champion Model            : {champion_name}")
        print(f" - Model ROC-AUC Score       : {results[champion_name]['metrics']['roc_auc']:.4f}")
        print(f" - Model Registry Status     : Registered in MLflow ('CustomerChurnClassifier')")
        print(f" - Production API Ready      : http://127.0.0.1:8000")
        print("=" * 65)
        logging.info(f"MLOps pipeline completed successfully in {total_duration:.2f}s.")
        return 0
        
    except Exception as e:
        print(f"\n[ERROR] MLOps pipeline failed: {e}")
        logging.error(f"MLOps pipeline failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    run_mlops_pipeline()
