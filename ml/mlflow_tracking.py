"""
MLflow Tracking & Model Registry Integration Module
---------------------------------------------------
Orchestrates experiment tracking, metrics logging, artifact logging,
and model registry transitions using MLflow with an SQLite backend store.
"""

import os
import sys

# Setup paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)
os.makedirs('logs', exist_ok=True)
os.makedirs('ml/artifacts', exist_ok=True)

import json
import logging
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

from ml.train import train_and_evaluate, FEATURE_COLUMNS

logging.basicConfig(
    filename='logs/mlflow.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

EXPERIMENT_NAME = "ECommerce_Customer_Churn_Prediction"
REGISTERED_MODEL_NAME = "CustomerChurnClassifier"
DB_PATH = os.path.join(PROJECT_DIR, "mlflow.db").replace('\\', '/')
TRACKING_URI = f"sqlite:///{DB_PATH}"

def setup_mlflow():
    """Configures MLflow tracking URI and initializes experiment."""
    mlflow.set_tracking_uri(TRACKING_URI)
    logging.info(f"Set MLflow tracking URI: {TRACKING_URI}")
    
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        exp_id = mlflow.create_experiment(
            EXPERIMENT_NAME,
            artifact_location=os.path.join(PROJECT_DIR, "mlruns").replace('\\', '/')
        )
        logging.info(f"Created new MLflow experiment: {EXPERIMENT_NAME} (ID: {exp_id})")
    else:
        exp_id = experiment.experiment_id
        
    mlflow.set_experiment(EXPERIMENT_NAME)
    return exp_id

def log_experiment_runs():
    """
    Executes training and logs all candidate models, metrics, plots,
    and signatures to MLflow, then registers the best model.
    """
    print("\n=========================================================")
    print("      MLOps Stage: MLflow Tracking & Model Registry       ")
    print("=========================================================")
    
    exp_id = setup_mlflow()
    client = MlflowClient()
    
    # Run training to get models & evaluation results
    results, champion_name, X_train, X_test, y_train, y_test = train_and_evaluate()
    
    logged_runs = {}
    
    for name, res in results.items():
        run_name = f"churn_{name.lower()}"
        logging.info(f"Starting MLflow run for {name}...")
        
        with mlflow.start_run(run_name=run_name, experiment_id=exp_id) as run:
            run_id = run.info.run_id
            logged_runs[name] = run_id
            
            # 1. Log Parameters
            model = res['model']
            clf = model.named_steps['clf']
            params = {
                "algorithm": name,
                "n_features": len(FEATURE_COLUMNS),
                "features_list": ",".join(FEATURE_COLUMNS),
                "test_split_ratio": 0.20,
                "random_state": 42
            }
            if hasattr(clf, 'get_params'):
                for p_key, p_val in clf.get_params().items():
                    if isinstance(p_val, (int, float, str, bool)) or p_val is None:
                        params[f"clf_{p_key}"] = str(p_val)
            mlflow.log_params(params)
            
            # 2. Log Metrics
            metrics = res['metrics']
            for m_key, m_val in metrics.items():
                if isinstance(m_val, (int, float)):
                    mlflow.log_metric(m_key, float(m_val))
                    
            # 3. Log Artifact Plots
            artifacts = [
                'ml/artifacts/roc_curves.png',
                'ml/artifacts/pr_curves.png',
                'ml/artifacts/confusion_matrix.png',
                'ml/artifacts/feature_importance.png',
                'ml/artifacts/evaluation_metrics.json'
            ]
            for art in artifacts:
                if os.path.exists(art):
                    mlflow.log_artifact(art, artifact_path="evaluation_plots")
                    
            # 4. Infer Model Signature & Log Model
            input_example = X_train.head(3)
            signature = infer_signature(X_train, res['y_pred'][:len(X_train)])
            
            try:
                mlflow.sklearn.log_model(
                    sk_model=model,
                    name="model",
                    signature=signature,
                    input_example=input_example,
                    serialization_format="cloudpickle"
                )
            except Exception as e_log:
                logging.warning(f"Fallback model log: {e_log}")
                mlflow.sklearn.log_model(
                    sk_model=model,
                    name="model",
                    signature=signature,
                    input_example=input_example
                )
            
            print(f"--> [MLflow Run Logged] {name} (Run ID: {run_id[:8]}..., ROC-AUC: {metrics['roc_auc']:.4f})")
            
    # 5. Model Registry for Champion Model
    champion_run_id = logged_runs[champion_name]
    champion_model_uri = f"runs:/{champion_run_id}/model"
    
    print(f"\n--> Registering Champion Model '{champion_name}' in MLflow Registry...")
    try:
        # Register model
        mv = mlflow.register_model(
            model_uri=champion_model_uri,
            name=REGISTERED_MODEL_NAME
        )
        print(f"--> Model Registered: '{REGISTERED_MODEL_NAME}' Version {mv.version}")
        
        # Add description and tags
        client.update_model_version(
            name=REGISTERED_MODEL_NAME,
            version=mv.version,
            description=f"Champion Customer Churn model ({champion_name}) with ROC-AUC {results[champion_name]['metrics']['roc_auc']:.4f}."
        )
        
        # Set production alias or transition to Production
        try:
            client.set_registered_model_alias(REGISTERED_MODEL_NAME, "champion", mv.version)
            client.set_registered_model_alias(REGISTERED_MODEL_NAME, "production", mv.version)
            print(f"--> Assigned aliases 'champion' and 'production' to Version {mv.version}")
        except Exception:
            pass
            
        print(f"--> Successfully Registered and Promoted to Production!")
        
    except Exception as e:
        print(f"[Warning] MLflow registry note: {e}")
        logging.warning(f"MLflow model registry note: {e}")
        
    print("=========================================================\n")
    return logged_runs, champion_name

def get_registered_models_summary():
    """Utility to query registered models and recent runs for dashboard & API."""
    setup_mlflow()
    client = MlflowClient()
    summary = {
        "tracking_uri": TRACKING_URI,
        "experiment_name": EXPERIMENT_NAME,
        "registered_models": []
    }
    
    try:
        for rm in client.search_registered_models():
            model_info = {
                "name": rm.name,
                "latest_versions": [
                    {
                        "version": v.version,
                        "status": v.status,
                        "current_stage": v.current_stage,
                        "description": v.description,
                        "run_id": v.run_id
                    }
                    for v in rm.latest_versions
                ]
            }
            summary["registered_models"].append(model_info)
    except Exception as e:
        summary["error"] = str(e)
        
    return summary

if __name__ == '__main__':
    log_experiment_runs()
