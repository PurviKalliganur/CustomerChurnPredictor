"""
Model Development & Evaluation Module
-------------------------------------
Trains, cross-validates, and evaluates multiple classification algorithms:
  - Baseline: Logistic Regression (StandardScaler)
  - Candidate 1: Random Forest Classifier
  - Candidate 2: Gradient Boosting Classifier

Computes comprehensive metrics: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC.
Generates publication-quality artifact plots:
  - ROC Curves (ml/artifacts/roc_curves.png)
  - Precision-Recall Curves (ml/artifacts/pr_curves.png)
  - Confusion Matrix Heatmap (ml/artifacts/confusion_matrix.png)
  - Feature Importance Rankings (ml/artifacts/feature_importance.png)
  - Metrics Summary JSON (ml/artifacts/evaluation_metrics.json)
Saves the champion model to ml/artifacts/best_model.joblib
"""

import os
import json
import logging
import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

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
import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)

from ml.feature_engineering import FEATURE_COLUMNS, run_feature_engineering

logging.basicConfig(
    filename='logs/train.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_or_generate_dataset(data_path='data/processed/train_features.csv'):
    """Loads time-split features or runs feature engineering if not present."""
    if not os.path.exists(data_path):
        print(f"File {data_path} not found. Running feature engineering...")
        train_df, _ = run_feature_engineering()
        return train_df
    return pd.read_csv(data_path)

def build_model_definitions():
    """Defines candidate models and pipelines."""
    models = {
        "Logistic_Regression": Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'))
        ]),
        "Random_Forest": Pipeline([
            ('clf', RandomForestClassifier(
                n_estimators=150,
                max_depth=6,
                min_samples_split=6,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ))
        ]),
        "Gradient_Boosting": Pipeline([
            ('clf', GradientBoostingClassifier(
                n_estimators=120,
                learning_rate=0.08,
                max_depth=4,
                subsample=0.85,
                random_state=42
            ))
        ])
    }
    return models

def evaluate_model(name, model, X_train, y_train, X_test, y_test, feature_cols):
    """Trains and rigorously evaluates a single model pipeline."""
    logging.info(f"Training model: {name}...")
    
    # Stratified 5-Fold Cross-Validation on training set
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1').mean()
    cv_roc = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc').mean()
    
    # Fit model
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    
    metrics = {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "cv_f1_mean": round(float(cv_f1), 4),
        "cv_roc_auc_mean": round(float(cv_roc), 4),
        "confusion_matrix": cm.tolist()
    }
    
    logging.info(f"Model {name} Evaluation: {metrics}")
    return {
        "name": name,
        "model": model,
        "metrics": metrics,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "cm": cm
    }

def plot_evaluation_artifacts(results, y_test, feature_cols, best_model_name):
    """Generates and saves visual plots for model comparison and interpretation."""
    sns.set_theme(style="whitegrid")
    
    # 1. ROC Curves Plot
    plt.figure(figsize=(8, 6))
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
        auc_score = res['metrics']['roc_auc']
        plt.plot(fpr, tpr, lw=2, label=f"{name.replace('_', ' ')} (AUC = {auc_score:.3f})")
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.5, label='Random Baseline')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12, fontweight='bold')
    plt.title('Receiver Operating Characteristic (ROC) Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    roc_path = 'ml/artifacts/roc_curves.png'
    plt.savefig(roc_path, dpi=300)
    plt.close()
    
    # 2. Precision-Recall Curves Plot
    plt.figure(figsize=(8, 6))
    for name, res in results.items():
        prec, rec, _ = precision_recall_curve(y_test, res['y_proba'])
        pr_score = res['metrics']['pr_auc']
        plt.plot(rec, prec, lw=2, label=f"{name.replace('_', ' ')} (PR-AUC = {pr_score:.3f})")
    plt.xlabel('Recall', fontsize=12, fontweight='bold')
    plt.ylabel('Precision', fontsize=12, fontweight='bold')
    plt.title('Precision-Recall Curves Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left", frameon=True)
    plt.tight_layout()
    pr_path = 'ml/artifacts/pr_curves.png'
    plt.savefig(pr_path, dpi=300)
    plt.close()
    
    # 3. Confusion Matrix for Champion Model
    best_res = results[best_model_name]
    cm = best_res['cm']
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', cbar=False,
        xticklabels=['Retained (0)', 'Churned (1)'],
        yticklabels=['Retained (0)', 'Churned (1)']
    )
    plt.title(f'Confusion Matrix: {best_model_name.replace("_", " ")}', fontsize=13, fontweight='bold')
    plt.xlabel('Predicted Class', fontsize=11, fontweight='bold')
    plt.ylabel('Actual Class', fontsize=11, fontweight='bold')
    plt.tight_layout()
    cm_path = 'ml/artifacts/confusion_matrix.png'
    plt.savefig(cm_path, dpi=300)
    plt.close()
    
    # 4. Feature Importance (Random Forest tree-based feature analysis)
    rf_res = results.get("Random_Forest") or results.get("Gradient_Boosting")
    if rf_res:
        rf_pipe = rf_res['model']
        clf = rf_pipe.named_steps['clf']
        if hasattr(clf, 'feature_importances_'):
            importances = clf.feature_importances_
            feat_df = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': importances
            }).sort_values('Importance', ascending=True)
            
            plt.figure(figsize=(9, 6))
            plt.barh(feat_df['Feature'], feat_df['Importance'], color='#0284c7', edgecolor='#0369a1')
            plt.title('Customer Behavioral Feature Importance (Random Forest)', fontsize=13, fontweight='bold')
            plt.xlabel('Importance Score (Mean Decrease in Impurity)', fontsize=11, fontweight='bold')
            plt.ylabel('Feature Name', fontsize=11, fontweight='bold')
            plt.tight_layout()
            fi_path = 'ml/artifacts/feature_importance.png'
            plt.savefig(fi_path, dpi=300)
            plt.close()
            
            feat_dict = dict(zip(feat_df['Feature'], feat_df['Importance'].round(4)))
            with open('ml/artifacts/feature_importance.json', 'w') as f:
                json.dump(feat_dict, f, indent=2)

def train_and_evaluate():
    print("\n=========================================================")
    print("      MLOps Stage: Model Development & Evaluation        ")
    print("=========================================================")
    
    df = load_or_generate_dataset()
    X = df[FEATURE_COLUMNS]
    y = df['churn_label']
    
    print(f"Loaded {len(df)} samples ({FEATURE_COLUMNS.__len__()} features). Target distribution: {y.value_counts().to_dict()}")
    
    # Train-test split (80/20 Stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Training split: {len(X_train)} | Test split: {len(X_test)}")
    
    models = build_model_definitions()
    results = {}
    
    print("\n--- Model Training & Validation Scores ---")
    for name, model in models.items():
        res = evaluate_model(name, model, X_train, y_train, X_test, y_test, FEATURE_COLUMNS)
        results[name] = res
        m = res['metrics']
        print(f"[{name}]")
        print(f"   Accuracy: {m['accuracy']:.4f} | F1-Score: {m['f1_score']:.4f} | ROC-AUC: {m['roc_auc']:.4f}")
        print(f"   Precision: {m['precision']:.4f} | Recall: {m['recall']:.4f} | 5-Fold CV ROC: {m['cv_roc_auc_mean']:.4f}")
        
    # Select best model based on ROC-AUC & F1
    best_model_name = max(results.keys(), key=lambda k: results[k]['metrics']['roc_auc'] + results[k]['metrics']['f1_score'])
    best_result = results[best_model_name]
    print(f"\n--> Champion Model Selected: {best_model_name} (ROC-AUC: {best_result['metrics']['roc_auc']:.4f})")
    
    # Plot artifacts
    plot_evaluation_artifacts(results, y_test, FEATURE_COLUMNS, best_model_name)
    
    # Save artifacts
    champion_pipeline = best_result['model']
    model_path = 'ml/artifacts/best_model.joblib'
    joblib.dump(champion_pipeline, model_path)
    print(f"--> Saved Champion Model to {model_path}")
    
    # Save feature columns
    with open('ml/artifacts/feature_columns.json', 'w') as f:
        json.dump(FEATURE_COLUMNS, f, indent=2)
        
    # Save all metrics summary
    metrics_summary = {
        "champion_model": best_model_name,
        "evaluation_timestamp": pd.Timestamp.now().isoformat(),
        "test_size": len(y_test),
        "train_size": len(y_train),
        "features": FEATURE_COLUMNS,
        "models": {k: v['metrics'] for k, v in results.items()}
    }
    with open('ml/artifacts/evaluation_metrics.json', 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    print("--> Saved metrics summary to ml/artifacts/evaluation_metrics.json")
    print("=========================================================\n")
    
    return results, best_model_name, X_train, X_test, y_train, y_test

if __name__ == '__main__':
    train_and_evaluate()
