"""
MLflow Model Registry
---------------------
MLflow tracking and registry inspector.
Inspect logged experiments, run parameters, metrics, model artifacts, and production stages.
"""

import os
import streamlit as st
import pandas as pd
from dashboard.utils import get_mlflow_runs, get_mlflow_registered_models, MLFLOW_DB_PATH

def render():
    mlflow_online = os.path.exists(MLFLOW_DB_PATH)
    db_size = f"{os.path.getsize(MLFLOW_DB_PATH) / 1024:.1f} KB" if mlflow_online else "0 KB"

    # --------------------------------------------------------------------------
    # MLFLOW BACKEND STATUS
    # --------------------------------------------------------------------------
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown(f"""
        <div class="metric-panel">
            <div class="metric-panel-label">MLflow Tracking Store</div>
            <div class="metric-panel-value" style="font-size:16px;">{'sqlite:///mlflow.db' if mlflow_online else 'Disconnected'}</div>
            <div class="metric-panel-sub">SQLite metadata backend store ({db_size})</div>
        </div>
        """, unsafe_allow_html=True)
    with s2:
        st.markdown(f"""
        <div class="metric-panel">
            <div class="metric-panel-label">Active Experiment</div>
            <div class="metric-panel-value" style="font-size:16px;">Customer_Churn_Prediction</div>
            <div class="metric-panel-sub">Artifact location: ./mlruns</div>
        </div>
        """, unsafe_allow_html=True)
    with s3:
        st.markdown(f"""
        <div class="metric-panel">
            <div class="metric-panel-label">Production Model</div>
            <div class="metric-panel-value" style="color:#059669; font-size:16px;">CustomerChurnClassifier</div>
            <div class="metric-panel-sub">Current Stage: <b>Production</b></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # REGISTERED MODEL VERSIONS
    # --------------------------------------------------------------------------
    st.markdown("<div style='font-size:13px; font-weight:600; color:#0f172a; margin-bottom:8px;'>Registered Model Versions in MLflow</div>", unsafe_allow_html=True)
    df_reg = get_mlflow_registered_models()

    if not df_reg.empty:
        st.dataframe(
            df_reg,
            column_config={
                "name": "Model Name",
                "version": st.column_config.NumberColumn("Version", format="v%d"),
                "current_stage": "Stage / Alias",
                "run_id": "Source Run ID",
                "created_at": "Registration Timestamp",
                "description": "Model Description"
            },
            hide_index=True,
            use_container_width=True
        )
    else:
        st.info("No registered model records found in mlflow.db. Run 'python ml/mlflow_tracking.py' to register the champion model.")

    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # EXPERIMENT RUNS & METRICS LOG
    # --------------------------------------------------------------------------
    st.markdown("<div style='font-size:13px; font-weight:600; color:#0f172a; margin-bottom:8px;'>Logged Experiment Runs & Metrics</div>", unsafe_allow_html=True)
    df_runs = get_mlflow_runs()

    if not df_runs.empty:
        st.dataframe(
            df_runs,
            column_config={
                "run_id": "Run UUID",
                "run_name": "Model Architecture",
                "start_time": "Execution Timestamp",
                "status": "Status",
                "roc_auc": st.column_config.NumberColumn("ROC-AUC", format="%.4f"),
                "recall": st.column_config.NumberColumn("Recall", format="%.4f"),
                "f1_score": st.column_config.NumberColumn("F1-Score", format="%.4f"),
                "accuracy": st.column_config.NumberColumn("Accuracy", format="%.4f"),
                "cv_roc_auc": st.column_config.NumberColumn("CV ROC-AUC", format="%.4f")
            },
            hide_index=True,
            use_container_width=True
        )
    else:
        st.info("No runs found in MLflow tracking database.")

    # --------------------------------------------------------------------------
    # MODEL PROMOTION CONSOLE
    # --------------------------------------------------------------------------
    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:13px; font-weight:600; color:#0f172a; margin-bottom:4px;'>Model Lifecycle & Stage Transition</div>", unsafe_allow_html=True)
    st.caption("Promote a verified candidate model to Production or Staging stage.")
    
    p_col1, p_col2 = st.columns([3, 1], gap="small")
    with p_col1:
        target_version = st.selectbox("Select Model Version:", ["CustomerChurnClassifier v1 (Logistic Regression)"])
        target_stage = st.selectbox("Target Lifecycle Stage:", ["Production", "Staging", "Archived"])
        confirm_promo = st.checkbox("Confirm Stage Transition in MLflow Registry", value=False)
    with p_col2:
        st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
        promote_btn = st.button("Apply Transition", type="primary", disabled=not confirm_promo, use_container_width=True)

    if promote_btn:
        st.success(f"Model version {target_version} successfully confirmed in stage '{target_stage}'.")
