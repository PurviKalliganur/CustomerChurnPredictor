"""
Model Evaluation & Benchmarks
------------------------------
ML experimentation workspace.
Compare candidate models (Logistic Regression, Random Forest, GBDT),
inspect diagnostic curves, review feature importance, and trigger automated retraining.
"""

import os
import sys
import time
import subprocess
import streamlit as st
import pandas as pd
from dashboard.utils import get_evaluation_metrics, PROJECT_DIR, ARTIFACTS_DIR

FEATURE_TAXONOMY = [
    {"Feature": "recency_days", "Description": "Days elapsed since the customer's most recent completed order", "Type": "Numeric"},
    {"Feature": "frequency", "Description": "Total distinct invoice transactions placed by the customer", "Type": "Numeric"},
    {"Feature": "monetary_total", "Description": "Cumulative lifetime revenue spent by the customer ($)", "Type": "Numeric"},
    {"Feature": "avg_order_value", "Description": "Average transaction value per invoice ($)", "Type": "Numeric"},
    {"Feature": "total_quantity", "Description": "Total aggregate physical units purchased", "Type": "Numeric"},
    {"Feature": "avg_basket_size", "Description": "Average item units per invoice transaction", "Type": "Numeric"},
    {"Feature": "unique_products", "Description": "Count of distinct product catalog SKUs purchased", "Type": "Numeric"},
    {"Feature": "tenure_days", "Description": "Days between customer's first purchase and cutoff date", "Type": "Numeric"},
    {"Feature": "purchase_span_days", "Description": "Active days between customer's first and last order", "Type": "Numeric"},
    {"Feature": "cancellation_count", "Description": "Number of cancelled or returned invoices", "Type": "Numeric"},
    {"Feature": "cancellation_rate", "Description": "Ratio of cancelled invoices to total orders", "Type": "Ratio"},
    {"Feature": "is_uk", "Description": "Binary indicator for United Kingdom domicile (1=UK, 0=Export)", "Type": "Binary"},
    {"Feature": "repeat_customer", "Description": "Binary indicator for placing multiple transactions (frequency > 1)", "Type": "Binary"}
]

def render():
    eval_data = get_evaluation_metrics()
    if not eval_data:
        st.warning("No evaluation metrics found in ml/artifacts/. Please execute model training first.")
        return

    champion = eval_data.get("champion_model", "Logistic_Regression")
    models = eval_data.get("models", {})

    # --------------------------------------------------------------------------
    # CHAMPION MODEL HIGHLIGHT BANNER
    # --------------------------------------------------------------------------
    st.markdown(f"""
    <div class="panel-card" style="border-left:4px solid #10b981; margin-bottom:14px;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <span style="font-size:10.5px; font-weight:700; color:#059669; text-transform:uppercase; letter-spacing:0.5px;">Current Production Champion</span>
                <div style="font-size:16px; font-weight:700; color:#0f172a; margin-top:2px;">
                    {champion.replace('_', ' ')} (StandardScaler + L2 Regularization)
                </div>
            </div>
            <span class="badge badge-success">MLflow: Production</span>
        </div>
        <div style="font-size:12px; color:#475569; margin-top:6px; line-height:1.5;">
            <b>Selection Rationale:</b> Selected for superior <b>78.28% Recall</b> on identifying churners in the 90-day future window. In commercial retention operations, false negatives (failing to detect high-value customers about to leave) are significantly more costly than false positives.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # CANDIDATE MODEL BENCHMARK TABLE
    # --------------------------------------------------------------------------
    st.markdown("<div style='font-size:13px; font-weight:600; color:#0f172a; margin-bottom:8px;'>Multi-Model Benchmark Comparison</div>", unsafe_allow_html=True)
    
    table_rows = []
    for m_name, m_stats in models.items():
        is_champ = (m_name == champion)
        table_rows.append({
            "Status": "Production Champion" if is_champ else "Candidate",
            "Model Name": m_name.replace('_', ' '),
            "ROC-AUC": m_stats.get("roc_auc", 0.0),
            "Recall (Churn)": m_stats.get("recall", 0.0),
            "F1-Score": m_stats.get("f1_score", 0.0),
            "Precision": m_stats.get("precision", 0.0),
            "Accuracy": m_stats.get("accuracy", 0.0),
            "PR-AUC": m_stats.get("pr_auc", 0.0),
            "5-Fold CV Mean": m_stats.get("cv_roc_auc_mean", 0.0)
        })

    df_benchmark = pd.DataFrame(table_rows)
    st.dataframe(
        df_benchmark,
        column_config={
            "ROC-AUC": st.column_config.NumberColumn(format="%.4f"),
            "Recall (Churn)": st.column_config.NumberColumn(format="%.4f"),
            "F1-Score": st.column_config.NumberColumn(format="%.4f"),
            "Precision": st.column_config.NumberColumn(format="%.4f"),
            "Accuracy": st.column_config.NumberColumn(format="%.4f"),
            "PR-AUC": st.column_config.NumberColumn(format="%.4f"),
            "5-Fold CV Mean": st.column_config.NumberColumn(format="%.4f"),
        },
        hide_index=True,
        use_container_width=True
    )

    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # DIAGNOSTIC ARTIFACT VISUALIZATIONS
    # --------------------------------------------------------------------------
    st.markdown("<div style='font-size:13px; font-weight:600; color:#0f172a; margin-bottom:8px;'>Diagnostic Evaluation Artifacts</div>", unsafe_allow_html=True)
    
    t_roc, t_pr, t_cm, t_fi = st.tabs([
        "ROC Curves Comparison",
        "Precision-Recall Curves",
        "Confusion Matrix",
        "Feature Importance Rankings"
    ])

    with t_roc:
        roc_path = os.path.join(ARTIFACTS_DIR, "roc_curves.png")
        if os.path.exists(roc_path):
            st.image(roc_path, caption="ROC Curves — Logistic Regression achieves 0.7252 AUC against candidate benchmarks.", use_container_width=True)
        else:
            st.warning("ROC curves plot not found.")

    with t_pr:
        pr_path = os.path.join(ARTIFACTS_DIR, "pr_curves.png")
        if os.path.exists(pr_path):
            st.image(pr_path, caption="Precision-Recall Curves — Evaluated under natural 43% cohort churn base rate.", use_container_width=True)
        else:
            st.warning("PR curves plot not found.")

    with t_cm:
        cm_path = os.path.join(ARTIFACTS_DIR, "confusion_matrix.png")
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Champion Confusion Matrix — 227 true churners captured out of 290 test positives.", use_container_width=True)
        else:
            st.warning("Confusion matrix plot not found.")

    with t_fi:
        fi_path = os.path.join(ARTIFACTS_DIR, "feature_importance.png")
        if os.path.exists(fi_path):
            st.image(fi_path, caption="Feature Importance — Recency, Monetary Total, and Frequency are dominant predictors.", use_container_width=True)
        else:
            st.warning("Feature importance plot not found.")

    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # FEATURE STORE SCHEMA TAXONOMY
    # --------------------------------------------------------------------------
    st.markdown("<div style='font-size:13px; font-weight:600; color:#0f172a; margin-bottom:8px;'>Customer Feature Store Schema (13 Features)</div>", unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(FEATURE_TAXONOMY), use_container_width=True, hide_index=True)

    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # RETRAIN TRIGGER CONTROLS
    # --------------------------------------------------------------------------
    st.markdown("<div style='font-size:13px; font-weight:600; color:#0f172a; margin-bottom:4px;'>Retrain Candidate Models</div>", unsafe_allow_html=True)
    st.caption("Executes multi-model training pipeline (`ml/train.py`) with 5-Fold Stratified Cross-Validation.")
    
    retrain_col1, retrain_col2 = st.columns([3, 1], gap="small")
    with retrain_col1:
        confirm_retrain = st.checkbox("Confirm automated model retraining execution", value=False)
    with retrain_col2:
        do_retrain = st.button("Execute Retraining", type="primary", disabled=not confirm_retrain, use_container_width=True)

    if do_retrain:
        with st.spinner("Executing model training pipeline (ml/train.py)..."):
            t0 = time.time()
            try:
                train_script = os.path.join(PROJECT_DIR, 'ml', 'train.py')
                res = subprocess.run([sys.executable, train_script], cwd=PROJECT_DIR, capture_output=True, text=True)
                if res.returncode == 0:
                    st.success(f"Model retraining completed successfully in {time.time() - t0:.2f}s!")
                    st.rerun()
                else:
                    st.error(f"Retraining failed: {res.stderr}")
            except Exception as e:
                st.error(f"Failed to execute training process: {e}")
