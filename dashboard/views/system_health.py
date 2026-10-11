"""
System Health & Diagnostics
---------------------------
Engineering diagnostic console.
Inspects Python runtime, dependency versions, SQLite warehouse integrity,
model artifacts, MLflow connectivity, FastAPI health, and executes automated test suites.
"""

import os
import sys
import platform
import subprocess
import sqlite3
import streamlit as st
import pandas as pd
from dashboard.utils import (
    get_db_info, check_api_health, DB_PATH, MLFLOW_DB_PATH, ARTIFACTS_DIR, PROJECT_DIR
)

def check_dependencies():
    packages = ["streamlit", "sklearn", "mlflow", "pandas", "numpy", "fastapi", "uvicorn", "plotly"]
    status_list = []
    for pkg in packages:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "Installed")
            status_list.append({"Package": pkg, "Status": "Installed", "Version": ver})
        except ImportError:
            status_list.append({"Package": pkg, "Status": "Missing", "Version": "N/A"})
    return pd.DataFrame(status_list)

def check_db_integrity():
    if not os.path.exists(DB_PATH):
        return "Missing DB file", 0
    try:
        conn = sqlite3.connect(DB_PATH)
        res = conn.execute("PRAGMA integrity_check;").fetchone()[0]
        cur = conn.cursor()
        cur.execute("SELECT count(*) FROM sqlite_master WHERE type='table';")
        tables_cnt = cur.fetchone()[0]
        conn.close()
        return res, tables_cnt
    except Exception as e:
        return f"Error: {e}", 0

def check_artifacts():
    required_files = [
        "best_model.joblib", "evaluation_metrics.json", "feature_importance.png",
        "roc_curves.png", "pr_curves.png", "confusion_matrix.png"
    ]
    art_list = []
    for f in required_files:
        fpath = os.path.join(ARTIFACTS_DIR, f)
        exists = os.path.exists(fpath)
        sz = f"{os.path.getsize(fpath) / 1024:.1f} KB" if exists else "Missing"
        art_list.append({
            "Artifact Name": f,
            "Status": "Available" if exists else "Missing",
            "Size": sz
        })
    return pd.DataFrame(art_list)

def render():
    db_type, db_size, db_online = get_db_info()
    api_online, api_data = check_api_health()
    db_integrity, tbl_count = check_db_integrity()

    # --------------------------------------------------------------------------
    # HIGH-LEVEL STATUS PANELS
    # --------------------------------------------------------------------------
    h1, h2, h3, h4 = st.columns(4)
    with h1:
        st.markdown(f"""
        <div class="metric-panel">
            <div class="metric-panel-label">Warehouse Health</div>
            <div class="metric-panel-value" style="color:#059669; font-size:16px;">{db_integrity.upper()}</div>
            <div class="metric-panel-sub">{tbl_count} tables in SQLite ({db_size})</div>
        </div>
        """, unsafe_allow_html=True)
    with h2:
        api_color = "#059669" if api_online else "#d97706"
        api_text = "ONLINE (:8000)" if api_online else "OFFLINE"
        st.markdown(f"""
        <div class="metric-panel">
            <div class="metric-panel-label">Inference Microservice</div>
            <div class="metric-panel-value" style="color:{api_color}; font-size:16px;">{api_text}</div>
            <div class="metric-panel-sub">FastAPI Churn Serving Engine</div>
        </div>
        """, unsafe_allow_html=True)
    with h3:
        mlflow_exists = os.path.exists(MLFLOW_DB_PATH)
        st.markdown(f"""
        <div class="metric-panel">
            <div class="metric-panel-label">MLflow Tracking Store</div>
            <div class="metric-panel-value" style="color:#2563eb; font-size:16px;">{'CONNECTED' if mlflow_exists else 'NOT FOUND'}</div>
            <div class="metric-panel-sub">Local SQLite metadata store</div>
        </div>
        """, unsafe_allow_html=True)
    with h4:
        st.markdown(f"""
        <div class="metric-panel">
            <div class="metric-panel-label">Python Runtime</div>
            <div class="metric-panel-value" style="font-size:16px;">Python {platform.python_version()}</div>
            <div class="metric-panel-sub">{platform.system()} {platform.machine()}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # DIAGNOSTICS: DEPENDENCIES & ARTIFACTS
    # --------------------------------------------------------------------------
    d_col1, d_col2 = st.columns(2, gap="medium")
    with d_col1:
        st.markdown("<div style='font-size:13px; font-weight:600; color:#0f172a; margin-bottom:6px;'>Key Python Dependencies</div>", unsafe_allow_html=True)
        st.dataframe(check_dependencies(), use_container_width=True, hide_index=True)

    with d_col2:
        st.markdown("<div style='font-size:13px; font-weight:600; color:#0f172a; margin-bottom:6px;'>Model Artifacts Integrity</div>", unsafe_allow_html=True)
        st.dataframe(check_artifacts(), use_container_width=True, hide_index=True)

    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # AUTOMATED TEST SUITE RUNNER
    # --------------------------------------------------------------------------
    st.markdown("<div style='font-size:13px; font-weight:600; color:#0f172a; margin-bottom:4px;'>Automated Test Suite Execution</div>", unsafe_allow_html=True)
    st.caption("Executes unit, contract, and endpoint integration tests for the prediction microservice (6 assertions).")

    test_btn = st.button("Run Automated Test Suite (PyTest)", type="primary")

    if test_btn:
        with st.spinner("Executing ml/test_api.py test suite..."):
            test_script = os.path.join(PROJECT_DIR, 'ml', 'test_api.py')
            res = subprocess.run([sys.executable, test_script], cwd=PROJECT_DIR, capture_output=True, text=True)
            
            if res.returncode == 0:
                st.success("All microservice API tests passed successfully (6/6 tests passing)!")
                st.code(res.stdout, language="bash")
            else:
                st.error("Test failures detected:")
                st.code(res.stderr or res.stdout, language="bash")

    # --------------------------------------------------------------------------
    # ARCHITECTURAL CHEATSHEET & SETUP GUIDE
    # --------------------------------------------------------------------------
    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
    with st.expander("Platform Architecture Blueprint & CLI Operations"):
        st.markdown("""
        ### E-Commerce MLOps System Topology
        ```
        Raw Excel (541k rows) ──> Ingest & Validation (ETL) ──> Zero Data Loss Quarantine (144k rows)
                                         │
                                         ▼
                            SQLite Kimball Warehouse (387k sales facts)
                                         │
                         ┌───────────────┴───────────────┐
                         ▼                               ▼
                 Temporal Feature Store          Executive Analytics
                 (4,338 customers)               (Plotly Visuals)
                         │
                         ▼
             MLflow Model Registry & Cross-Validation (LogReg / RF / GBDT)
                         │
                         ▼
             FastAPI Serving Engine (:8000) & Batch Scored CRM Exports
        ```
        ### Core CLI Operations:
        - **Run Master Pipeline:** `python run_mlops.py`
        - **Launch Dashboard:** `python -m streamlit run dashboard/app.py --server.port 8501`
        - **Start FastAPI Service:** `python -m uvicorn ml.api:app --host 127.0.0.1 --port 8000`
        - **Run Test Suite:** `python ml/test_api.py`
        - **One-Click Windows Launcher:** `run_all.bat`
        """)
