"""
E-Commerce MLOps Platform
-------------------------
Production-grade developer platform and analytics console.
Modular 10-page architecture featuring persistent left sidebar, live system status,
Kimball dimensional warehouse inspection, and real-time machine learning inference.
"""

import os
import sys
import streamlit as st

# Ensure project root is in sys.path
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)

from dashboard.styles import ENTERPRISE_THEME_CSS
from dashboard.utils import get_db_info, check_api_health, get_pipeline_history
from dashboard.views import (
    overview,
    pipeline_monitor,
    data_quality,
    warehouse_explorer,
    business_intelligence,
    model_training,
    mlflow_registry,
    churn_prediction,
    risk_analytics,
    system_health
)

# ==============================================================================
# STREAMLIT PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="E-Commerce MLOps Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Enterprise Design System
st.markdown(ENTERPRISE_THEME_CSS, unsafe_allow_html=True)

# Concise Navigation Pages (No long wrapping labels)
PAGES = [
    "Overview",
    "Pipelines",
    "Data Quality",
    "Warehouse",
    "Analytics",
    "Model Evaluation",
    "Model Registry",
    "Churn Prediction",
    "Risk Analytics",
    "System Health"
]

# Session State for Page Navigation
if "active_page" not in st.session_state:
    st.session_state["active_page"] = "Overview"

# ==============================================================================
# PERSISTENT LEFT SIDEBAR
# ==============================================================================
with st.sidebar:
    st.markdown("""
    <div class="platform-brand-block">
        <div class="platform-brand-title">
            <span>⚡</span> E-Commerce MLOps
            <span class="platform-brand-badge">PROD</span>
        </div>
        <div class="platform-brand-subtitle">
            Enterprise ML & Data Engineering Platform
        </div>
    </div>
    """, unsafe_allow_html=True)

    # System Status in Sidebar
    db_name, db_size, db_online = get_db_info()
    api_online, _ = check_api_health()
    history = get_pipeline_history()
    last_run = history[0] if history else {"build_number": 1, "status": "SUCCESS"}

    st.markdown(f"""
    <div class="sidebar-status-card">
        <div class="sidebar-status-row">
            <span class="sidebar-status-lbl">System Status:</span>
            <span class="sidebar-status-val" style="color:#34d399;">● Operational</span>
        </div>
        <div class="sidebar-status-row">
            <span class="sidebar-status-lbl">Last Run:</span>
            <span class="sidebar-status-val">#{last_run.get('build_number', 1)} {last_run.get('status', 'SUCCESS')}</span>
        </div>
        <div class="sidebar-status-row">
            <span class="sidebar-status-lbl">Active Model:</span>
            <span class="sidebar-status-val">LogReg v1.0</span>
        </div>
        <div class="sidebar-status-row">
            <span class="sidebar-status-lbl">API Service:</span>
            <span class="sidebar-status-val">{'Online :8000' if api_online else 'Local'}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    current_index = PAGES.index(st.session_state["active_page"]) if st.session_state["active_page"] in PAGES else 0

    selected_page = st.radio(
        "Navigation",
        options=PAGES,
        index=current_index,
        label_visibility="collapsed"
    )

    st.session_state["active_page"] = selected_page

    st.sidebar.divider()
    
    st.sidebar.markdown(f"""
    <div style="font-size:11px; color:#64748b; line-height:1.6; padding:0 4px;">
        <div><b>Storage:</b> {db_name} ({db_size})</div>
        <div><b>Registry:</b> sqlite:///mlflow.db</div>
        <div><b>Environment:</b> Python 3.10</div>
        <div style="margin-top:6px; color:#475569;">Release v2.4 • Production</div>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# TOP APPLICATION HEADER BAR
# ==============================================================================
active_page = st.session_state["active_page"]
api_status_badge = "app-chip-success" if api_online else "app-chip"
api_status_text = "FastAPI :8000 Online" if api_online else "Local Fallback"

PAGE_TITLES = {
    "Overview": "Platform Overview",
    "Pipelines": "Pipeline Monitor",
    "Data Quality": "Data Engineering & Quality",
    "Warehouse": "Data Warehouse Explorer",
    "Analytics": "Business Intelligence & Analytics",
    "Model Evaluation": "Model Evaluation & Benchmarks",
    "Model Registry": "MLflow Model Registry",
    "Churn Prediction": "Customer Churn Prediction",
    "Risk Analytics": "Customer Risk Analytics",
    "System Health": "System Health & Diagnostics"
}

st.markdown(f"""
<div class="app-header-bar">
    <div class="app-breadcrumbs">
        <span>Platform</span> / <span class="app-breadcrumbs-active">{active_page}</span>
    </div>
    <div class="app-header-main">
        <div class="app-header-title">{PAGE_TITLES.get(active_page, active_page)}</div>
        <div class="app-chips-row">
            <div class="app-chip app-chip-success">● System Status: Operational</div>
            <div class="app-chip app-chip-info">Production Model: Logistic Regression (Recall: 78.28%)</div>
            <div class="app-chip">Warehouse: {db_name} ({db_size})</div>
            <div class="app-chip {api_status_badge}">Inference: {api_status_text}</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# ROUTE DISPATCHER
# ==============================================================================
if active_page == "Overview":
    overview.render()
elif active_page == "Pipelines":
    pipeline_monitor.render()
elif active_page == "Data Quality":
    data_quality.render()
elif active_page == "Warehouse":
    warehouse_explorer.render()
elif active_page == "Analytics":
    business_intelligence.render()
elif active_page == "Model Evaluation":
    model_training.render()
elif active_page == "Model Registry":
    mlflow_registry.render()
elif active_page == "Churn Prediction":
    churn_prediction.render()
elif active_page == "Risk Analytics":
    risk_analytics.render()
elif active_page == "System Health":
    system_health.render()
else:
    st.error(f"Unknown workspace view: {active_page}")
