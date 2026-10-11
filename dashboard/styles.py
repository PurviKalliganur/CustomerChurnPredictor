"""
Enterprise Design System & Theme Engine
---------------------------------------
Sophisticated, cohesive developer-platform UI theme with a restrained
navy and slate color palette, clean typography, compact cards, and responsive sidebar.
"""

ENTERPRISE_THEME_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global Typography & Canvas */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    code, pre, .terminal-text {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Streamlit Main Container Spacing */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2.5rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 1400px !important;
    }

    /* Fixed Proportioned Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0b0f19 !important;
        border-right: 1px solid #1e293b !important;
        min-width: 280px !important;
        max-width: 300px !important;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.25rem !important;
        padding-left: 1.25rem !important;
        padding-right: 1.25rem !important;
    }

    /* Sidebar Platform Identity */
    .platform-brand-block {
        padding: 4px 0 16px 0;
        border-bottom: 1px solid #1e293b;
        margin-bottom: 16px;
    }

    .platform-brand-title {
        font-size: 14px;
        font-weight: 700;
        color: #f8fafc;
        letter-spacing: -0.2px;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .platform-brand-badge {
        display: inline-block;
        font-size: 10px;
        font-weight: 600;
        padding: 1px 6px;
        border-radius: 4px;
        background: rgba(37, 99, 235, 0.2);
        color: #60a5fa;
        border: 1px solid rgba(37, 99, 235, 0.4);
        margin-left: auto;
    }

    .platform-brand-subtitle {
        font-size: 11px;
        color: #64748b;
        margin-top: 4px;
    }

    /* Sidebar Status Block */
    .sidebar-status-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 6px;
        padding: 10px 12px;
        margin-bottom: 18px;
        font-size: 11px;
    }

    .sidebar-status-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 4px;
    }

    .sidebar-status-row:last-child {
        margin-bottom: 0;
    }

    .sidebar-status-lbl {
        color: #64748b;
    }

    .sidebar-status-val {
        color: #e2e8f0;
        font-weight: 600;
    }

    /* Sidebar Navigation Radios Styling */
    [data-testid="stSidebar"] [data-testid="stRadio"] > div {
        gap: 4px !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        padding: 7px 12px !important;
        border-radius: 6px !important;
        font-size: 12.5px !important;
        font-weight: 500 !important;
        color: #94a3b8 !important;
        transition: all 0.15s ease !important;
        margin: 0 !important;
        cursor: pointer !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background: #1e293b !important;
        color: #f8fafc !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-checked="true"],
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
        background: #1e293b !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        border-left: 3px solid #3b82f6 !important;
    }

    /* Top Platform Header Bar */
    .app-header-bar {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 20px;
        display: flex;
        flex-direction: column;
        gap: 8px;
    }

    .app-breadcrumbs {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        color: #64748b;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    .app-breadcrumbs-active {
        color: #38bdf8;
        font-weight: 600;
    }

    .app-header-main {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
    }

    .app-header-title {
        font-size: 20px;
        font-weight: 700;
        color: #f8fafc;
        margin: 0;
        letter-spacing: -0.3px;
    }

    .app-chips-row {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        align-items: center;
    }

    .app-chip {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 4px;
        padding: 3px 8px;
        font-size: 11px;
        font-weight: 500;
        color: #cbd5e1;
    }

    .app-chip-success {
        background: rgba(16, 185, 129, 0.12);
        border-color: rgba(16, 185, 129, 0.3);
        color: #34d399;
    }

    .app-chip-info {
        background: rgba(56, 189, 248, 0.12);
        border-color: rgba(56, 189, 248, 0.3);
        color: #38bdf8;
    }

    /* Compact Information Panels & Metric Cards */
    .panel-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        margin-bottom: 12px;
    }

    .metric-panel {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 14px 16px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
    }

    .metric-panel-label {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        color: #64748b;
        letter-spacing: 0.5px;
    }

    .metric-panel-value {
        font-size: 22px;
        font-weight: 700;
        color: #0f172a;
        margin: 4px 0 2px 0;
        letter-spacing: -0.4px;
    }

    .metric-panel-sub {
        font-size: 11px;
        color: #94a3b8;
    }

    /* Dark Panel Alternative (for operations & metrics) */
    .panel-card-dark {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 16px;
        color: #f8fafc;
        margin-bottom: 14px;
    }

    /* Clean Status Badges */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.2px;
    }

    .badge-success { background: #ecfdf5; color: #059669; border: 1px solid #a7f3d0; }
    .badge-failed { background: #fef2f2; color: #dc2626; border: 1px solid #fecaca; }
    .badge-running { background: #eff6ff; color: #2563eb; border: 1px solid #bfdbfe; }
    .badge-pending { background: #f8fafc; color: #64748b; border: 1px solid #e2e8f0; }

    /* Developer Terminal Log Viewer */
    .log-terminal {
        background: #090d16;
        border: 1px solid #1e293b;
        border-radius: 8px;
        overflow: hidden;
        margin: 14px 0;
    }

    .log-terminal-header {
        background: #0f172a;
        border-bottom: 1px solid #1e293b;
        padding: 8px 14px;
        font-size: 11.5px;
        font-family: 'JetBrains Mono', monospace;
        color: #94a3b8;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .log-terminal-body {
        padding: 14px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        line-height: 1.5;
        color: #cbd5e1;
        background: #090d16;
        max-height: 360px;
        overflow-y: auto;
        white-space: pre-wrap;
        word-break: break-all;
    }

    /* Stage Sequence Progression */
    .stage-pill {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 10px 12px;
        display: flex;
        flex-direction: column;
        gap: 4px;
        transition: border-color 0.15s ease;
    }

    .stage-pill-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .stage-pill-name {
        font-size: 12.5px;
        font-weight: 600;
        color: #0f172a;
    }

    .stage-pill-meta {
        font-size: 11px;
        color: #64748b;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Clean Enterprise Tables */
    div[data-testid="stDataFrame"] {
        border: 1px solid #e2e8f0 !important;
        border-radius: 6px !important;
        overflow: hidden !important;
    }

    /* Form Controls & Buttons */
    button[kind="primary"] {
        background: #2563eb !important;
        border-color: #1d4ed8 !important;
        color: #ffffff !important;
        font-weight: 500 !important;
        font-size: 13px !important;
        border-radius: 6px !important;
        padding: 6px 16px !important;
    }

    button[kind="secondary"] {
        border-color: #cbd5e1 !important;
        background: #ffffff !important;
        color: #1e293b !important;
        font-weight: 500 !important;
        font-size: 13px !important;
        border-radius: 6px !important;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 2px;
    }

    .stTabs [data-baseweb="tab"] {
        font-size: 13px;
        font-weight: 500;
        padding: 6px 14px;
        color: #64748b;
        border-radius: 6px 6px 0 0;
    }

    .stTabs [aria-selected="true"] {
        color: #2563eb !important;
        border-bottom: 2px solid #2563eb !important;
        font-weight: 600 !important;
    }
</style>
"""
