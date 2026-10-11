"""
Pipeline Monitor
----------------
Engineering operations workspace for orchestrating automated data pipelines,
tracking stage execution states, viewing real-time execution logs, and inspecting run history.
"""

import os
import sys
import time
import subprocess
from datetime import datetime
import streamlit as st
import pandas as pd
from dashboard.utils import get_pipeline_history, record_pipeline_run, PROJECT_DIR

def render():
    # Initialize session state for execution
    if "pipeline_running" not in st.session_state:
        st.session_state["pipeline_running"] = False
    if "latest_log" not in st.session_state:
        st.session_state["latest_log"] = ""
    if "last_run_status" not in st.session_state:
        st.session_state["last_run_status"] = None

    history = get_pipeline_history()
    last_run = history[0] if history else None
    is_running = st.session_state["pipeline_running"]

    # Current execution status indicator
    if is_running:
        status_badge = '<span class="badge badge-running">● Running</span>'
    elif st.session_state.get("last_run_status") == "SUCCESS":
        status_badge = '<span class="badge badge-success">● Last Run: Success</span>'
    elif st.session_state.get("last_run_status") == "FAILED":
        status_badge = '<span class="badge badge-failed">● Last Run: Failed</span>'
    else:
        status_badge = '<span class="badge badge-pending">● System Ready</span>'

    # --------------------------------------------------------------------------
    # PAGE HEADER & OPERATIONAL TOOLBAR
    # --------------------------------------------------------------------------
    st.markdown(f"""
    <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:12px;">
        <div>
            <div style="font-size:18px; font-weight:700; color:#0f172a; letter-spacing:-0.2px;">Pipeline Monitor</div>
            <div style="font-size:12px; color:#64748b; margin-top:2px;">
                Automated data engineering, model retraining, and customer scoring pipeline orchestrator.
            </div>
        </div>
        <div>{status_badge}</div>
    </div>
    """, unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # PIPELINE STAGE SEQUENCE (Reflects real implementation)
    # --------------------------------------------------------------------------
    st.markdown("<div style='font-size:12px; font-weight:600; text-transform:uppercase; color:#64748b; margin-bottom:8px; letter-spacing:0.5px;'>Pipeline Execution Stages</div>", unsafe_allow_html=True)

    stages = [
        {"idx": "1", "name": "Data Ingestion", "desc": "Ingest raw Excel sales facts", "time": "2.1s"},
        {"idx": "2", "name": "Data Validation", "desc": "Segregate dirty rows to quarantine", "time": "4.8s"},
        {"idx": "3", "name": "Warehouse Modeling", "desc": "Kimball Star Schema facts & dims", "time": "5.3s"},
        {"idx": "4", "name": "Feature Engineering", "desc": "13 leakage-free temporal features", "time": "3.2s"},
        {"idx": "5", "name": "Model Training", "desc": "LogReg, RF, GBDT + 5-fold CV", "time": "6.8s"},
        {"idx": "6", "name": "Model Evaluation", "desc": "Calculate metrics & diagnostic plots", "time": "1.5s"},
        {"idx": "7", "name": "Model Registry", "desc": "Register champion in MLflow", "time": "2.4s"},
        {"idx": "8", "name": "Batch Scoring", "desc": "Score 4,338 customer entities", "time": "1.9s"},
        {"idx": "9", "name": "Automated Tests", "desc": "Validate microservice endpoints", "time": "1.2s"}
    ]

    # Render stage sequence in responsive grid
    st_cols = st.columns(len(stages))
    for i, stage in enumerate(stages):
        with st_cols[i]:
            if is_running:
                border_color = "#3b82f6"
                status_text = "Running"
                status_class = "color:#3b82f6;"
            elif last_run and last_run.get("status") == "SUCCESS":
                border_color = "#10b981"
                status_text = "Verified"
                status_class = "color:#059669;"
            else:
                border_color = "#cbd5e1"
                status_text = "Ready"
                status_class = "color:#64748b;"

            st.markdown(f"""
            <div style="background:#ffffff; border:1px solid #e2e8f0; border-top:3px solid {border_color}; border-radius:6px; padding:8px 10px; height:105px; display:flex; flex-direction:column; justify-content:space-between; box-shadow:0 1px 2px rgba(0,0,0,0.02);">
                <div>
                    <div style="font-size:10px; font-weight:700; color:#94a3b8; text-transform:uppercase;">Stage {stage['idx']}</div>
                    <div style="font-weight:600; font-size:11.5px; color:#0f172a; margin-top:2px; line-height:1.2;">{stage['name']}</div>
                    <div style="font-size:10px; color:#64748b; margin-top:2px; line-height:1.2;">{stage['desc']}</div>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:center; border-top:1px solid #f1f5f9; padding-top:4px;">
                    <span style="font-size:10px; {status_class} font-weight:600;">{status_text}</span>
                    <span style="font-size:10px; color:#94a3b8; font-family:'JetBrains Mono';">{stage['time']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # COMPACT PIPELINE CONTROLS TOOLBAR
    # --------------------------------------------------------------------------
    st.markdown("""
    <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:6px; padding:12px 16px; margin-bottom:14px; box-shadow:0 1px 2px rgba(0,0,0,0.02);">
        <div style="font-size:12px; font-weight:600; color:#0f172a; margin-bottom:2px;">Pipeline Execution Console</div>
        <div style="font-size:11px; color:#64748b;">Configure execution parameters and trigger automated pipelines safely.</div>
    </div>
    """, unsafe_allow_html=True)

    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([3, 1.5, 1.5], gap="small")
    with ctrl_col1:
        pipeline_choice = st.selectbox(
            "Target Pipeline:",
            [
                "Master MLOps Lifecycle (run_mlops.py)",
                "Warehouse Data Processing (etl/pipeline.py)",
                "Model Retraining & Scoring (ml/train.py & ml/predict.py)"
            ],
            disabled=is_running,
            label_visibility="collapsed"
        )
    with ctrl_col2:
        confirm_run = st.checkbox("Confirm Execution", value=False, disabled=is_running)
    with ctrl_col3:
        run_btn = st.button("Execute Pipeline", type="primary", disabled=is_running or not confirm_run, use_container_width=True)

    # --------------------------------------------------------------------------
    # EXECUTION LOGS (Developer-tool Terminal Viewer)
    # --------------------------------------------------------------------------
    st.markdown("<div style='display:flex; justify-content:space-between; align-items:center; margin-top:12px; margin-bottom:6px;'><span style='font-size:13px; font-weight:600; color:#0f172a;'>Execution Logs</span><span style='font-size:11px; color:#64748b;'>stdout & stderr stream</span></div>", unsafe_allow_html=True)
    
    terminal_placeholder = st.empty()

    if not st.session_state["latest_log"]:
        terminal_placeholder.markdown("""
        <div class="log-terminal">
            <div class="log-terminal-header">
                <span>TERMINAL: idle // awaiting execution</span>
                <span>STATUS: READY</span>
            </div>
            <div class="log-terminal-body" style="color:#64748b;">
[IDLE] No active execution. Select a pipeline and click 'Execute Pipeline' to stream real-time logs.
[INFO] Output will stream here with execution timestamps, process details, and exit codes.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        status_text = st.session_state.get("last_run_status", "COMPLETED")
        terminal_placeholder.markdown(f"""
        <div class="log-terminal">
            <div class="log-terminal-header">
                <span>TERMINAL: pipeline-execution.log</span>
                <span>STATUS: {status_text}</span>
            </div>
            <div class="log-terminal-body">
{st.session_state['latest_log']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Subprocess Pipeline Execution Handler
    if run_btn and not is_running:
        st.session_state["pipeline_running"] = True
        st.session_state["latest_log"] = f"[{datetime.now().strftime('%H:%M:%S')}] Initializing pipeline execution: {pipeline_choice}...\n"
        
        script_to_run = "run_mlops.py"
        pipeline_label = "Master MLOps Lifecycle"
        if "etl/pipeline.py" in pipeline_choice:
            script_to_run = os.path.join("etl", "pipeline.py")
            pipeline_label = "Warehouse Data Processing"
        elif "ml/train.py" in pipeline_choice:
            script_to_run = os.path.join("ml", "train.py")
            pipeline_label = "Model Retraining & Scoring"

        script_path = os.path.join(PROJECT_DIR, script_to_run)
        
        t_start = time.time()
        start_ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        with st.spinner(f"Executing {pipeline_label}... Streaming logs..."):
            try:
                proc = subprocess.Popen(
                    [sys.executable, script_path],
                    cwd=PROJECT_DIR,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )

                accumulated_logs = f"[{start_ts}] PIPELINE TRIGGERED: {pipeline_label}\n"
                accumulated_logs += f"[EXEC] {sys.executable} {script_to_run}\n" + ("=" * 65) + "\n"

                for line in iter(proc.stdout.readline, ''):
                    accumulated_logs += line
                    st.session_state["latest_log"] = accumulated_logs
                    terminal_placeholder.markdown(f"""
                    <div class="log-terminal">
                        <div class="log-terminal-header">
                            <span>TERMINAL: {script_to_run} (PID {proc.pid})</span>
                            <span style="color:#38bdf8;">EXECUTING...</span>
                        </div>
                        <div class="log-terminal-body">
{accumulated_logs}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                proc.stdout.close()
                return_code = proc.wait()
                duration = time.time() - t_start

                status = "SUCCESS" if return_code == 0 else "FAILED"
                accumulated_logs += f"\n" + ("=" * 65) + f"\n[{datetime.now().strftime('%H:%M:%S')}] PROCESS TERMINATED WITH EXIT CODE {return_code} ({status}) IN {duration:.2f}s\n"
                st.session_state["latest_log"] = accumulated_logs
                st.session_state["last_run_status"] = status

                # Update persistent execution history
                next_run_num = len(history) + 1
                new_record = {
                    "build_number": next_run_num,
                    "pipeline_type": pipeline_label,
                    "timestamp": start_ts,
                    "duration_seconds": round(duration, 2),
                    "status": status,
                    "triggered_by": "Platform Operator",
                    "stages_completed": "All Stages Completed" if status == "SUCCESS" else f"Exited with code {return_code}"
                }
                record_pipeline_run(new_record)

                if status == "SUCCESS":
                    st.success(f"Pipeline Run #{next_run_num} completed successfully in {duration:.2f}s!")
                else:
                    st.error(f"Pipeline Run #{next_run_num} failed with return code {return_code}. Check execution logs.")

            except Exception as e:
                st.session_state["latest_log"] += f"\n[CRITICAL ERROR] Failed to spawn process: {e}\n"
                st.error(f"Execution Error: {e}")
            finally:
                st.session_state["pipeline_running"] = False
                st.rerun()

    # --------------------------------------------------------------------------
    # PIPELINE RUN HISTORY
    # --------------------------------------------------------------------------
    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:13px; font-weight:600; color:#0f172a; margin-bottom:8px;'>Pipeline Run History</div>", unsafe_allow_html=True)
    
    history_records = get_pipeline_history()
    filter_col, search_col = st.columns([1, 3], gap="small")
    with filter_col:
        status_filter = st.selectbox("Status Filter:", ["All", "SUCCESS", "FAILED"], label_visibility="collapsed")
    with search_col:
        text_filter = st.text_input("Search runs by name or trigger:", "", placeholder="Filter by pipeline name, status, or trigger source...", label_visibility="collapsed")

    filtered_history = [
        r for r in history_records
        if (status_filter == "All" or r.get("status") == status_filter)
        and (not text_filter or text_filter.lower() in str(r).lower())
    ]

    if filtered_history:
        st.dataframe(
            pd.DataFrame(filtered_history),
            column_config={
                "build_number": st.column_config.NumberColumn("Run ID", format="#%d"),
                "pipeline_type": "Pipeline Name",
                "timestamp": "Start Time",
                "duration_seconds": st.column_config.NumberColumn("Duration", format="%.2fs"),
                "status": "Outcome",
                "triggered_by": "Trigger Source",
                "stages_completed": "Stage Summary"
            },
            hide_index=True,
            use_container_width=True
        )
    else:
        st.info("No pipeline run records match the selected filter.")
