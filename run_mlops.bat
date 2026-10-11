@echo off
title E-Commerce Enterprise MLOps Pipeline Runner
echo ==================================================================
echo   E-Commerce Customer Churn Prediction: MLOps Master Runner
echo ==================================================================
cd /d "%~dp0"

echo.
echo [1/3] Executing End-to-End MLOps Cycle...
python run_mlops.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] MLOps pipeline failed! Please check logs/mlops_pipeline.log.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [2/3] Starting Background FastAPI Inference Microservice (:8000)...
start "FastAPI Inference Service" cmd /k "python -m uvicorn ml.api:app --host 127.0.0.1 --port 8000"

echo.
echo [3/3] Launching Interactive Streamlit Platform (:8501)...
start "" http://localhost:8501
python -m streamlit run dashboard/app.py
pause
