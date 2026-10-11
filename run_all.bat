@echo off
title E-Commerce Data Engineering & MLOps Platform Runner
echo ==================================================================
echo   E-Commerce Sales Analytics & MLOps Platform Master Launcher
echo ==================================================================
cd /d "%~dp0"

echo.
echo [1/3] Executing Automated Python ETL Pipeline...
python etl/pipeline.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] ETL Pipeline failed! Please check output above.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [2/3] Executing Automated MLOps Pipeline & Model Registry...
python run_mlops.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] MLOps Pipeline failed! Please check logs.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [3/3] Launching Background FastAPI Microservice & MLOps Platform Dashboard...
echo  - FastAPI Inference API:      http://127.0.0.1:8000/docs
echo  - MLOps Platform Dashboard:   http://localhost:8501
start "FastAPI Inference Microservice" cmd /k "python -m uvicorn ml.api:app --host 127.0.0.1 --port 8000"
start "" http://localhost:8501
python -m streamlit run dashboard/app.py --server.port 8501
pause
