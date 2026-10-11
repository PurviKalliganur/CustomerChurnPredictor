@echo off
echo ====================================================
echo Launching E-Commerce MLOps Platform Dashboard...
echo ====================================================
cd /d "%~dp0"
python -m streamlit run dashboard/app.py --server.port 8501
pause
