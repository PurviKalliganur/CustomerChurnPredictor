@echo off
echo ====================================================
echo Starting E-Commerce ETL Pipeline...
echo ====================================================
cd /d "%~dp0"
if exist "E_Commerce Analytics\etl\pipeline.py" cd "E_Commerce Analytics"
python etl/pipeline.py
pause
