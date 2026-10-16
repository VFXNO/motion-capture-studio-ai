@echo off
cd /d "%~dp0"
py -3.12 -c "import sys" >nul 2>nul
if not errorlevel 1 goto python312
where python >nul 2>nul
if errorlevel 1 goto missing
start "" http://127.0.0.1:8765
python serve.py
pause
exit /b
:python312
start "" http://127.0.0.1:8765
py -3.12 serve.py
pause
exit /b
:missing
echo Install Python 3.12, then run this launcher again.
pause
exit /b 1
