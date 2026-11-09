@echo off
rem Windows: double-click to start Demand ^& Inventory Planner (first start installs what is needed).
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (
  py -3 launcher.py %*
) else (
  where python >nul 2>nul
  if errorlevel 1 (
    echo Python 3.11 or newer is needed. Install it from https://www.python.org/downloads/ ^(tick "Add python.exe to PATH"^) and double-click again.
    pause
    exit /b 1
  )
  python launcher.py %*
)
if errorlevel 1 pause
