@echo off
set SCRIPT_DIR=%~dp0
set REPO_ROOT=%SCRIPT_DIR%..
cd /d %REPO_ROOT%
conda run --no-capture-output -n impact-synergy-clean python scripts\impact_desktop_app.py
