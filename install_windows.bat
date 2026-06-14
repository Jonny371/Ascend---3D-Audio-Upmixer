@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
echo.
echo  Ascend - Windows installer
echo  ==========================
echo.

REM --- find a Python launcher / interpreter ---
set "PY="
where py >nul 2>&1 && set "PY=py"
if not defined PY ( where python >nul 2>&1 && set "PY=python" )

if not defined PY (
  echo  Python 3 was not found on this PC.
  where winget >nul 2>&1
  if !errorlevel! == 0 (
    echo  Installing Python 3 via winget ...
    winget install -e --id Python.Python.3.12 --accept-source-agreements --accept-package-agreements
    echo.
    echo  Python was installed. Please CLOSE this window and run install_windows.bat again
    echo  ^(a new window picks up the updated PATH^).
    pause
    exit /b 0
  ) else (
    echo  Could not find winget to auto-install Python.
    echo  Please install Python 3 from https://www.python.org/downloads/
    echo  and tick "Add python.exe to PATH" in the installer, then run this file again.
    pause
    exit /b 1
  )
)

echo  Using: %PY%
%PY% install.py
echo.
pause
