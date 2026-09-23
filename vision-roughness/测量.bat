@echo off
title SurfaceIQ Measure
cd /d "%~dp0"
if "%~1"=="" (
    start "" venv\Scripts\pythonw.exe src\gui_ring.py
) else (
    if "%~x1"=="" (
        start "" venv\Scripts\pythonw.exe src\gui_ring.py %*
    ) else (
        start "" venv\Scripts\pythonw.exe src\gui_ring.py --image "%~1"
    )
)
