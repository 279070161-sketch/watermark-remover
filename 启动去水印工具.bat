@echo off
chcp 65001 >nul
title AI 智能去水印工具 (LaMa + OpenCV)
cd /d "%~dp0"
echo =======================================================
echo   正在启动 AI 智能去水印工具，请稍候...
echo   启动后将自动打开浏览器页面 (http://127.0.0.1:8765)
echo =======================================================
".venv\Scripts\python.exe" app.py
pause
