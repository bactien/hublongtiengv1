@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo ===================================================================
echo   🦜 TELEGRAM BOT LỒNG TIẾNG TỰ ĐỘNG PHONG CÁCH ADAM BỰA
echo ===================================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [LỖI] Không tìm thấy môi trường ảo .venv!
    pause
    exit /b 1
)

.\.venv\Scripts\python.exe telegram_bot.py %*

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Bot đã dừng hoặc gặp sự cố. Nhấn phím bất kỳ để thoát...
    pause
)
