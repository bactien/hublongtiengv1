@echo off
setlocal
cd /d "%~dp0"

echo =======================================================
echo   🦜 DUBBING PHONG CACH ADAM BUA (XUNG ANH - CAC CON VO)
echo =======================================================

if "%~1"=="" (
    echo Dang chay thu nghiem tren video mau e2e-tests\englishvideo.mp4...
    .\.venv\Scripts\python.exe auto_dub.py --input_file e2e-tests\englishvideo.mp4 --voice "Adam bựa" --style teen_bua --burn_subtitles
) else (
    .\.venv\Scripts\python.exe auto_dub.py --input_file "%~1" --voice "Adam bựa" --style teen_bua --burn_subtitles %2 %3 %4 %5
)
