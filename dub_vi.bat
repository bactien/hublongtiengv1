@echo off
setlocal
cd /d "%~dp0"

set "VOICE=Adam"
set "ENGINE=vieneu"

if "%~1"=="" (
    echo [Cach su dung]: dub_vi.bat ^<duong_dan_video.mp4^> [ten_giong] [--burn_subtitles]
    echo Cac giong Adam pho bien: "Adam" (Mien Nam), "Adam bua" (Mien Bac)
    echo Vi du: .\dub_vi.bat samples\jobinterview.mp4 "Adam" --burn_subtitles
    echo.
    echo Dang chay thu nghiem voi giong Adam tren video mau e2e-tests\englishvideo.mp4...
    .\.venv\Scripts\python.exe auto_dub.py --input_file e2e-tests\englishvideo.mp4 --tts_engine vieneu --voice Adam --burn_subtitles
) else (
    .\.venv\Scripts\python.exe auto_dub.py --input_file "%~1" --tts_engine vieneu --voice "%VOICE%" %2 %3 %4 %5
)
