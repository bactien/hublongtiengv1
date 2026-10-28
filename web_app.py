import os
import sys
import time
import shutil
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import psutil

# Nạp cấu hình từ .env
from dotenv import load_dotenv
load_dotenv()

from auto_dub import process_video

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("WebApp")

app = FastAPI(title="Hub Lồng Tiếng AI - Quản Lý & Biên Tập Video")

BASE_DIR = Path(__file__).resolve().parent
DOWNLOADS_DIR = BASE_DIR / "downloads"
OUTPUT_DIR = BASE_DIR / "output"
DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Quản lý trạng thái các tác vụ lồng tiếng
JOBS: Dict[str, Dict[str, Any]] = {}
PROCESSING_LOCK = asyncio.Lock()


def get_system_stats():
    """Lấy thông số CPU, RAM và dung lượng ổ cứng."""
    try:
        cpu = psutil.cpu_percent(interval=0.1)
        ram = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        return {
            "cpu_percent": cpu,
            "ram_used_gb": round((ram.total - ram.available) / (1024**3), 2),
            "ram_total_gb": round(ram.total / (1024**3), 2),
            "ram_percent": ram.percent,
            "disk_free_gb": round(disk.free / (1024**3), 2),
            "disk_total_gb": round(disk.total / (1024**3), 2),
            "disk_percent": disk.percent,
        }
    except Exception:
        return {}


def scan_processed_files():
    """Quét toàn bộ video thành phẩm trong thư mục output (bao gồm từ Telegram và Web)."""
    items = []
    if not OUTPUT_DIR.exists():
        return items

    for sub in sorted(OUTPUT_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if sub.is_dir() and sub.name != "temp":
            videos = list(sub.glob("*.mp4"))
            srts = list(sub.glob("*.srt"))
            jsons = list(sub.glob("*.json"))

            if videos:
                vid = videos[0]
                stat = vid.stat()
                items.append({
                    "id": sub.name,
                    "title": vid.name,
                    "video_url": f"/stream/{sub.name}/{vid.name}",
                    "download_url": f"/download/{sub.name}/{vid.name}",
                    "srt_url": f"/download/{sub.name}/{srts[0].name}" if srts else None,
                    "json_url": f"/download/{sub.name}/{jsons[0].name}" if jsons else None,
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "created_at": time.strftime("%H:%M %d/%m/%Y", time.localtime(stat.st_mtime)),
                    "timestamp": stat.st_mtime,
                })
    return items


def run_dubbing_job(job_id: str, input_path: str, output_subdir: str, voice: str, style: str, burn_subs: bool, model: str):
    """Tiến trình thực thi lồng tiếng ngầm."""
    try:
        JOBS[job_id]["status"] = "processing"
        JOBS[job_id]["step"] = 1
        JOBS[job_id]["message"] = "Đang bắt đầu xử lý..."

        def progress_cb(step, total, desc):
            JOBS[job_id]["step"] = step
            JOBS[job_id]["total"] = total
            JOBS[job_id]["message"] = desc
            JOBS[job_id]["progress"] = int((step / total) * 100)

        out_vid, srt_p, json_p = process_video(
            input_video=input_path,
            output_dir=output_subdir,
            whisper_model_size=model,
            tts_engine="vieneu",
            voice=voice,
            style=style,
            burn_subtitles=burn_subs,
            progress_callback=progress_cb
        )

        if out_vid == "NO_AUDIO":
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["error"] = "Video không chứa luồng âm thanh (Audio stream)! Vui lòng kiểm tra video gốc."
            return

        if not out_vid or not Path(out_vid).exists():
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["error"] = "Không phát hiện giọng nói người trong video."
            return

        vid_file = Path(out_vid).name
        sub_name = Path(output_subdir).name
        srt_file = Path(srt_p).name if srt_p and Path(srt_p).exists() else None

        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["progress"] = 100
        JOBS[job_id]["message"] = "Lồng tiếng hoàn tất thành công!"
        JOBS[job_id]["result_video"] = f"/stream/{sub_name}/{vid_file}"
        JOBS[job_id]["download_video"] = f"/download/{sub_name}/{vid_file}"
        JOBS[job_id]["download_srt"] = f"/download/{sub_name}/{srt_file}" if srt_file else None

    except Exception as e:
        logger.exception("Lỗi khi xử lý video")
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)
    finally:
        # Xóa file upload tạm
        try:
            if Path(input_path).exists():
                Path(input_path).unlink()
        except Exception:
            pass


@app.get("/api/system")
def api_system():
    return get_system_stats()


@app.get("/api/files")
def api_files():
    return scan_processed_files()


@app.get("/api/jobs/{job_id}")
def api_job_status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    return JOBS[job_id]


@app.post("/api/upload")
async def api_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    voice: str = Form("Adam bựa"),
    style: str = Form("teen_bua"),
    burn_subtitles: bool = Form(True),
    whisper_model: str = Form("base")
):
    timestamp = int(time.time())
    safe_name = Path(file.filename).name.replace(" ", "_")
    input_file_path = DOWNLOADS_DIR / f"web_{timestamp}_{safe_name}"

    # Lưu file tải lên
    with open(input_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    job_id = f"job_{timestamp}"
    user_out_dir = OUTPUT_DIR / f"web_{timestamp}"
    user_out_dir.mkdir(parents=True, exist_ok=True)

    JOBS[job_id] = {
        "id": job_id,
        "filename": safe_name,
        "status": "queued",
        "step": 0,
        "total": 6,
        "progress": 0,
        "message": "Đang xếp hàng chờ xử lý...",
        "created_at": time.strftime("%H:%M:%S", time.localtime(timestamp))
    }

    background_tasks.add_task(
        run_dubbing_job,
        job_id=job_id,
        input_path=str(input_file_path),
        output_subdir=str(user_out_dir),
        voice=voice,
        style=style,
        burn_subs=burn_subtitles,
        model=whisper_model
    )

    return {"job_id": job_id, "message": "File đã được tải lên và bắt đầu xử lý!"}


@app.get("/download/{dir_name}/{filename}")
def download_file(dir_name: str, filename: str):
    file_path = OUTPUT_DIR / dir_name / filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File không tồn tại")
    return FileResponse(file_path, media_type="application/octet-stream", filename=filename)


@app.get("/stream/{dir_name}/{filename}")
def stream_file(dir_name: str, filename: str):
    file_path = OUTPUT_DIR / dir_name / filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Video không tồn tại")
    return FileResponse(file_path, media_type="video/mp4")


@app.delete("/api/files/{dir_name}")
def delete_processed_folder(dir_name: str):
    folder_path = OUTPUT_DIR / dir_name
    if folder_path.exists() and folder_path.is_dir():
        shutil.rmtree(folder_path, ignore_errors=True)
        return {"success": True, "message": f"Đã xóa {dir_name}"}
    raise HTTPException(status_code=404, detail="Thư mục không tồn tại")


@app.get("/", response_class=HTMLResponse)
def index_page():
    html_content = """<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Hub Lồng Tiếng AI - Adam Bựa (VieNeu-TTS)</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #090d16;
      --card-bg: rgba(22, 30, 49, 0.7);
      --card-border: rgba(255, 255, 255, 0.08);
      --primary: #6366f1;
      --primary-hover: #4f46e5;
      --primary-gradient: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
      --accent: #ec4899;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --success: #10b981;
      --warning: #f59e0b;
      --danger: #ef4444;
      --radius: 16px;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Plus Jakarta Sans', sans-serif;
      background-color: var(--bg);
      color: var(--text);
      min-height: 100vh;
      background-image: 
        radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.15) 0px, transparent 50%),
        radial-gradient(at 100% 100%, rgba(168, 85, 247, 0.15) 0px, transparent 50%);
      background-attachment: fixed;
      padding: 30px 20px;
    }
    .container { max-width: 1200px; margin: 0 auto; }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 30px;
      padding-bottom: 20px;
      border-bottom: 1px solid var(--card-border);
      flex-wrap: wrap;
      gap: 15px;
    }
    .logo { display: flex; align-items: center; gap: 12px; }
    .logo-icon {
      font-size: 32px;
      background: var(--primary-gradient);
      padding: 10px;
      border-radius: 14px;
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 8px 20px rgba(99, 102, 241, 0.3);
    }
    .logo-text h1 { font-size: 22px; font-weight: 800; letter-spacing: -0.5px; }
    .logo-text p { font-size: 13px; color: var(--text-muted); }
    .system-badges { display: flex; gap: 10px; flex-wrap: wrap; }
    .badge {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      padding: 8px 14px;
      border-radius: 100px;
      font-size: 12px;
      font-weight: 600;
      display: flex;
      align-items: center;
      gap: 6px;
      backdrop-filter: blur(10px);
    }
    .badge span { color: var(--primary); }
    
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 30px; }
    @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }

    .card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: var(--radius);
      padding: 24px;
      backdrop-filter: blur(16px);
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    }
    .card-title {
      font-size: 18px;
      font-weight: 700;
      margin-bottom: 18px;
      display: flex;
      align-items: center;
      gap: 10px;
    }

    /* Upload Zone */
    .drop-zone {
      border: 2px dashed rgba(99, 102, 241, 0.4);
      border-radius: 14px;
      padding: 36px 20px;
      text-align: center;
      cursor: pointer;
      transition: all 0.2s ease;
      background: rgba(99, 102, 241, 0.03);
      margin-bottom: 18px;
    }
    .drop-zone:hover, .drop-zone.dragover {
      border-color: var(--primary);
      background: rgba(99, 102, 241, 0.08);
      transform: translateY(-2px);
    }
    .drop-zone-icon { font-size: 40px; margin-bottom: 10px; }
    .drop-zone-text { font-size: 14px; color: var(--text); font-weight: 600; margin-bottom: 4px; }
    .drop-zone-sub { font-size: 12px; color: var(--text-muted); }
    #videoFile { display: none; }
    .selected-file-name {
      font-size: 13px;
      color: var(--success);
      font-weight: 600;
      margin-top: 10px;
      word-break: break-all;
    }

    .form-row { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 14px; }
    .form-group label {
      display: block;
      font-size: 12px;
      font-weight: 600;
      color: var(--text-muted);
      margin-bottom: 6px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    select, input[type="text"] {
      width: 100%;
      background: rgba(15, 23, 42, 0.8);
      border: 1px solid var(--card-border);
      border-radius: 10px;
      padding: 10px 14px;
      color: var(--text);
      font-family: inherit;
      font-size: 14px;
      outline: none;
      transition: border 0.2s;
    }
    select:focus, input:focus { border-color: var(--primary); }

    .btn {
      width: 100%;
      background: var(--primary-gradient);
      color: white;
      border: none;
      border-radius: 12px;
      padding: 14px;
      font-size: 15px;
      font-weight: 700;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
      box-shadow: 0 8px 24px rgba(99, 102, 241, 0.35);
      transition: all 0.2s;
    }
    .btn:hover { transform: translateY(-2px); box-shadow: 0 12px 28px rgba(99, 102, 241, 0.5); }
    .btn:disabled { opacity: 0.6; cursor: not-allowed; transform: none; }

    /* Progress Box */
    .progress-box {
      margin-top: 18px;
      padding: 16px;
      background: rgba(15, 23, 42, 0.7);
      border-radius: 12px;
      border: 1px solid var(--card-border);
      display: none;
    }
    .progress-bar-container {
      height: 8px;
      background: rgba(255, 255, 255, 0.1);
      border-radius: 100px;
      overflow: hidden;
      margin: 10px 0;
    }
    .progress-bar {
      height: 100%;
      width: 0%;
      background: var(--primary-gradient);
      border-radius: 100px;
      transition: width 0.3s;
    }
    .progress-status { display: flex; justify-content: space-between; font-size: 13px; font-weight: 600; }
    .progress-desc { font-size: 12px; color: var(--text-muted); margin-top: 4px; }

    /* File List & Player */
    .video-preview-player {
      width: 100%;
      border-radius: 12px;
      margin-bottom: 16px;
      background: #000;
      display: none;
      max-height: 320px;
    }
    .file-table-container { overflow-x: auto; max-height: 480px; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; text-align: left; }
    th {
      padding: 12px 14px;
      color: var(--text-muted);
      font-weight: 600;
      border-bottom: 1px solid var(--card-border);
      text-transform: uppercase;
      font-size: 11px;
    }
    td { padding: 14px; border-bottom: 1px solid rgba(255, 255, 255, 0.04); vertical-align: middle; }
    tr:hover td { background: rgba(255, 255, 255, 0.02); }
    .action-btns { display: flex; gap: 8px; }
    .act-btn {
      padding: 6px 10px;
      border-radius: 8px;
      font-size: 12px;
      font-weight: 600;
      text-decoration: none;
      display: inline-flex;
      align-items: center;
      gap: 4px;
      cursor: pointer;
      border: 1px solid transparent;
      transition: all 0.2s;
    }
    .btn-play { background: rgba(99, 102, 241, 0.15); color: #818cf8; border-color: rgba(99, 102, 241, 0.3); }
    .btn-play:hover { background: var(--primary); color: white; }
    .btn-down { background: rgba(16, 185, 129, 0.15); color: #34d399; border-color: rgba(16, 185, 129, 0.3); }
    .btn-down:hover { background: var(--success); color: white; }
    .btn-del { background: rgba(239, 68, 68, 0.15); color: #f87171; border-color: rgba(239, 68, 68, 0.3); }
    .btn-del:hover { background: var(--danger); color: white; }

    .empty-state { text-align: center; padding: 40px 20px; color: var(--text-muted); font-size: 14px; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="logo">
        <div class="logo-icon">🦜</div>
        <div class="logo-text">
          <h1>Hub Lồng Tiếng AI - Adam Bựa</h1>
          <p>VieNeu-TTS 48kHz & Faster-Whisper & Gemini 2.5 Flash</p>
        </div>
      </div>
      <div class="system-badges" id="sysBadges">
        <div class="badge">CPU: <span id="cpuVal">--%</span></div>
        <div class="badge">RAM: <span id="ramVal">-- GB</span></div>
        <div class="badge">Ổ cứng trống: <span id="diskVal">-- GB</span></div>
      </div>
    </header>

    <div class="grid">
      <!-- Upload Card -->
      <div class="card">
        <div class="card-title">📤 Tải Lên & Lồng Tiếng Video Mới</div>
        <form id="uploadForm">
          <div class="drop-zone" id="dropZone">
            <div class="drop-zone-icon">🎬</div>
            <div class="drop-zone-text">Kéo thả video vào đây hoặc bấm để chọn file</div>
            <div class="drop-zone-sub">Hỗ trợ MP4, MOV, MKV, WebM (Không giới hạn dung lượng 20MB)</div>
            <div class="selected-file-name" id="fileNameDisplay"></div>
            <input type="file" id="videoFile" name="file" accept="video/*" required>
          </div>

          <div class="form-row">
            <div class="form-group">
              <label>Giọng Lồng Tiếng</label>
              <select name="voice" id="voiceSelect">
                <option value="Adam bựa" selected>🦜 Adam bựa (VieNeu-TTS Hot)</option>
                <option value="Adam">🎙️ Adam chuẩn</option>
                <option value="Hải Đăng">🎙️ Hải Đăng (Nam)</option>
                <option value="Mai Anh">🎙️ Mai Anh (Nữ)</option>
              </select>
            </div>
            <div class="form-group">
              <label>Phong Cách Dịch</label>
              <select name="style" id="styleSelect">
                <option value="teen_bua" selected>🎭 Teen Bựa (Xưng anh - Con vợ)</option>
                <option value="normal">👔 Tiếng Việt chuẩn</option>
              </select>
            </div>
          </div>

          <div class="form-row">
            <div class="form-group">
              <label>Khắc Phụ Đề Cứng (Hardsub)</label>
              <select name="burn_subtitles" id="subSelect">
                <option value="true" selected>✅ Bật (Dán chữ lên video)</option>
                <option value="false">❌ Tắt (Chỉ lồng tiếng âm thanh)</option>
              </select>
            </div>
            <div class="form-group">
              <label>Model Nhận Diện AI</label>
              <select name="whisper_model" id="whisperSelect">
                <option value="base" selected>⚡ Base (Rất nhanh & Chuẩn)</option>
                <option value="small">🎯 Small (Chính xác hơn)</option>
                <option value="medium">💎 Medium (Chính xác cao nhất)</option>
              </select>
            </div>
          </div>

          <button type="submit" class="btn" id="submitBtn">
            <span>🚀 Bắt Đầu Lồng Tiếng Video</span>
          </button>
        </form>

        <div class="progress-box" id="progressBox">
          <div class="progress-status">
            <span id="progStep">Đang chuẩn bị...</span>
            <span id="progPercent">0%</span>
          </div>
          <div class="progress-bar-container">
            <div class="progress-bar" id="progressBar"></div>
          </div>
          <div class="progress-desc" id="progDesc">Hệ thống đang khởi động AI...</div>
        </div>
      </div>

      <!-- File Manager Card -->
      <div class="card">
        <div class="card-title">
          <span>📁 Kho Video Thành Phẩm & Phụ Đề</span>
          <button onclick="loadFiles()" style="background:transparent; border:none; color:var(--text-muted); cursor:pointer; font-size:16px; margin-left:auto;" title="Làm mới">🔄</button>
        </div>

        <video id="inlinePlayer" class="video-preview-player" controls autoplay></video>

        <div class="file-table-container">
          <table>
            <thead>
              <tr>
                <th>Tên File</th>
                <th>Dung Lượng</th>
                <th>Thời Gian</th>
                <th>Thao Tác</th>
              </tr>
            </thead>
            <tbody id="fileTableBody">
              <tr><td colspan="4" class="empty-state">Đang tải danh sách video...</td></tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  </div>

  <script>
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('videoFile');
    const fileNameDisplay = document.getElementById('fileNameDisplay');
    const uploadForm = document.getElementById('uploadForm');
    const submitBtn = document.getElementById('submitBtn');
    const progressBox = document.getElementById('progressBox');
    const progressBar = document.getElementById('progressBar');
    const progStep = document.getElementById('progStep');
    const progPercent = document.getElementById('progPercent');
    const progDesc = document.getElementById('progDesc');
    const inlinePlayer = document.getElementById('inlinePlayer');

    // Drag and Drop
    dropZone.addEventListener('click', () => fileInput.click());
    dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
    dropZone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropZone.classList.remove('dragover');
      if (e.dataTransfer.files.length) {
        fileInput.files = e.dataTransfer.files;
        updateSelectedFile();
      }
    });
    fileInput.addEventListener('change', updateSelectedFile);

    function updateSelectedFile() {
      if (fileInput.files.length) {
        const file = fileInput.files[0];
        const sizeMb = (file.size / (1024 * 1024)).toFixed(1);
        fileNameDisplay.textContent = `✅ Đã chọn: ${file.name} (${sizeMb} MB)`;
      }
    }

    // Submit Upload
    uploadForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      if (!fileInput.files.length) {
        alert('Vui lòng chọn 1 file video!');
        return;
      }

      const formData = new FormData(uploadForm);
      submitBtn.disabled = true;
      progressBox.style.display = 'block';
      progressBar.style.width = '5%';
      progPercent.textContent = '5%';
      progStep.textContent = 'Đang tải file lên VPS...';
      progDesc.textContent = 'Vui lòng giữ nguyên trang web trong giây lát...';

      try {
        const resp = await fetch('/api/upload', { method: 'POST', body: formData });
        const data = await resp.json();
        if (data.job_id) {
          trackJob(data.job_id);
        } else {
          alert('Lỗi tải lên: ' + JSON.stringify(data));
          resetUploadState();
        }
      } catch (err) {
        alert('Lỗi kết nối máy chủ: ' + err);
        resetUploadState();
      }
    });

    function trackJob(jobId) {
      const interval = setInterval(async () => {
        try {
          const res = await fetch(`/api/jobs/${jobId}`);
          const job = await res.json();

          if (job.status === 'processing' || job.status === 'queued') {
            const p = job.progress || 10;
            progressBar.style.width = p + '%';
            progPercent.textContent = p + '%';
            progStep.textContent = `[Bước ${job.step || 1}/${job.total || 6}] ${job.message || 'Đang xử lý'}`;
            progDesc.textContent = job.message || 'Hệ thống AI đang thực hiện...';
          } else if (job.status === 'completed') {
            clearInterval(interval);
            progressBar.style.width = '100%';
            progPercent.textContent = '100%';
            progStep.textContent = '🎉 Hoàn tất thành công!';
            progDesc.textContent = 'Video đã sẵn sàng!';

            // Tự động phát video
            if (job.result_video) {
              inlinePlayer.src = job.result_video;
              inlinePlayer.style.display = 'block';
              inlinePlayer.scrollIntoView({ behavior: 'smooth' });
            }
            resetUploadState();
            loadFiles();
          } else if (job.status === 'failed') {
            clearInterval(interval);
            progressBar.style.width = '100%';
            progressBar.style.background = 'var(--danger)';
            progStep.textContent = '❌ Thất bại';
            progDesc.textContent = job.error || 'Có lỗi xảy ra trong quá trình lồng tiếng.';
            resetUploadState();
          }
        } catch (e) {
          console.error(e);
        }
      }, 2000);
    }

    function resetUploadState() {
      submitBtn.disabled = false;
    }

    // Load File List
    async function loadFiles() {
      try {
        const res = await fetch('/api/files');
        const files = await res.json();
        const tbody = document.getElementById('fileTableBody');

        if (!files.length) {
          tbody.innerHTML = '<tr><td colspan="4" class="empty-state">Chưa có video thành phẩm nào. Hãy tải lên một video để bắt đầu!</td></tr>';
          return;
        }

        tbody.innerHTML = files.map(f => `
          <tr>
            <td style="font-weight:600; max-width:220px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;" title="${f.title}">
              🎬 ${f.title}
            </td>
            <td>${f.size_mb} MB</td>
            <td style="color:var(--text-muted);">${f.created_at}</td>
            <td>
              <div class="action-btns">
                <button class="act-btn btn-play" onclick="playVideo('${f.video_url}')">▶ Xem</button>
                <a class="act-btn btn-down" href="${f.download_url}" download>📥 Tải Video</a>
                ${f.srt_url ? `<a class="act-btn btn-down" href="${f.srt_url}" download style="background:rgba(59,130,246,0.15); color:#60a5fa; border-color:rgba(59,130,246,0.3)">📄 SRT</a>` : ''}
                <button class="act-btn btn-del" onclick="deleteFile('${f.id}')">🗑️</button>
              </div>
            </td>
          </tr>
        `).join('');
      } catch (err) {
        console.error(err);
      }
    }

    function playVideo(url) {
      inlinePlayer.src = url;
      inlinePlayer.style.display = 'block';
      inlinePlayer.scrollIntoView({ behavior: 'smooth' });
      inlinePlayer.play();
    }

    async function deleteFile(id) {
      if (confirm('Bạn có chắc muốn xóa video này khỏi máy chủ?')) {
        await fetch(`/api/files/${id}`, { method: 'DELETE' });
        loadFiles();
      }
    }

    // System Stats
    async function updateSystemStats() {
      try {
        const res = await fetch('/api/system');
        const data = await res.json();
        if (data.cpu_percent !== undefined) {
          document.getElementById('cpuVal').textContent = data.cpu_percent + '%';
          document.getElementById('ramVal').textContent = `${data.ram_used_gb}/${data.ram_total_gb} GB (${data.ram_percent}%)`;
          document.getElementById('diskVal').textContent = `${data.disk_free_gb} GB`;
        }
      } catch (e) {}
    }

    loadFiles();
    updateSystemStats();
    setInterval(updateSystemStats, 5000);
  </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    uvicorn.run("web_app:app", host="0.0.0.0", port=8080, reload=False)
