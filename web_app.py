import os
import sys
import time
import shutil
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
import uvicorn

try:
    import psutil
except ImportError:
    psutil = None

# Nạp cấu hình từ .env
from dotenv import load_dotenv
load_dotenv()

from auto_dub import process_video, VOICE_MAP

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("WebApp")

app = FastAPI(title="Hub Lồng Tiếng & Text To Speech AI")

BASE_DIR = Path(__file__).resolve().parent
DOWNLOADS_DIR = BASE_DIR / "downloads"
OUTPUT_DIR = BASE_DIR / "output"
TTS_DIR = OUTPUT_DIR / "tts"

DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TTS_DIR.mkdir(parents=True, exist_ok=True)

# Quản lý hàng đợi và trạng thái các tác vụ
JOBS: Dict[str, Dict[str, Any]] = {}
JOB_QUEUE = asyncio.Queue()
QUEUE_WORKER_TASK = None

# Khởi tạo mô hình TTS dùng chung để tránh load lại nhiều lần
VIENEU_MODEL = None


def get_vieneu_model():
    global VIENEU_MODEL
    if VIENEU_MODEL is None:
        from vieneu import Vieneu
        VIENEU_MODEL = Vieneu()
        ref_path = Path(__file__).parent / "samples" / "voices" / "nguyen_ngoc_ngan.wav"
        if "Nguyễn Ngọc Ngạn" not in VIENEU_MODEL._preset_voices and ref_path.exists():
            try:
                VIENEU_MODEL.add_voice(
                    "Nguyễn Ngọc Ngạn",
                    str(ref_path),
                    description="MC Nguyễn Ngọc Ngạn - Giọng kể chuyện Paris By Night",
                    gender="Nam",
                    save=True
                )
            except Exception as e:
                print(f"Warning: could not register Nguyễn Ngọc Ngạn: {e}")
    return VIENEU_MODEL


def get_system_stats():
    """Lấy thông số CPU, RAM và dung lượng ổ cứng."""
    if not psutil:
        return {}
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
    """Quét toàn bộ video thành phẩm trong thư mục output."""
    items = []
    if not OUTPUT_DIR.exists():
        return items

    for sub in sorted(OUTPUT_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if sub.is_dir() and sub.name not in ["temp", "tts"]:
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


def scan_tts_files():
    """Quét các file âm thanh Text To Speech đã tạo."""
    items = []
    if not TTS_DIR.exists():
        return items

    for audio_file in sorted(TTS_DIR.glob("*.wav"), key=lambda x: x.stat().st_mtime, reverse=True):
        stat = audio_file.stat()
        items.append({
            "filename": audio_file.name,
            "stream_url": f"/stream-tts/{audio_file.name}",
            "download_url": f"/download-tts/{audio_file.name}",
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "created_at": time.strftime("%H:%M %d/%m/%Y", time.localtime(stat.st_mtime)),
            "timestamp": stat.st_mtime,
        })
    return items


def process_dubbing_job_sync(job_id: str, job_info: dict):
    """Xử lý lồng tiếng đồng bộ cho một video trong hàng đợi."""
    try:
        JOBS[job_id]["status"] = "processing"
        JOBS[job_id]["step"] = 1
        JOBS[job_id]["message"] = "Bắt đầu bóc tách âm thanh..."
        JOBS[job_id]["progress"] = 10

        def progress_cb(step, total, desc):
            JOBS[job_id]["step"] = step
            JOBS[job_id]["total"] = total
            JOBS[job_id]["message"] = desc
            JOBS[job_id]["progress"] = int((step / total) * 100)

        out_vid, srt_p, json_p = process_video(
            input_video=job_info["input_path"],
            output_dir=job_info["output_subdir"],
            whisper_model_size=job_info["model"],
            tts_engine="vieneu",
            voice=job_info["voice"],
            style=job_info["style"],
            burn_subtitles=job_info["burn_subs"],
            progress_callback=progress_cb
        )

        if out_vid == "NO_AUDIO":
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["error"] = "Video không chứa luồng âm thanh (Audio stream)!"
            return

        if not out_vid or not Path(out_vid).exists():
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["error"] = "Không phát hiện lời thoại người nói trong video."
            return

        vid_file = Path(out_vid).name
        sub_name = Path(job_info["output_subdir"]).name
        srt_file = Path(srt_p).name if srt_p and Path(srt_p).exists() else None

        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["progress"] = 100
        JOBS[job_id]["message"] = "Hoàn tất thành công!"
        JOBS[job_id]["result_video"] = f"/stream/{sub_name}/{vid_file}"
        JOBS[job_id]["download_video"] = f"/download/{sub_name}/{vid_file}"
        JOBS[job_id]["download_srt"] = f"/download/{sub_name}/{srt_file}" if srt_file else None

    except Exception as e:
        logger.exception("Lỗi khi xử lý video")
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)
    finally:
        try:
            p = Path(job_info["input_path"])
            if p.exists():
                p.unlink()
        except Exception:
            pass


async def queue_worker():
    """Hàng đợi ngầm xử lý lần lượt từng video một (Tuần tự FIFO)."""
    logger.info("Khởi động Hàng đợi xử lý tuần tự...")
    while True:
        job_id, job_info = await JOB_QUEUE.get()
        logger.info(f"Bắt đầu xử lý Job từ hàng đợi: {job_id} ({job_info['filename']})")
        await asyncio.to_thread(process_dubbing_job_sync, job_id, job_info)
        JOB_QUEUE.task_done()
        logger.info(f"Hoàn thành Job {job_id}, chuyển sang job tiếp theo...")


@app.on_event("startup")
async def startup_event():
    global QUEUE_WORKER_TASK
    QUEUE_WORKER_TASK = asyncio.create_task(queue_worker())


# ==========================================
# API ENDPOINTS: VIDEO DUBBING & BATCH UPLOAD
# ==========================================

@app.get("/api/system")
def api_system():
    return get_system_stats()


@app.get("/api/files")
def api_files():
    return scan_processed_files()


@app.get("/api/jobs")
def api_all_jobs():
    """Lấy danh sách trạng thái của tất cả các jobs trong hàng đợi."""
    return list(JOBS.values())


@app.get("/api/jobs/{job_id}")
def api_job_status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    return JOBS[job_id]


@app.post("/api/upload")
async def api_upload_batch(
    files: List[UploadFile] = File(...),
    voice: str = Form("Adam bựa"),
    style: str = Form("teen_bua"),
    burn_subtitles: bool = Form(True),
    whisper_model: str = Form("base")
):
    """Hỗ trợ upload 1 hoặc nhiều video cùng lúc, đưa vào hàng đợi xử lý lần lượt."""
    queued_jobs = []

    for idx, file in enumerate(files):
        timestamp = int(time.time())
        safe_name = Path(file.filename).name.replace(" ", "_")
        job_id = f"job_{timestamp}_{idx}"
        input_file_path = DOWNLOADS_DIR / f"batch_{timestamp}_{idx}_{safe_name}"

        # Lưu file tải lên
        with open(input_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        user_out_dir = OUTPUT_DIR / f"out_{timestamp}_{idx}"
        user_out_dir.mkdir(parents=True, exist_ok=True)

        job_info = {
            "id": job_id,
            "filename": safe_name,
            "input_path": str(input_file_path),
            "output_subdir": str(user_out_dir),
            "voice": voice,
            "style": style,
            "burn_subs": burn_subtitles,
            "model": whisper_model,
            "status": "queued",
            "step": 0,
            "total": 6,
            "progress": 0,
            "message": "Đang xếp hàng chờ xử lý lần lượt...",
            "created_at": time.strftime("%H:%M:%S", time.localtime(timestamp))
        }

        JOBS[job_id] = job_info
        await JOB_QUEUE.put((job_id, job_info))
        queued_jobs.append(job_id)

    return {
        "success": True,
        "count": len(queued_jobs),
        "job_ids": queued_jobs,
        "message": f"Đã thêm {len(queued_jobs)} video vào hàng đợi xử lý tuần tự!"
    }


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


# ==========================================
# API ENDPOINTS: TEXT TO SPEECH (TTS)
# ==========================================

@app.get("/api/tts/list")
def api_tts_list():
    return scan_tts_files()


@app.post("/api/tts")
async def api_create_tts(
    text: str = Form(...),
    voice: str = Form("Adam bựa")
):
    """Chuyển văn bản thành giọng nói bằng VieNeu-TTS."""
    clean_text = text.strip()
    if not clean_text:
        raise HTTPException(status_code=400, detail="Văn bản không được để trống!")

    timestamp = int(time.time())
    safe_voice = voice.replace(" ", "_")
    out_filename = f"tts_{timestamp}_{safe_voice}.wav"
    out_filepath = TTS_DIR / out_filename

    def _generate():
        model = get_vieneu_model()
        actual_voice = VOICE_MAP.get(voice, voice)
        if actual_voice not in model._preset_voices and "ngan" in actual_voice.lower():
            actual_voice = "Thanh Bình"
        audio = model.infer(clean_text, voice=actual_voice)
        model.save(audio, str(out_filepath))

    await asyncio.to_thread(_generate)

    return {
        "success": True,
        "filename": out_filename,
        "voice": voice,
        "stream_url": f"/stream-tts/{out_filename}",
        "download_url": f"/download-tts/{out_filename}",
        "message": f"Tạo giọng đọc {voice} thành công!"
    }


@app.get("/stream-tts/{filename}")
def stream_tts_file(filename: str):
    file_path = TTS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File âm thanh không tồn tại")
    return FileResponse(file_path, media_type="audio/wav")


@app.get("/download-tts/{filename}")
def download_tts_file(filename: str):
    file_path = TTS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File âm thanh không tồn tại")
    return FileResponse(file_path, media_type="application/octet-stream", filename=filename)


@app.delete("/api/tts/{filename}")
def delete_tts_file(filename: str):
    file_path = TTS_DIR / filename
    if file_path.exists():
        file_path.unlink()
        return {"success": True, "message": "Đã xóa file âm thanh"}
    raise HTTPException(status_code=404, detail="File không tồn tại")


# ==========================================
# FRONTEND HTML USER INTERFACE
# ==========================================

@app.get("/", response_class=HTMLResponse)
def index_page():
    html_content = """<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Hub Lồng Tiếng & Text-to-Speech AI</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #090d16;
      --card-bg: rgba(20, 27, 45, 0.75);
      --card-border: rgba(255, 255, 255, 0.08);
      --primary: #6366f1;
      --primary-hover: #4f46e5;
      --primary-gradient: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
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
      padding: 24px 20px;
    }
    .container { max-width: 1240px; margin: 0 auto; }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 24px;
      padding-bottom: 18px;
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

    /* Navigation Tabs */
    .tabs-nav {
      display: flex;
      gap: 12px;
      margin-bottom: 24px;
      background: rgba(15, 23, 42, 0.6);
      padding: 6px;
      border-radius: 14px;
      border: 1px solid var(--card-border);
      width: fit-content;
    }
    .tab-btn {
      background: transparent;
      color: var(--text-muted);
      border: none;
      padding: 10px 20px;
      border-radius: 10px;
      font-weight: 700;
      font-size: 14px;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 8px;
      transition: all 0.2s;
    }
    .tab-btn.active {
      background: var(--primary-gradient);
      color: white;
      box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
    }

    .tab-pane { display: none; }
    .tab-pane.active { display: block; }

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

    /* Drag & Drop */
    .drop-zone {
      border: 2px dashed rgba(99, 102, 241, 0.4);
      border-radius: 14px;
      padding: 32px 20px;
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
    .drop-zone-icon { font-size: 36px; margin-bottom: 8px; }
    .drop-zone-text { font-size: 14px; color: var(--text); font-weight: 600; margin-bottom: 4px; }
    .drop-zone-sub { font-size: 12px; color: var(--text-muted); }
    #videoFiles { display: none; }
    .selected-files-list {
      margin-top: 10px;
      max-height: 120px;
      overflow-y: auto;
      font-size: 12px;
      text-align: left;
    }
    .file-badge-item {
      background: rgba(16, 185, 129, 0.15);
      color: #34d399;
      padding: 4px 10px;
      border-radius: 6px;
      margin-bottom: 4px;
      display: flex;
      justify-content: space-between;
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
    select, input[type="text"], textarea {
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
    select:focus, input:focus, textarea:focus { border-color: var(--primary); }

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

    /* Queue Cards */
    .queue-monitor {
      margin-top: 18px;
      max-height: 280px;
      overflow-y: auto;
    }
    .job-card {
      background: rgba(15, 23, 42, 0.85);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 12px 14px;
      margin-bottom: 10px;
      font-size: 13px;
    }
    .job-card-header { display: flex; justify-content: space-between; font-weight: 600; margin-bottom: 6px; }
    .job-progress-bar { height: 6px; background: rgba(255, 255, 255, 0.1); border-radius: 100px; overflow: hidden; margin-top: 6px; }
    .job-progress-fill { height: 100%; background: var(--primary-gradient); transition: width 0.3s; }
    .status-queued { color: var(--warning); }
    .status-processing { color: var(--primary); }
    .status-completed { color: var(--success); }
    .status-failed { color: var(--danger); }

    /* Player & Table */
    .video-preview-player { width: 100%; border-radius: 12px; margin-bottom: 16px; background: #000; display: none; max-height: 300px; }
    .file-table-container { overflow-x: auto; max-height: 480px; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; text-align: left; }
    th { padding: 12px 14px; color: var(--text-muted); font-weight: 600; border-bottom: 1px solid var(--card-border); text-transform: uppercase; font-size: 11px; }
    td { padding: 12px 14px; border-bottom: 1px solid rgba(255, 255, 255, 0.04); vertical-align: middle; }
    tr:hover td { background: rgba(255, 255, 255, 0.02); }
    .action-btns { display: flex; gap: 6px; }
    .act-btn {
      padding: 5px 9px;
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
    .empty-state { text-align: center; padding: 36px 20px; color: var(--text-muted); font-size: 14px; }

    /* TTS Quick Tag Buttons */
    .quick-tags { display: flex; gap: 8px; margin-bottom: 10px; flex-wrap: wrap; }
    .tag-btn {
      background: rgba(255, 255, 255, 0.06);
      border: 1px solid var(--card-border);
      color: var(--text);
      font-size: 12px;
      padding: 4px 10px;
      border-radius: 8px;
      cursor: pointer;
      font-weight: 600;
    }
    .tag-btn:hover { background: rgba(99, 102, 241, 0.2); border-color: var(--primary); }
    .tts-player-box {
      margin-top: 18px;
      background: rgba(15, 23, 42, 0.9);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 16px;
      display: none;
    }
    audio { width: 100%; margin-top: 8px; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="logo">
        <div class="logo-icon">🦜</div>
        <div class="logo-text">
          <h1>Hub Lồng Tiếng & Text-To-Speech AI</h1>
          <p>VieNeu-TTS 48kHz & Faster-Whisper & Gemini 2.5 Flash</p>
        </div>
      </div>
      <div class="system-badges" id="sysBadges">
        <div class="badge">CPU: <span id="cpuVal">--%</span></div>
        <div class="badge">RAM: <span id="ramVal">-- GB</span></div>
        <div class="badge">Ổ cứng trống: <span id="diskVal">-- GB</span></div>
      </div>
    </header>

    <!-- Navigation Tabs -->
    <div class="tabs-nav">
      <button class="tab-btn active" onclick="switchTab('dubbingTab')">🎬 Lồng Tiếng Video (Batch Queue)</button>
      <button class="tab-btn" onclick="switchTab('ttsTab')">🎙️ Chuyển Văn Bản Thành Giọng Nói (TTS)</button>
    </div>

    <!-- TAB 1: VIDEO DUBBING (BATCH QUEUE) -->
    <div id="dubbingTab" class="tab-pane active">
      <div class="grid">
        <!-- Upload Card -->
        <div class="card">
          <div class="card-title">📤 Tải Lên Nhiều Video (Xử Lý Lần Lượt)</div>
          <form id="uploadForm">
            <div class="drop-zone" id="dropZone">
              <div class="drop-zone-icon">🎬</div>
              <div class="drop-zone-text">Kéo thả 1 hoặc NHIỀU video vào đây</div>
              <div class="drop-zone-sub">Hỗ trợ chọn hàng loạt file MP4, MOV, MKV (Không giới hạn dung lượng 20MB)</div>
              <input type="file" id="videoFiles" name="files" accept="video/*" multiple required>
              <div class="selected-files-list" id="filesListDisplay"></div>
            </div>

            <div class="form-row">
              <div class="form-group">
                <label>Giọng Lồng Tiếng</label>
                <select name="voice" id="voiceSelect">
                  <optgroup label="🌟 Giọng Nổi Tiếng">
                    <option value="Adam bựa" selected>🦜 Adam bựa (Hot Streamer)</option>
                    <option value="Nguyễn Ngọc Ngạn">🎙️ Nguyễn Ngọc Ngạn (Kể chuyện miền Nam)</option>
                    <option value="Thiện Tâm">🎙️ Thiện Tâm (Kể chuyện miền Bắc, đạo lý)</option>
                    <option value="Ngọc Huyền">🎙️ Ngọc Huyền (Nữ miền Bắc truyền cảm)</option>
                  </optgroup>
                  <optgroup label="📻 Các Giọng Khác">
                    <option value="Adam chuẩn">🎙️ Adam chuẩn (Nam Nam)</option>
                    <option value="Mai Anh">🎙️ Mai Anh (Nữ Bắc · Tin tức)</option>
                    <option value="Hải Đăng">🎙️ Hải Đăng (Nam Bắc · Tự nhiên)</option>
                    <option value="Thùy Dung">🎙️ Thùy Dung (Nữ Nam · Tin tức)</option>
                    <option value="Quang Sơn">🎙️ Quang Sơn (Nam Trung)</option>
                    <option value="Ngọc Trân">🎙️ Ngọc Trân (Nữ Trung)</option>
                    <option value="Mỹ Duyên">🎙️ Mỹ Duyên (Nữ Nam · Đọc truyện)</option>
                    <option value="Quỳnh Anh">🎙️ Quỳnh Anh (Nữ Bắc · Đọc truyện)</option>
                  </optgroup>
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
                <select name="burn_subtitles">
                  <option value="true" selected>✅ Bật (Dán chữ lên video)</option>
                  <option value="false">❌ Tắt (Chỉ lồng tiếng âm thanh)</option>
                </select>
              </div>
              <div class="form-group">
                <label>Model Nhận Diện AI</label>
                <select name="whisper_model">
                  <option value="base" selected>⚡ Base (Rất nhanh & Chuẩn)</option>
                  <option value="small">🎯 Small (Chính xác hơn)</option>
                  <option value="medium">💎 Medium (Chính xác cao nhất)</option>
                </select>
              </div>
            </div>

            <button type="submit" class="btn" id="submitBtn">
              <span>🚀 Bắt Đầu Đưa Vào Hàng Đợi Lồng Tiếng</span>
            </button>
          </form>

          <!-- Queue Monitor -->
          <div class="queue-monitor" id="queueMonitor"></div>
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

    <!-- TAB 2: TEXT TO SPEECH (TTS STUDIO) -->
    <div id="ttsTab" class="tab-pane">
      <div class="grid">
        <!-- TTS Input Card -->
        <div class="card">
          <div class="card-title">🎙️ Studio Chuyển Văn Bản Thành Giọng Nói (TTS)</div>
          <form id="ttsForm">
            <div class="form-group" style="margin-bottom:14px;">
              <label>Chọn Giọng Đọc</label>
              <select name="voice" id="ttsVoiceSelect">
                <optgroup label="🌟 Giọng Nổi Tiếng & Yêu Thích">
                  <option value="Nguyễn Ngọc Ngạn">🎙️ Nguyễn Ngọc Ngạn (Kể chuyện miền Nam rùng rợn, truyền cảm)</option>
                  <option value="Thiện Tâm">🎙️ Thiện Tâm (Kể chuyện miền Bắc, đạo lý, triết lý sống)</option>
                  <option value="Ngọc Huyền">🎙️ Ngọc Huyền (Nữ miền Bắc truyền cảm, tự nhiên)</option>
                  <option value="Adam bựa" selected>🦜 Adam bựa (Hot Streamer VieNeu-TTS)</option>
                </optgroup>
                <optgroup label="📻 Các Giọng Khác">
                  <option value="Adam chuẩn">🎙️ Adam chuẩn (Nam Nam · Tự nhiên)</option>
                  <option value="Mai Anh">🎙️ Mai Anh (Nữ Bắc · Tin tức)</option>
                  <option value="Hải Đăng">🎙️ Hải Đăng (Nam Bắc · Tự nhiên)</option>
                  <option value="Thùy Dung">🎙️ Thùy Dung (Nữ Nam · Tin tức)</option>
                  <option value="Quang Sơn">🎙️ Quang Sơn (Nam Trung)</option>
                  <option value="Ngọc Trân">🎙️ Ngọc Trân (Nữ Trung)</option>
                  <option value="Mỹ Duyên">🎙️ Mỹ Duyên (Nữ Nam · Đọc truyện)</option>
                  <option value="Quỳnh Anh">🎙️ Quỳnh Anh (Nữ Bắc · Đọc truyện)</option>
                  <option value="Đức Trí">🎙️ Đức Trí (Nam Nam · Đọc truyện)</option>
                  <option value="Kim Thanh">🎙️ Kim Thanh (Nữ Nam · Đọc truyện)</option>
                </optgroup>
              </select>
            </div>

            <div class="form-group" style="margin-bottom:10px;">
              <label>Chèn Nhanh Biểu Cảm VieNeu-TTS</label>
              <div class="quick-tags">
                <button type="button" class="tag-btn" onclick="insertTag('[cười]')">😄 [cười]</button>
                <button type="button" class="tag-btn" onclick="insertTag('[thở dài]')">😮‍💨 [thở dài]</button>
                <button type="button" class="tag-btn" onclick="insertTag('[hắng giọng]')">🗣️ [hắng giọng]</button>
                <button type="button" class="tag-btn" onclick="insertTag('ha ha ha!')">🤣 ha ha ha!</button>
                <button type="button" class="tag-btn" onclick="insertTag('các con vợ')">🔥 các con vợ</button>
              </div>
            </div>

            <div class="form-group" style="margin-bottom:16px;">
              <label>Nội Dung Văn Bản Tiếng Việt</label>
              <textarea id="ttsTextInput" name="text" rows="7" placeholder="Nhập hoặc dán đoạn văn bản, truyện ma, bài review hoặc lời thoại bạn muốn đọc vào đây..." required></textarea>
              <div style="font-size:12px; color:var(--text-muted); margin-top:4px; text-align:right;">
                <span id="charCount">0</span> ký tự
              </div>
            </div>

            <button type="submit" class="btn" id="ttsSubmitBtn">
              <span>🔊 Tạo Giọng Đọc Ngay (VieNeu-TTS)</span>
            </button>
          </form>

          <div class="tts-player-box" id="ttsPlayerBox">
            <div style="font-weight:700; margin-bottom:6px; color:var(--success);">✅ Đã tạo giọng đọc thành công!</div>
            <audio id="ttsAudioPlayer" controls></audio>
            <div style="margin-top:12px; display:flex; gap:10px;">
              <a id="ttsDownloadBtn" class="act-btn btn-down" style="padding:10px 16px; font-size:13px;" download>📥 Tải File Âm Thanh (.wav)</a>
            </div>
          </div>
        </div>

        <!-- TTS History Card -->
        <div class="card">
          <div class="card-title">
            <span>📻 Kho File Âm Thanh Đã Tạo</span>
            <button onclick="loadTtsFiles()" style="background:transparent; border:none; color:var(--text-muted); cursor:pointer; font-size:16px; margin-left:auto;" title="Làm mới">🔄</button>
          </div>

          <div class="file-table-container">
            <table>
              <thead>
                <tr>
                  <th>Tên File</th>
                  <th>Kích Thước</th>
                  <th>Thời Gian</th>
                  <th>Thao Tác</th>
                </tr>
              </thead>
              <tbody id="ttsTableBody">
                <tr><td colspan="4" class="empty-state">Đang tải danh sách âm thanh...</td></tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  </div>

  <script>
    function switchTab(tabId) {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
      event.currentTarget.classList.add('active');
      document.getElementById(tabId).classList.add('active');
      if (tabId === 'ttsTab') loadTtsFiles();
      if (tabId === 'dubbingTab') loadFiles();
    }

    // Drag and Drop (Multiple files)
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('videoFiles');
    const filesListDisplay = document.getElementById('filesListDisplay');
    const uploadForm = document.getElementById('uploadForm');
    const submitBtn = document.getElementById('submitBtn');
    const queueMonitor = document.getElementById('queueMonitor');
    const inlinePlayer = document.getElementById('inlinePlayer');

    dropZone.addEventListener('click', () => fileInput.click());
    dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
    dropZone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropZone.classList.remove('dragover');
      if (e.dataTransfer.files.length) {
        fileInput.files = e.dataTransfer.files;
        updateSelectedFiles();
      }
    });
    fileInput.addEventListener('change', updateSelectedFiles);

    function updateSelectedFiles() {
      if (fileInput.files.length) {
        let html = '';
        let totalMb = 0;
        for (let i = 0; i < fileInput.files.length; i++) {
          const f = fileInput.files[i];
          const mb = (f.size / (1024 * 1024)).toFixed(1);
          totalMb += parseFloat(mb);
          html += `<div class="file-badge-item"><span>🎬 ${f.name}</span><span>${mb} MB</span></div>`;
        }
        filesListDisplay.innerHTML = `<div style="margin-bottom:6px; font-weight:700; color:var(--text);">Đã chọn ${fileInput.files.length} video (Tổng: ${totalMb.toFixed(1)} MB):</div>${html}`;
      }
    }

    // Batch Upload Submission
    uploadForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      if (!fileInput.files.length) {
        alert('Vui lòng chọn ít nhất 1 video!');
        return;
      }

      const formData = new FormData(uploadForm);
      submitBtn.disabled = true;

      try {
        const resp = await fetch('/api/upload', { method: 'POST', body: formData });
        const data = await resp.json();
        if (data.success) {
          alert(`Đã thêm ${data.count} video vào hàng đợi! Hệ thống sẽ xử lý lần lượt.`);
          fileInput.value = '';
          filesListDisplay.innerHTML = '';
          pollJobs();
        } else {
          alert('Lỗi: ' + JSON.stringify(data));
        }
      } catch (err) {
        alert('Lỗi kết nối máy chủ: ' + err);
      } finally {
        submitBtn.disabled = false;
      }
    });

    // Queue Monitor Polling
    async function pollJobs() {
      try {
        const res = await fetch('/api/jobs');
        const jobs = await res.json();
        if (!jobs.length) {
          queueMonitor.innerHTML = '';
          return;
        }

        queueMonitor.innerHTML = jobs.slice(-6).reverse().map(j => {
          let statusClass = `status-${j.status}`;
          let statusText = j.status === 'queued' ? '⏳ Đang chờ lượt' :
                           j.status === 'processing' ? `🔄 Đang xử lý [${j.step}/${j.total}]` :
                           j.status === 'completed' ? '✅ Hoàn tất' : '❌ Lỗi';

          return `
            <div class="job-card">
              <div class="job-card-header">
                <span style="max-width:240px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">🎬 ${j.filename}</span>
                <span class="${statusClass}">${statusText}</span>
              </div>
              <div style="font-size:12px; color:var(--text-muted);">${j.message || ''}</div>
              <div class="job-progress-bar">
                <div class="job-progress-fill" style="width:${j.progress || 0}%;"></div>
              </div>
              ${j.result_video ? `
                <div style="margin-top:8px; display:flex; gap:6px;">
                  <button class="act-btn btn-play" onclick="playVideo('${j.result_video}')">▶ Xem ngay</button>
                  <a class="act-btn btn-down" href="${j.download_video}" download>📥 Tải Video</a>
                  ${j.download_srt ? `<a class="act-btn btn-down" href="${j.download_srt}" download style="background:rgba(59,130,246,0.15); color:#60a5fa;">📄 SRT</a>` : ''}
                </div>
              ` : ''}
            </div>
          `;
        }).join('');
      } catch (e) {
        console.error(e);
      }
    }
    setInterval(pollJobs, 3000);
    pollJobs();

    // Load Dubbed File List
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
                <a class="act-btn btn-down" href="${f.download_url}" download>📥 Tải</a>
                ${f.srt_url ? `<a class="act-btn btn-down" href="${f.srt_url}" download style="background:rgba(59,130,246,0.15); color:#60a5fa;">📄 SRT</a>` : ''}
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
      if (confirm('Bạn có chắc muốn xóa video này?')) {
        await fetch(`/api/files/${id}`, { method: 'DELETE' });
        loadFiles();
      }
    }

    // ==========================================
    // TEXT TO SPEECH (TTS) LOGIC
    // ==========================================
    const ttsForm = document.getElementById('ttsForm');
    const ttsTextInput = document.getElementById('ttsTextInput');
    const ttsSubmitBtn = document.getElementById('ttsSubmitBtn');
    const ttsPlayerBox = document.getElementById('ttsPlayerBox');
    const ttsAudioPlayer = document.getElementById('ttsAudioPlayer');
    const ttsDownloadBtn = document.getElementById('ttsDownloadBtn');
    const charCount = document.getElementById('charCount');

    ttsTextInput.addEventListener('input', () => {
      charCount.textContent = ttsTextInput.value.length;
    });

    function insertTag(tag) {
      const start = ttsTextInput.selectionStart;
      const end = ttsTextInput.selectionEnd;
      const text = ttsTextInput.value;
      ttsTextInput.value = text.substring(0, start) + ' ' + tag + ' ' + text.substring(end);
      ttsTextInput.focus();
      charCount.textContent = ttsTextInput.value.length;
    }

    ttsForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const text = ttsTextInput.value.trim();
      if (!text) return;

      const formData = new FormData(ttsForm);
      ttsSubmitBtn.disabled = true;
      ttsSubmitBtn.innerHTML = '<span>⏳ Đang sinh giọng đọc AI...</span>';

      try {
        const resp = await fetch('/api/tts', { method: 'POST', body: formData });
        const data = await resp.json();
        if (data.success) {
          ttsAudioPlayer.src = data.stream_url;
          ttsDownloadBtn.href = data.download_url;
          ttsPlayerBox.style.display = 'block';
          ttsAudioPlayer.play();
          loadTtsFiles();
        } else {
          alert('Lỗi tạo giọng đọc: ' + JSON.stringify(data));
        }
      } catch (err) {
        alert('Lỗi kết nối: ' + err);
      } finally {
        ttsSubmitBtn.disabled = false;
        ttsSubmitBtn.innerHTML = '<span>🔊 Tạo Giọng Đọc Ngay (VieNeu-TTS)</span>';
      }
    });

    async function loadTtsFiles() {
      try {
        const res = await fetch('/api/tts/list');
        const files = await res.json();
        const tbody = document.getElementById('ttsTableBody');

        if (!files.length) {
          tbody.innerHTML = '<tr><td colspan="4" class="empty-state">Chưa có file âm thanh nào được tạo.</td></tr>';
          return;
        }

        tbody.innerHTML = files.map(f => `
          <tr>
            <td style="font-weight:600; max-width:220px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;" title="${f.filename}">
              🎙️ ${f.filename}
            </td>
            <td>${f.size_mb} MB</td>
            <td style="color:var(--text-muted);">${f.created_at}</td>
            <td>
              <div class="action-btns">
                <button class="act-btn btn-play" onclick="playTtsAudio('${f.stream_url}')">▶ Nghe</button>
                <a class="act-btn btn-down" href="${f.download_url}" download>📥 Tải WAV</a>
                <button class="act-btn btn-del" onclick="deleteTtsFile('${f.filename}')">🗑️</button>
              </div>
            </td>
          </tr>
        `).join('');
      } catch (e) {
        console.error(e);
      }
    }

    function playTtsAudio(url) {
      ttsAudioPlayer.src = url;
      ttsDownloadBtn.href = url.replace('/stream-tts/', '/download-tts/');
      ttsPlayerBox.style.display = 'block';
      ttsPlayerBox.scrollIntoView({ behavior: 'smooth' });
      ttsAudioPlayer.play();
    }

    async function deleteTtsFile(filename) {
      if (confirm('Bạn có chắc muốn xóa file âm thanh này?')) {
        await fetch(`/api/tts/${filename}`, { method: 'DELETE' });
        loadTtsFiles();
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
    uvicorn.run("web_app:app", host="0.0.0.0", port=8000, reload=False)
