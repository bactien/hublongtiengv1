import os
import re
import sys
import json
import asyncio
import argparse
import subprocess
import urllib.request
import urllib.parse
from datetime import timedelta
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
import edge_tts
from dotenv import load_dotenv

# Tự động nạp file .env nếu có
load_dotenv()

# Đảm bảo in tiếng Việt trên console Windows không bị lỗi font
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")


def format_srt_time(seconds: float) -> str:
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    ms = int((td.total_seconds() - total_seconds) * 1000)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02}:{minutes:02}:{secs:02},{ms:03}"


VOICE_MAP = {
    "Adam bựa": "Adam bựa",
    "Nguyễn Ngọc Ngạn": "Nguyễn Ngọc Ngạn",
    "Ngọc Ngạn": "Nguyễn Ngọc Ngạn",
    "Thiện Tâm": "Thiền Tâm Đức",
    "Thiền Tâm Đức": "Thiền Tâm Đức",
    "Ngọc Huyền": "Ngọc Huyền",
    "Adam chuẩn": "Adam",
    "Adam": "Adam",
    "Trúc Ly": "Trúc Ly",
    "Thiện Minh": "Thiện Minh",
    "Mai Anh": "Mai Anh",
    "Hải Đăng": "Hải Đăng",
    "Thùy Dung": "Thùy Dung",
    "Quang Sơn": "Quang Sơn",
    "Ngọc Trân": "Ngọc Trân",
    "Minh Đức": "Minh Đức",
    "Phạm Tuyên": "Phạm Tuyên",
    "Thái Sơn": "Thái Sơn",
    "Xuân Vĩnh": "Xuân Vĩnh",
    "Thanh Bình": "Thanh Bình",
    "Ngọc Linh": "Ngọc Linh",
    "Đoan Trang": "Đoan Trang",
    "Thục Đoan": "Thục Đoan",
    "Minh Triết": "Minh Triết",
    "Mỹ Duyên": "Mỹ Duyên",
    "Quỳnh Anh": "Quỳnh Anh",
    "Đức Trí": "Đức Trí",
    "Kim Thanh": "Kim Thanh",
    "Quốc Tuấn": "Quốc Tuấn",
}


def translate_text(text: str, target_lang: str = "vi") -> str:
    """Dịch văn bản sang tiếng Việt qua Google Translate API."""
    if not text.strip():
        return ""
    try:
        url = "https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=" + target_lang + "&dt=t&q=" + urllib.parse.quote(text)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            translated = "".join([part[0] for part in data[0] if part[0]])
            return translated.strip()
    except Exception as e:
        print(f"  [Lỗi dịch]: {e}, giữ nguyên text gốc.")
        return text


def call_llm_translate(segments_list: list, style: str = "teen_bua") -> bool:
    """Sử dụng LLM (Gemini, Groq, DeepSeek, OpenAI) để biên kịch kịch bản tự nhiên, thông minh, không máy móc."""
    gemini_key = os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")

    if not any([gemini_key, openai_key, deepseek_key, groq_key]):
        return False

    # Chuẩn bị dữ liệu đầu vào rút gọn
    input_items = [{"id": i, "en": seg["text"]} for i, seg in enumerate(segments_list)]
    input_json_str = json.dumps(input_items, ensure_ascii=False)

    if style == "teen_bua":
        system_prompt = (
            "Bạn là một biên kịch và content creator lồng tiếng video đỉnh cao phong cách 'Adam bựa' (văn phong YouTuber/streamer dí dỏm, duyên dáng, gần gũi như Độ Mixi, BLV hài hước Việt Nam).\n"
            "Nhiệm vụ: Dịch và chuyển thể danh sách câu thoại tiếng Anh sau sang tiếng Việt.\n\n"
            "QUY TẮC BẮT BUỘC:\n"
            "1. Nhân xưng: Người nói luôn xưng 'anh' (hoặc 'anh đây', 'bọn anh'), gọi khán giả/đối phương là 'các con vợ' hoặc 'con vợ' (nếu nói với 1 người).\n"
            "2. Văn phong: Cực kỳ tự nhiên, mượt mà, thuần văn nói tiếng Việt hiện đại. Dịch thoát ý, không dịch máy móc theo kiểu Google Translate.\n"
            "3. KIỂM SOÁT TIẾNG CƯỜI (QUAN TRỌNG NHẤT): TUYỆT ĐỐI KHÔNG chèn 'ha ha ha' ở cuối mỗi câu! Cả video CHỈ ĐƯỢC chèn 'ha ha ha' hoặc '[cười]' tối đa 1 đến 2 lần tại câu mở đầu hoặc chỗ thật sự gây cười, châm biếm. Nếu câu bình thường thì kết thúc câu tự nhiên bằng dấu chấm hoặc dấu chấm than.\n"
            "4. Thêm biểu cảm nhẹ nhàng: Có thể dùng [cười], [thở dài], [hắng giọng] đúng ngữ cảnh (tối đa 1 lần/video).\n"
            "5. Độ dài câu: Giữ câu ngắn gọn, súc tích để tốc độ đọc vừa khớp thời lượng video.\n"
            "6. Định dạng đầu ra: Bắt buộc trả về đúng định dạng JSON: [{\"id\": 0, \"vi\": \"câu tiếng việt\"}, ...]"
        )
    else:
        system_prompt = (
            "Bạn là dịch giả phụ đề và thuyết minh video chuyên nghiệp.\n"
            "Dịch danh sách câu thoại tiếng Anh sang tiếng Việt tự nhiên, chuẩn ngữ pháp, mượt mà.\n"
            "Định dạng đầu ra: Bắt buộc trả về JSON: [{\"id\": 0, \"vi\": \"câu tiếng việt\"}, ...]"
        )

    # 1. Thử gọi Google Gemini (Miễn phí & dịch tiếng Việt tốt nhất)
    if gemini_key:
        try:
            print("  🤖 Đang biên kịch chất lượng cao bằng Google Gemini AI (gemini-2.5-flash)...")
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={gemini_key}"
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": f"{system_prompt}\n\nDữ liệu đầu vào:\n{input_json_str}"}
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.7,
                    "responseMimeType": "application/json"
                }
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                res_data = json.loads(resp.read().decode("utf-8"))
                text_content = res_data["candidates"][0]["content"]["parts"][0]["text"]
                parsed = json.loads(text_content)
                for item in parsed:
                    idx = item.get("id")
                    if idx is not None and idx < len(segments_list):
                        segments_list[idx]["vi_text"] = item.get("vi", "").strip()
                print("  ✨ Gemini AI đã biên kịch thành công kịch bản xuất sắc!")
                return True
        except Exception as e:
            print(f"  ⚠️ Lỗi gọi Gemini API: {e}, thử phương án tiếp theo...")

    # 2. Thử gọi OpenAI / DeepSeek / Groq (Chuẩn OpenAI format)
    endpoint = None
    headers = {"Content-Type": "application/json"}
    model_name = "gpt-4o-mini"
    chosen_key = None

    if groq_key:
        endpoint = "https://api.groq.com/openai/v1/chat/completions"
        model_name = "llama-3.3-70b-versatile"
        chosen_key = groq_key
    elif deepseek_key:
        endpoint = "https://api.deepseek.com/chat/completions"
        model_name = "deepseek-chat"
        chosen_key = deepseek_key
    elif openai_key:
        endpoint = "https://api.openai.com/v1/chat/completions"
        model_name = "gpt-4o-mini"
        chosen_key = openai_key

    if endpoint and chosen_key:
        try:
            print(f"  🤖 Đang biên kịch bằng {model_name}...")
            headers["Authorization"] = f"Bearer {chosen_key}"
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Hãy dịch danh sách sau:\n{input_json_str}"}
                ],
                "response_format": {"type": "json_object"} if "gpt" in model_name or "deepseek" in model_name else None,
                "temperature": 0.7
            }
            req = urllib.request.Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                res_data = json.loads(resp.read().decode("utf-8"))
                content = res_data["choices"][0]["message"]["content"]
                # Tìm JSON array trong content
                m = re.search(r"\[.*\]", content, re.DOTALL)
                if m:
                    parsed = json.loads(m.group(0))
                else:
                    parsed = json.loads(content)
                    if isinstance(parsed, dict) and "data" in parsed:
                        parsed = parsed["data"]

                for item in parsed:
                    idx = item.get("id")
                    if idx is not None and idx < len(segments_list):
                        segments_list[idx]["vi_text"] = item.get("vi", "").strip()
                print("  ✨ LLM đã biên kịch thành công kịch bản xuất sắc!")
                return True
        except Exception as e:
            print(f"  ⚠️ Lỗi gọi LLM ({model_name}): {e}")

    return False


def convert_to_teen_bua(vi_text: str, en_text: str = "", seg_index: int = 0, total_segs: int = 1) -> str:
    """Chuyển đổi câu dịch sang phong cách teen bựa thông minh, duyên dáng, KHÔNG spam ha ha ha."""
    t = vi_text.strip()
    if not t:
        return ""

    is_first = (seg_index == 0)
    is_last = (seg_index == total_segs - 1)

    # 1. Chào hỏi mở đầu: CHỈ áp dụng cho câu đầu tiên của video
    if is_first:
        chao_hoi_map = [
            (r"(?i)\bchào buổi sáng\b", "Hế lô các con vợ nha"),
            (r"(?i)\bchào buổi chiều\b", "Hế lô các con vợ nhá"),
            (r"(?i)\bchào buổi tối\b", "Gút nai các con vợ nhá"),
            (r"(?i)\bxin chào tất cả các bạn\b", "Chào cả lò nhà các con vợ nhá"),
            (r"(?i)\bxin chào các bạn\b", "Hế lô các con vợ ha ha"),
            (r"(?i)\bxin chào\b", "Hế lô các con vợ ha ha"),
            (r"(?i)\bchào bạn\b", "Hế lô con vợ"),
            (r"(?i)\brất vui được gặp\b", "Rất vui được gặp các con vợ nhá"),
        ]
        for pattern, rep in chao_hoi_map:
            t = re.sub(pattern, rep, t)

    # 2. Tạm biệt kết thúc: CHỈ áp dụng cho câu cuối cùng của video
    if is_last:
        tam_biet_map = [
            (r"(?i)\bcảm ơn các bạn đã theo dõi\b", "Đa tạ các con vợ đã theo dõi nhá"),
            (r"(?i)\bcảm ơn mọi người đã theo dõi\b", "Đa tạ các con vợ đã theo dõi nhá"),
            (r"(?i)\bcảm ơn mọi người\b", "Đa tạ các con vợ đã xem nhá"),
            (r"(?i)\bcảm ơn đã xem\b", "Đa tạ các con vợ đã theo dõi nhá"),
            (r"(?i)\bcảm ơn các bạn\b", "Đa tạ các con vợ nhá"),
            (r"(?i)\btạm biệt và hẹn gặp lại\b", "Anh lượn đây các con vợ ha ha ha, bái bai"),
            (r"(?i)\btạm biệt\b", "Anh lượn đây các con vợ ha ha ha"),
            (r"(?i)\bhẹn gặp lại\b", "Hẹn gặp lại các con vợ sau nhá bái bai"),
        ]
        for pattern, rep in tam_biet_map:
            t = re.sub(pattern, rep, t)

    # 3. Đại từ xưng hô tự nhiên (Cụm từ dài thay trước)
    xung_ho_map = [
        (r"(?i)\bcảm ơn tất cả các bạn\b", "đa tạ cả nhà các con vợ nhá"),
        (r"(?i)\bcảm ơn các bạn\b", "đa tạ các con vợ nhá"),
        (r"(?i)\bcảm ơn bạn\b", "đa tạ con vợ nhá"),
        (r"(?i)\btên tôi là\b", "anh là"),
        (r"(?i)\btôi tên là\b", "anh là"),
        (r"(?i)\btôi là\b", "anh là"),
        (r"(?i)\btôi đến từ\b", "quê anh ở"),
        (r"(?i)\bchúng tôi\b", "bọn anh"),
        (r"(?i)\bchúng ta\b", "anh em mình"),
        (r"(?i)\bcác bạn\b", "các con vợ"),
        (r"(?i)\bbạn\b", "con vợ"),
        (r"(?i)\btôi\b", "anh"),
        (r"(?i)\bmọi người\b", "các con vợ"),
    ]
    for pattern, rep in xung_ho_map:
        t = re.sub(pattern, rep, t)

    # 4. Tiếng lóng hài hước, chọn lọc tự nhiên (Không thay bừa bãi làm vỡ nghĩa)
    slang_map = [
        (r"(?i)\byêu thích\b", "mê tít"),
        (r"(?i)\btôi yêu\b", "anh mê vãi chưởng"),
        (r"(?i)\btuyệt vời\b", "hết nước chấm"),
        (r"(?i)\brất tốt\b", "chuẩn cmn bài"),
        (r"(?i)\bquá tốt\b", "chuẩn cmn bài"),
        (r"(?i)\bđược rồi\b", "uki luôn"),
        (r"(?i)\bvâng\b", "chuẩn luôn"),
        (r"(?i)\bcảm ơn\b", "đa tạ con vợ nhá"),
        (r"(?i)\bkinh ngạc\b", "ảo ma canada"),
        (r"(?i)\blàm việc chăm chỉ\b", "cày bục mặt"),
        (r"(?i)\bkhó khăn\b", "khoai vãi"),
        (r"(?i)\bkhông sao\b", "có gì đâu mà xoắn"),
    ]
    for pattern, rep in slang_map:
        t = re.sub(pattern, rep, t)

    # TUYỆT ĐỐI KHÔNG tự tiện chèn "ha ha ha" ở đuôi mọi câu!
    # Chỉ giữ dấu chấm/than gọn gàng
    t = t.strip()
    if t:
        t = t[0].upper() + t[1:]
    return t


async def generate_edge_tts(text: str, voice: str, output_path: str):
    """Tạo file âm thanh giọng đọc bằng Edge TTS."""
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)


def change_audio_speed(input_path: str, output_path: str, speed_factor: float):
    """Điều chỉnh tốc độ âm thanh bằng ffmpeg filter atempo."""
    filters = []
    current_speed = speed_factor
    while current_speed > 2.0:
        filters.append("atempo=2.0")
        current_speed /= 2.0
    while current_speed < 0.5:
        filters.append("atempo=0.5")
        current_speed /= 0.5
    filters.append(f"atempo={current_speed:.4f}")
    filter_str = ",".join(filters)

    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-filter:a", filter_str,
        "-vn", output_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


def check_video_has_audio(video_path: str) -> bool:
    """Kiểm tra xem file video có luồng âm thanh hay không bằng ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=codec_type",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return "audio" in res.stdout.lower()
    except Exception:
        return True


def process_video(
    input_video: str,
    output_dir: str = "output",
    whisper_model_size: str = "base",
    tts_engine: str = "vieneu",
    voice: str = "Adam bựa",
    style: str = "teen_bua",
    original_volume: float = 0.15,
    burn_subtitles: bool = True,
    progress_callback = None
):
    os.makedirs(output_dir, exist_ok=True)
    temp_dir = os.path.join(output_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(input_video))[0]
    safe_voice_name = voice.replace(" ", "_")
    out_video_path = os.path.join(output_dir, f"{base_name}_dubbed_{safe_voice_name}.mp4")
    srt_path = os.path.join(output_dir, f"{base_name}_vi.srt")
    metadata_json_path = os.path.join(output_dir, f"{base_name}_metadata.json")

    print(f"\n=======================================================")
    print(f"🎬 BẮT ĐẦU QUY TRÌNH LỒNG TIẾNG & SUBTITLE TIẾNG VIỆT")
    print(f"📹 Video đầu vào:   {input_video}")
    print(f"🎙️ Công nghệ TTS:   {tts_engine.upper()}")
    print(f"🗣️ Giọng đọc:       {voice}")
    print(f"🎭 Phong cách dịch: {style.upper()} (Chào các con vợ...)")
    print(f"🤖 Model nhận diện: {whisper_model_size}")
    print(f"=======================================================\n")

    # BƯỚC 1: Trích xuất âm thanh từ video
    print("▶ [Bước 1/6] Trích xuất âm thanh từ video gốc...")
    if progress_callback:
        progress_callback(1, 6, "Trích xuất âm thanh từ video gốc...")

    # Kiểm tra xem video có tiếng hay không
    if not check_video_has_audio(input_video):
        print("⚠️ Cảnh báo: Video không chứa luồng âm thanh (Audio stream)!")
        return "NO_AUDIO", None, None

    orig_wav = os.path.join(temp_dir, "original_audio.wav")
    cmd_extract = [
        "ffmpeg", "-y", "-i", input_video,
        "-vn", "-acodec", "pcm_s16le", "-ar", "24000", "-ac", "2",
        orig_wav
    ]
    res_extract = subprocess.run(cmd_extract, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res_extract.returncode != 0:
        err_msg = res_extract.stderr.lower()
        if "does not contain any stream" in err_msg or "matches no streams" in err_msg:
            print("⚠️ Cảnh báo: Video không có âm thanh!")
            return "NO_AUDIO", None, None
        raise RuntimeError(f"Lỗi FFmpeg khi trích xuất âm thanh: {res_extract.stderr.strip()[:200]}")

    orig_data, sample_rate = sf.read(orig_wav)
    total_samples = len(orig_data)

    # BƯỚC 2: Nhận diện giọng nói hoặc tải metadata đã chỉnh sửa
    segments_list = []
    if os.path.exists(metadata_json_path):
        print(f"▶ [Bước 2/6] Tìm thấy file lời thoại có sẵn: {metadata_json_path}")
        with open(metadata_json_path, "r", encoding="utf-8") as f:
            segments_list = json.load(f)
        print(f"  Đã nạp {len(segments_list)} câu từ file cấu hình!")
    else:
        print(f"▶ [Bước 2/6] Nhận diện lời thoại (Faster-Whisper {whisper_model_size})...")
        if progress_callback:
            progress_callback(2, 6, "Nhận diện lời thoại bằng AI...")
        model = WhisperModel(whisper_model_size, device="cpu", compute_type="int8")
        segments, info = model.transcribe(orig_wav, beam_size=5)

        for seg in segments:
            text = seg.text.strip()
            if text:
                segments_list.append({
                    "start": seg.start,
                    "end": seg.end,
                    "text": text,
                    "vi_text": ""
                })
                print(f"  [{seg.start:.2f}s -> {seg.end:.2f}s]: {text}")

    if not segments_list:
        print("⚠️ Không phát hiện giọng nói trong video!")
        return None, None, None

    # BƯỚC 3: Dịch lời thoại sang phong cách Teen Bựa & Lưu Metadata + File Subtitle (.srt)
    print(f"\n▶ [Bước 3/6] Dịch sang tiếng Việt (Phong cách: {style}) & tạo file phụ đề SRT...")
    if progress_callback:
        progress_callback(3, 6, f"Biên kịch lời thoại sang tiếng Việt ({style})...")

    # Kiểm tra các câu chưa có bản dịch
    needs_translation = any(not seg.get("vi_text") for seg in segments_list)
    if needs_translation:
        # 1. Thử gọi LLM AI thông minh (Gemini, Groq, DeepSeek, OpenAI) để biên kịch chất lượng cao
        llm_success = call_llm_translate(segments_list, style=style)
        if not llm_success:
            print("  ℹ️ Sử dụng bộ dịch tự động có chọn lọc từ vựng & kiểm soát tiếng cười...")
            total_segs = len(segments_list)
            for idx, seg in enumerate(segments_list):
                if not seg.get("vi_text"):
                    normal_vi = translate_text(seg["text"], target_lang="vi")
                    if style == "teen_bua":
                        seg["vi_text"] = convert_to_teen_bua(normal_vi, seg["text"], seg_index=idx, total_segs=total_segs)
                    else:
                        seg["vi_text"] = normal_vi

    srt_lines = []
    for i, seg in enumerate(segments_list, 1):
        final_vi = seg.get("vi_text", "")
        print(f"  {i}. {seg['text']}")
        print(f"     => {final_vi}")

        srt_lines.append(f"{i}")
        srt_lines.append(f"{format_srt_time(seg['start'])} --> {format_srt_time(seg['end'])}")
        srt_lines.append(f"{final_vi}\n")

    # Lưu metadata json để người dùng có thể tự chỉnh sửa câu bựa theo ý muốn
    with open(metadata_json_path, "w", encoding="utf-8") as f:
        json.dump(segments_list, f, ensure_ascii=False, indent=2)
    print(f"  📝 Đã lưu cấu hình lời thoại (bạn có thể mở file sửa câu): {metadata_json_path}")

    # Lưu file SRT
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_lines))
    print(f"  💾 Đã lưu phụ đề: {srt_path}")

    # BƯỚC 4: Khởi tạo giọng đọc VieNeu-TTS (Adam bựa, Ngọc Ngạn, Thiện Tâm, Ngọc Huyền...) hoặc Edge TTS
    if progress_callback:
        progress_callback(4, 6, f"Lồng tiếng bằng giọng đọc {voice}...")
    vieneu_model = None
    actual_voice = VOICE_MAP.get(voice, voice)
    if tts_engine == "vieneu":
        print(f"\n▶ [Bước 4/6] Khởi tạo mô hình VieNeu-TTS (Giọng: {voice} -> {actual_voice})...")
        from vieneu import Vieneu
        vieneu_model = Vieneu()
        ref_path = Path(__file__).parent / "samples" / "voices" / "nguyen_ngoc_ngan.wav"
        if "Nguyễn Ngọc Ngạn" not in vieneu_model._preset_voices and ref_path.exists():
            try:
                vieneu_model.add_voice(
                    "Nguyễn Ngọc Ngạn",
                    str(ref_path),
                    description="MC Nguyễn Ngọc Ngạn - Giọng kể chuyện Paris By Night",
                    gender="Nam",
                    save=True
                )
            except Exception as e:
                print(f"  ⚠️ Không thể nạp giọng Nguyễn Ngọc Ngạn: {e}")
        if actual_voice not in vieneu_model._preset_voices and "ngan" in actual_voice.lower():
            actual_voice = "Thanh Bình"
    else:
        print(f"\n▶ [Bước 4/6] Khởi tạo Microsoft Edge TTS ({voice})...")

    dubbed_track = np.zeros_like(orig_data)

    for i, seg in enumerate(segments_list, 1):
        raw_tts_path = os.path.join(temp_dir, f"seg_{i}_raw.wav")
        target_wav_path = os.path.join(temp_dir, f"seg_{i}.wav")

        # Sinh âm thanh tiếng Việt
        if tts_engine == "vieneu":
            audio_out = vieneu_model.infer(seg["vi_text"], voice=actual_voice)
            vieneu_model.save(audio_out, raw_tts_path)
        else:
            asyncio.run(generate_edge_tts(seg["vi_text"], voice, raw_tts_path))

        # Đồng bộ về chuẩn 24kHz stereo
        cmd_convert = [
            "ffmpeg", "-y", "-i", raw_tts_path,
            "-ar", str(sample_rate), "-ac", "2", target_wav_path
        ]
        subprocess.run(cmd_convert, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        seg_data, _ = sf.read(target_wav_path)
        seg_duration = len(seg_data) / sample_rate
        available_duration = seg["end"] - seg["start"]

        # Nếu nói dài hơn khung hình, tăng tốc nhẹ bằng atempo
        if seg_duration > available_duration and available_duration > 0.5:
            speed_ratio = seg_duration / available_duration
            if speed_ratio > 1.05:
                speed_ratio = min(speed_ratio, 1.45)
                adjusted_wav = os.path.join(temp_dir, f"seg_{i}_adjusted.wav")
                change_audio_speed(target_wav_path, adjusted_wav, speed_ratio)
                seg_data, _ = sf.read(adjusted_wav)

        start_sample = int(seg["start"] * sample_rate)
        end_sample = min(start_sample + len(seg_data), total_samples)
        copied_len = end_sample - start_sample

        if copied_len > 0:
            dubbed_track[start_sample:end_sample] += seg_data[:copied_len]

    # BƯỚC 5: Trộn âm thanh
    print("\n▶ [Bước 5/6] Hòa trộn âm thanh lồng tiếng với âm thanh nền...")
    if progress_callback:
        progress_callback(5, 6, "Hòa trộn âm thanh lồng tiếng và nhạc nền...")
    mixed_audio = (orig_data * original_volume) + dubbed_track
    max_val = np.max(np.abs(mixed_audio))
    if max_val > 0.95:
        mixed_audio = mixed_audio / max_val * 0.95

    final_wav_path = os.path.join(temp_dir, "final_dubbed_audio.wav")
    sf.write(final_wav_path, mixed_audio, sample_rate)

    # BƯỚC 6: Ghép vào video và xuất bản
    print("\n▶ [Bước 6/6] Ghép video thành phẩm với âm thanh mới & phụ đề...")
    if progress_callback:
        progress_callback(6, 6, "Ghép video thành phẩm và xuất bản...")
    if burn_subtitles:
        escaped_srt = srt_path.replace("\\", "/").replace(":", "\\:")
        cmd_mux = [
            "ffmpeg", "-y",
            "-i", input_video,
            "-i", final_wav_path,
            "-c:v", "libx264",
            "-vf", f"subtitles='{escaped_srt}'",
            "-c:a", "aac", "-b:a", "192k",
            "-map", "0:v:0", "-map", "1:a:0",
            "-shortest",
            out_video_path
        ]
    else:
        cmd_mux = [
            "ffmpeg", "-y",
            "-i", input_video,
            "-i", final_wav_path,
            "-i", srt_path,
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-c:s", "mov_text",
            "-map", "0:v:0", "-map", "1:a:0", "-map", "2:s:0",
            "-metadata:s:s:0", "language=vie",
            "-shortest",
            out_video_path
        ]

    res_mux = subprocess.run(cmd_mux, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res_mux.returncode != 0:
        print(f"⚠️ Thử ghép video không dán subtitle do lỗi: {res_mux.stderr.strip()[:100]}")
        cmd_fallback = [
            "ffmpeg", "-y",
            "-i", input_video,
            "-i", final_wav_path,
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-map", "0:v:0", "-map", "1:a:0",
            "-shortest",
            out_video_path
        ]
        subprocess.run(cmd_fallback, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    print("\n=======================================================")
    print("🎉 HOÀN TẤT THÀNH CÔNG!")
    print(f"🎬 Video lồng tiếng: {out_video_path}")
    print(f"📄 File phụ đề:      {srt_path}")
    print(f"📝 File kịch bản:    {metadata_json_path}")
    print("=======================================================\n")
    return out_video_path, srt_path, metadata_json_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tự động lồng tiếng phong cách Teen Bựa & tạo phụ đề tiếng Việt cho video")
    parser.add_argument("--input_file", default="e2e-tests/englishvideo.mp4", help="Đường dẫn file video đầu vào")
    parser.add_argument("--output_dir", default="output", help="Thư mục lưu video thành phẩm")
    parser.add_argument("--whisper_model", default="base", choices=["tiny", "base", "small", "medium", "large-v3"], help="Kích thước model Whisper")
    parser.add_argument("--tts_engine", default="vieneu", choices=["vieneu", "edge"], help="Engine giọng đọc")
    parser.add_argument("--voice", default="Adam bựa", help="Tên giọng đọc (Mặc định: 'Adam bựa')")
    parser.add_argument("--style", default="teen_bua", choices=["teen_bua", "normal"], help="Phong cách dịch (teen_bua hoặc normal)")
    parser.add_argument("--original_volume", type=float, default=0.15, help="Âm lượng video gốc khi lồng tiếng (0.15 = 15%)")
    parser.add_argument("--burn_subtitles", action="store_true", default=True, help="In phụ đề trực tiếp lên hình ảnh (Hardsub)")

    args = parser.parse_args()
    process_video(
        input_video=args.input_file,
        output_dir=args.output_dir,
        whisper_model_size=args.whisper_model,
        tts_engine=args.tts_engine,
        voice=args.voice,
        style=args.style,
        original_volume=args.original_volume,
        burn_subtitles=args.burn_subtitles
    )
