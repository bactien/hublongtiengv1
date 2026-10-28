import os
import sys
import time
import asyncio
import logging
import argparse
import subprocess
from pathlib import Path
from dotenv import load_dotenv

import numpy as np
from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    constants,
)
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

try:
    import psutil
except ImportError:
    psutil = None

# Đảm bảo in tiếng Việt trên console Windows không lỗi font
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

# Cấu hình logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("TelegramBot")

# Nạp cấu hình từ .env
load_dotenv()

# Khóa xử lý đồng thời (Chạy 1 video/TTS tại một thời điểm để tối ưu CPU/RAM)
PROCESSING_LOCK = asyncio.Lock()

# Thư mục lưu trữ tạm thời
DOWNLOAD_DIR = Path("downloads")
OUTPUT_DIR = Path("output")
TTS_DIR = OUTPUT_DIR / "tts"

DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TTS_DIR.mkdir(parents=True, exist_ok=True)

# Cấu hình mặc định cho người dùng
DEFAULT_SETTINGS = {
    "voice": "Adam bựa",
    "style": "teen_bua",
    "speed": 1.0,
    "burn_subtitles": True,
    "whisper_model": "base",
    "original_volume": 0.15,
}

# Giới hạn kích thước file Telegram Bot API tải trực tiếp (20 MB)
MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024

# Danh mục phân loại giọng đọc VieNeu-TTS
VOICE_CATEGORIES = {
    "hot": {
        "title": "🌟 Giọng Nổi Tiếng & Hot Nhất",
        "voices": [
            ("🎙️ Nguyễn Ngọc Ngạn", "Nguyễn Ngọc Ngạn"),
            ("🦜 Adam bựa (Hot)", "Adam bựa"),
            ("🧘 Thiền Tâm Đức", "Thiền Tâm Đức"),
            ("🌸 Ngọc Huyền", "Ngọc Huyền"),
        ]
    },
    "north": {
        "title": "🍁 Giọng Miền Bắc",
        "voices": [
            ("🎙️ Trúc Ly (Nữ)", "Trúc Ly"),
            ("🎙️ Mai Anh (Nữ)", "Mai Anh"),
            ("🎙️ Quỳnh Anh (Nữ)", "Quỳnh Anh"),
            ("🎙️ Đoan Trang (Nữ)", "Đoan Trang"),
            ("🎙️ Ngọc Linh (Nữ)", "Ngọc Linh"),
            ("🎙️ Thiện Minh (Nam)", "Thiện Minh"),
            ("🎙️ Thanh Bình (Nam)", "Thanh Bình"),
            ("🎙️ Hải Đăng (Nam)", "Hải Đăng"),
            ("🎙️ Xuân Vĩnh (Nam)", "Xuân Vĩnh"),
            ("🎙️ Phạm Tuyên (Nam)", "Phạm Tuyên"),
            ("🎙️ Quốc Tuấn (Nam)", "Quốc Tuấn"),
            ("🎙️ Minh Đức (Nam)", "Minh Đức"),
        ]
    },
    "south": {
        "title": "🌴 Giọng Miền Nam",
        "voices": [
            ("🎙️ Thái Sơn (Nam)", "Thái Sơn"),
            ("🎙️ Minh Triết (Nam)", "Minh Triết"),
            ("🎙️ Đức Trí (Nam)", "Đức Trí"),
            ("🎙️ Thùy Dung (Nữ)", "Thùy Dung"),
            ("🎙️ Mỹ Duyên (Nữ)", "Mỹ Duyên"),
            ("🎙️ Thục Đoan (Nữ)", "Thục Đoan"),
            ("🎙️ Kim Thanh (Nữ)", "Kim Thanh"),
            ("🎙️ Adam chuẩn", "Adam chuẩn"),
        ]
    },
    "central": {
        "title": "🌾 Giọng Miền Trung",
        "voices": [
            ("🎙️ Quang Sơn (Nam)", "Quang Sơn"),
            ("🎙️ Ngọc Trân (Nữ)", "Ngọc Trân"),
        ]
    }
}


def get_user_settings(context: ContextTypes.DEFAULT_TYPE) -> dict:
    """Lấy hoặc khởi tạo cấu hình riêng cho từng người dùng."""
    if "settings" not in context.user_data:
        context.user_data["settings"] = DEFAULT_SETTINGS.copy()
    return context.user_data["settings"]


def get_settings_keyboard(settings: dict) -> InlineKeyboardMarkup:
    """Tạo bàn phím nút bấm tương tác cho bảng cài đặt chính."""
    style_text = "🎭 Teen bựa (Xưng anh - Con vợ)" if settings["style"] == "teen_bua" else "👔 Tiếng Việt chuẩn"
    sub_text = "✅ Đang BẬT" if settings["burn_subtitles"] else "❌ Đang TẮT"
    speed_text = f"⚡ {settings.get('speed', 1.0)}x"
    vol_text = f"🔊 {int(settings.get('original_volume', 0.15) * 100)}%"

    keyboard = [
        [
            InlineKeyboardButton(f"🎭 Phong cách dịch: {style_text}", callback_data="toggle_style")
        ],
        [
            InlineKeyboardButton(f"🗣️ Giọng đọc: {settings['voice']}", callback_data="show_voice_cats")
        ],
        [
            InlineKeyboardButton(f"⚡ Tốc độ: {speed_text}", callback_data="show_speed_menu"),
            InlineKeyboardButton(f"🔊 Tiếng gốc: {vol_text}", callback_data="show_vol_menu"),
        ],
        [
            InlineKeyboardButton(f"📝 Phụ đề cứng: {sub_text}", callback_data="toggle_sub")
        ],
        [
            InlineKeyboardButton("🔄 Đặt lại mặc định", callback_data="reset_settings"),
            InlineKeyboardButton("❌ Đóng menu", callback_data="close_settings"),
        ]
    ]
    return InlineKeyboardMarkup(keyboard)


def get_voice_cats_keyboard() -> InlineKeyboardMarkup:
    """Menu chọn danh mục vùng miền giọng đọc."""
    keyboard = [
        [InlineKeyboardButton("🌟 Giọng Nổi Tiếng & Hot Nhất (Ngọc Ngạn, Adam...)", callback_data="vcat_hot")],
        [InlineKeyboardButton("🍁 Giọng Miền Bắc (Trúc Ly, Mai Anh, Thiện Minh...)", callback_data="vcat_north")],
        [InlineKeyboardButton("🌴 Giọng Miền Nam (Thái Sơn, Minh Triết, Đức Trí...)", callback_data="vcat_south")],
        [InlineKeyboardButton("🌾 Giọng Miền Trung (Quang Sơn, Ngọc Trân)", callback_data="vcat_central")],
        [InlineKeyboardButton("⬅️ Quay lại bảng cài đặt", callback_data="back_to_settings")],
    ]
    return InlineKeyboardMarkup(keyboard)


def get_voices_in_category_keyboard(cat_key: str) -> InlineKeyboardMarkup:
    """Tạo danh sách các giọng trong 1 danh mục (xếp 2 cột)."""
    cat_data = VOICE_CATEGORIES.get(cat_key, {})
    voices = cat_data.get("voices", [])

    keyboard = []
    row = []
    for text, voice_name in voices:
        row.append(InlineKeyboardButton(text, callback_data=f"voice_{voice_name}"))
        if len(row) == 2:
            keyboard.append(row)
            row = []
    if row:
        keyboard.append(row)

    keyboard.append([
        InlineKeyboardButton("📁 Chọn vùng khác", callback_data="show_voice_cats"),
        InlineKeyboardButton("⬅️ Menu chính", callback_data="back_to_settings"),
    ])
    return InlineKeyboardMarkup(keyboard)


def get_speed_keyboard() -> InlineKeyboardMarkup:
    """Menu chọn tốc độ đọc."""
    keyboard = [
        [
            InlineKeyboardButton("🐢 0.85x (Chậm)", callback_data="set_spd_0.85"),
            InlineKeyboardButton("⚡ 1.0x (Chuẩn)", callback_data="set_spd_1.0"),
        ],
        [
            InlineKeyboardButton("🚀 1.15x (Hơi nhanh)", callback_data="set_spd_1.15"),
            InlineKeyboardButton("🔥 1.25x (Nhanh)", callback_data="set_spd_1.25"),
        ],
        [InlineKeyboardButton("⬅️ Quay lại", callback_data="back_to_settings")]
    ]
    return InlineKeyboardMarkup(keyboard)


def get_vol_keyboard() -> InlineKeyboardMarkup:
    """Menu chọn âm lượng âm thanh gốc trong video."""
    keyboard = [
        [
            InlineKeyboardButton("🔉 10% (Nhỏ vừa)", callback_data="set_vol_0.10"),
            InlineKeyboardButton("🔊 15% (Chuẩn)", callback_data="set_vol_0.15"),
        ],
        [
            InlineKeyboardButton("📢 25% (Rõ hơn)", callback_data="set_vol_0.25"),
            InlineKeyboardButton("🔇 0% (Tắt hẳn tiếng gốc)", callback_data="set_vol_0.0"),
        ],
        [InlineKeyboardButton("⬅️ Quay lại", callback_data="back_to_settings")]
    ]
    return InlineKeyboardMarkup(keyboard)


# ==========================================
# CÔNG CỤ TẠO GIỌNG ĐỌC TTS CHO TELEGRAM
# ==========================================
def synthesize_tts_for_telegram(text: str, voice: str = "Adam bựa", speed: float = 1.0, temperature: float = 0.8) -> Path:
    """Tổng hợp giọng đọc VieNeu-TTS và xuất ra file .ogg Opus chuẩn Voice Note của Telegram."""
    from auto_dub import VOICE_MAP
    from web_app import split_text_into_chunks, get_vieneu_model

    model = get_vieneu_model()
    actual_voice = VOICE_MAP.get(voice, voice)
    if actual_voice not in model._preset_voices and "ngan" in actual_voice.lower():
        actual_voice = "Thanh Bình"

    chunks = split_text_into_chunks(text, max_chars=180)
    audio_segments = []
    silence = np.zeros(int(48000 * 0.22), dtype=np.float32)

    for idx, chunk in enumerate(chunks):
        chunk_audio = model.infer(chunk, voice=actual_voice, temperature=temperature)
        if chunk_audio is not None and len(chunk_audio) > 0:
            audio_segments.append(chunk_audio)
            if idx < len(chunks) - 1:
                audio_segments.append(silence)

    if not audio_segments:
        raise ValueError("Không tạo được âm thanh từ văn bản cung cấp.")

    full_audio = np.concatenate(audio_segments)
    timestamp = int(time.time())
    raw_path = TTS_DIR / f"raw_tg_{timestamp}.wav"
    model.save(full_audio, str(raw_path))

    # Convert sang file OGG Opus chuẩn Voice Note của Telegram
    ogg_path = TTS_DIR / f"voice_tg_{timestamp}.ogg"

    filter_args = []
    if abs(float(speed) - 1.0) > 0.02:
        filter_args = ["-filter:a", f"atempo={speed}"]

    cmd = ["ffmpeg", "-y", "-i", str(raw_path)] + filter_args + [
        "-c:a", "libopus", "-b:a", "48k", "-application", "voip", str(ogg_path)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    if raw_path.exists():
        raw_path.unlink()

    return ogg_path


# ==========================================
# CÁC LỆNH TELEGRAM BOT
# ==========================================

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Xử lý lệnh /start."""
    welcome_text = (
        "👋 **Hế lô các con vợ ha ha ha!**\n\n"
        "Anh là **Bot Lồng Tiếng & Đọc Giọng AI Đỉnh Cao** (VieNeu-TTS 48kHz)!\n\n"
        "🔥 **Anh có thể giúp gì cho con vợ?**\n"
        "1️⃣ **Lồng tiếng video siêu tốc:**\n"
        "   - Gửi video vào đây ➜ Tự động bóc băng ➜ Dịch teen bựa/chuẩn ➜ Lồng tiếng VieNeu-TTS ➜ In phụ đề sắc nét!\n"
        "2️⃣ **Đọc văn bản thành giọng nói (TTS):**\n"
        "   - Gửi bất kỳ đoạn văn bản nào, hoặc dùng lệnh: `/tts [nội dung]`\n"
        "   - Anh đọc ngay bằng giọng **Nguyễn Ngọc Ngạn**, **Adam bựa**, **Thiền Tâm Đức**, **Ngọc Huyền**...\n\n"
        "⚙️ **Cài đặt cá nhân:** Gõ lệnh /settings để đổi giọng đọc, tốc độ nói, âm lượng tiếng gốc.\n"
        "📊 **Kiểm tra server:** Gõ lệnh /status\n"
        "❓ **Xem trợ giúp:** Gõ lệnh /help\n\n"
        "👉 *Giờ thì bắn ngay video hoặc đoạn chữ vào đây để anh biểu diễn nào các con vợ!*"
    )
    await update.message.reply_text(welcome_text, parse_mode=constants.ParseMode.MARKDOWN)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Xử lý lệnh /help."""
    help_text = (
        "📖 **HƯỚNG DẪN SỬ DỤNG BOT LỒNG TIẾNG & TTS ADAM BỰA**\n\n"
        "🎬 **1. Lồng tiếng cho Video:**\n"
        "- Gửi trực tiếp file Video hoặc File Document (.mp4, .mov, .mkv, .webm).\n"
        "- Giới hạn file qua Telegram thông thường: tối đa **20 MB** (Nên gửi clip dưới 2 phút để xử lý nhanh nhất).\n\n"
        "🗣️ **2. Chuyển Văn Bản Thành Giọng Nói (TTS):**\n"
        "- Cách 1: Gửi lệnh `/tts <nội dung văn bản>`\n"
        "- Cách 2: Nhắn trực tiếp một đoạn văn bản vào bot ➜ Bot sẽ hỏi bạn có muốn đọc thành Voice Note không.\n"
        "- Hỗ trợ các biểu cảm: `[cười]`, `[thở dài]`, `[hắng giọng]`\n\n"
        "⚙️ **3. Các lệnh chính:**\n"
        "- /settings : Tùy chỉnh giọng đọc, tốc độ, phong cách dịch, âm lượng tiếng gốc\n"
        "- /status : Xem tài nguyên CPU, RAM, hàng đợi server\n"
        "- /start : Mở đầu & giới thiệu\n\n"
        "🎉 Chúc con vợ có những giây phút trải nghiệm tuyệt vời ha ha ha!"
    )
    await update.message.reply_text(help_text, parse_mode=constants.ParseMode.MARKDOWN)


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Xử lý lệnh /status hiển thị thông số hệ thống."""
    settings = get_user_settings(context)
    is_busy = PROCESSING_LOCK.locked()

    sys_info = ""
    if psutil:
        try:
            cpu = psutil.cpu_percent(interval=0.1)
            ram = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            sys_info = (
                f"🖥️ **CPU:** `{cpu}%`\n"
                f"🧠 **RAM:** `{round((ram.total - ram.available)/(1024**3), 1)}/{round(ram.total/(1024**3), 1)} GB` (`{ram.percent}%`)\n"
                f"💾 **Ổ cứng trống:** `{round(disk.free/(1024**3), 1)} GB`\n"
            )
        except Exception:
            pass

    status_text = (
        f"📊 **TRẠNG THÁI HỆ THỐNG BOT AI**\n\n"
        f"{sys_info}"
        f"🚦 **Trạng thái xử lý:** {'🟡 Đang xử lý tác vụ...' if is_busy else '🟢 Đang rảnh (Sẵn sàng nhận lệnh)'}\n\n"
        f"⚙️ **Cài đặt hiện tại của con vợ:**\n"
        f"• Giọng đọc: `{settings['voice']}`\n"
        f"• Tốc độ nói: `{settings.get('speed', 1.0)}x`\n"
        f"• Phong cách: `{'Teen bựa' if settings['style'] == 'teen_bua' else 'Tiếng Việt chuẩn'}`\n"
        f"• Phụ đề cứng: `{'BẬT' if settings['burn_subtitles'] else 'TẮT'}`\n"
        f"• m lượng nền: `{int(settings.get('original_volume', 0.15) * 100)}%`"
    )
    await update.message.reply_text(status_text, parse_mode=constants.ParseMode.MARKDOWN)


async def settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Xử lý lệnh /settings."""
    settings = get_user_settings(context)
    reply_markup = get_settings_keyboard(settings)
    await update.message.reply_text(
        "⚙️ **BẢNG ĐIỀU KHIỂN & CÀI ĐẶT LỒNG TIẾNG**\n\n"
        "Chọn các nút bấm bên dưới để điều chỉnh theo ý con vợ:",
        reply_markup=reply_markup,
        parse_mode=constants.ParseMode.MARKDOWN
    )


async def tts_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Xử lý lệnh /tts <nội dung văn bản>."""
    message = update.message
    if not message:
        return

    # Lấy nội dung sau /tts
    args = context.args
    if not args:
        await message.reply_text(
            "⚠️ **Con vợ chưa nhập nội dung cần đọc!**\n\n"
            "Ví dụ:\n`/tts Kính thưa quý vị, hôm nay tôi xin kể một câu chuyện rất kỳ lạ.`",
            parse_mode=constants.ParseMode.MARKDOWN
        )
        return

    text = " ".join(args).strip()
    settings = get_user_settings(context)
    voice = settings["voice"]
    speed = settings.get("speed", 1.0)

    status_msg = await message.reply_text(
        f"🎙️ **Đang tổng hợp giọng đọc {voice}...**\n"
        f"⚡ Tốc độ: `{speed}x`\n"
        f"⏳ Chờ anh xíu nhé con vợ!",
        parse_mode=constants.ParseMode.MARKDOWN
    )

    try:
        ogg_path = await asyncio.to_thread(
            synthesize_tts_for_telegram,
            text=text,
            voice=voice,
            speed=speed,
            temperature=0.8
        )

        caption = f"🎙️ **Giọng đọc:** {voice} | ⚡ **Tốc độ:** {speed}x"
        with open(ogg_path, "rb") as vf:
            await message.reply_voice(
                voice=vf,
                caption=caption,
                parse_mode=constants.ParseMode.MARKDOWN
            )

        await status_msg.delete()

        # Dọn dẹp file tạm
        try:
            if ogg_path.exists():
                ogg_path.unlink()
        except Exception:
            pass

    except Exception as e:
        logger.exception("Lỗi sinh âm thanh TTS từ Telegram")
        await status_msg.edit_text(f"❌ Lỗi khi tạo giọng đọc: `{e}`", parse_mode=constants.ParseMode.MARKDOWN)


async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Xử lý khi người dùng gửi một tin nhắn văn bản thông thường."""
    message = update.message
    if not message or not message.text:
        return

    text = message.text.strip()
    if text.startswith("/"):
        return

    # Lưu đoạn text vào user_data
    context.user_data["pending_tts_text"] = text
    settings = get_user_settings(context)
    voice = settings["voice"]

    preview = text[:80] + "..." if len(text) > 80 else text

    keyboard = [
        [
            InlineKeyboardButton(f"🔊 Đọc Ngay bằng giọng {voice}", callback_data="do_tts_now")
        ],
        [
            InlineKeyboardButton("⚙️ Đổi Giọng / Cài Đặt", callback_data="show_voice_cats")
        ]
    ]

    await message.reply_text(
        f"💡 **Con vợ có muốn anh đọc đoạn văn bản này không?**\n\n"
        f"💬 *\"{preview}\"*\n\n"
        f"👉 Bấm nút bên dưới để anh biểu diễn VieNeu-TTS 48kHz ngay nhé:",
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode=constants.ParseMode.MARKDOWN
    )


async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Xử lý các sự kiện bấm nút inline."""
    query = update.callback_query
    await query.answer()
    data = query.data
    settings = get_user_settings(context)

    if data == "toggle_style":
        settings["style"] = "normal" if settings["style"] == "teen_bua" else "teen_bua"
        await query.edit_message_reply_markup(reply_markup=get_settings_keyboard(settings))

    elif data == "toggle_sub":
        settings["burn_subtitles"] = not settings["burn_subtitles"]
        await query.edit_message_reply_markup(reply_markup=get_settings_keyboard(settings))

    elif data == "show_voice_cats":
        await query.edit_message_text(
            "🗣️ **CHỌN DANH MỤC VÙNG MIỀN GIỌNG ĐỌC**\n\nChọn một danh mục để xem danh sách giọng:",
            reply_markup=get_voice_cats_keyboard(),
            parse_mode=constants.ParseMode.MARKDOWN
        )

    elif data.startswith("vcat_"):
        cat_key = data.replace("vcat_", "")
        cat_title = VOICE_CATEGORIES.get(cat_key, {}).get("title", "Danh sách giọng")
        await query.edit_message_text(
            f"🗣️ **{cat_title}**\n\nBấm vào một giọng để chọn:",
            reply_markup=get_voices_in_category_keyboard(cat_key),
            parse_mode=constants.ParseMode.MARKDOWN
        )

    elif data.startswith("voice_"):
        chosen_voice = data.replace("voice_", "")
        settings["voice"] = chosen_voice
        await query.edit_message_text(
            f"✅ **Đã chọn giọng đọc:** `{chosen_voice}`\n\nQuay lại bảng điều khiển:",
            reply_markup=get_settings_keyboard(settings),
            parse_mode=constants.ParseMode.MARKDOWN
        )

    elif data == "show_speed_menu":
        await query.edit_message_text(
            "⚡ **ĐIỀU CHỈNH TỐC ĐỘ NÓI (SPEED)**\n\nChọn tốc độ đọc mong muốn:",
            reply_markup=get_speed_keyboard(),
            parse_mode=constants.ParseMode.MARKDOWN
        )

    elif data.startswith("set_spd_"):
        spd = float(data.replace("set_spd_", ""))
        settings["speed"] = spd
        await query.edit_message_text(
            f"✅ **Đã đặt tốc độ nói:** `{spd}x`\n\nQuay lại bảng điều khiển:",
            reply_markup=get_settings_keyboard(settings),
            parse_mode=constants.ParseMode.MARKDOWN
        )

    elif data == "show_vol_menu":
        await query.edit_message_text(
            "🔊 **ĐIỀU CHỈNH ÂM LƯỢNG TIẾNG GỐC TRONG VIDEO**\n\nChọn mức âm lượng nền:",
            reply_markup=get_vol_keyboard(),
            parse_mode=constants.ParseMode.MARKDOWN
        )

    elif data.startswith("set_vol_"):
        vol = float(data.replace("set_vol_", ""))
        settings["original_volume"] = vol
        await query.edit_message_text(
            f"✅ **Đã đặt âm lượng tiếng gốc:** `{int(vol * 100)}%`\n\nQuay lại bảng điều khiển:",
            reply_markup=get_settings_keyboard(settings),
            parse_mode=constants.ParseMode.MARKDOWN
        )

    elif data == "back_to_settings":
        await query.edit_message_text(
            "⚙️ **BẢNG ĐIỀU KHIỂN & CÀI ĐẶT LỒNG TIẾNG**",
            reply_markup=get_settings_keyboard(settings),
            parse_mode=constants.ParseMode.MARKDOWN
        )

    elif data == "reset_settings":
        context.user_data["settings"] = DEFAULT_SETTINGS.copy()
        await query.edit_message_reply_markup(reply_markup=get_settings_keyboard(context.user_data["settings"]))

    elif data == "close_settings":
        await query.edit_message_text("✅ Đã lưu cấu hình cài đặt của con vợ!")

    elif data == "do_tts_now":
        pending_text = context.user_data.get("pending_tts_text", "")
        if not pending_text:
            await query.edit_message_text("⚠️ Không tìm thấy đoạn văn bản cần đọc!")
            return

        voice = settings["voice"]
        speed = settings.get("speed", 1.0)
        await query.edit_message_text(
            f"🎙️ **Đang tổng hợp giọng đọc {voice}...**\n"
            f"⚡ Tốc độ: `{speed}x`\n"
            f"⏳ Chờ anh vài giây nhé con vợ!",
            parse_mode=constants.ParseMode.MARKDOWN
        )

        try:
            ogg_path = await asyncio.to_thread(
                synthesize_tts_for_telegram,
                text=pending_text,
                voice=voice,
                speed=speed,
                temperature=0.8
            )

            caption = f"🎙️ **Giọng đọc:** {voice} | ⚡ **Tốc độ:** {speed}x"
            with open(ogg_path, "rb") as vf:
                await query.message.reply_voice(
                    voice=vf,
                    caption=caption,
                    parse_mode=constants.ParseMode.MARKDOWN
                )

            await query.message.delete()

            # Dọn dẹp file tạm
            try:
                if ogg_path.exists():
                    ogg_path.unlink()
            except Exception:
                pass

        except Exception as e:
            logger.exception("Lỗi sinh âm thanh TTS từ Telegram")
            await query.message.reply_text(f"❌ Lỗi khi tạo giọng đọc: `{e}`", parse_mode=constants.ParseMode.MARKDOWN)


class ProgressTracker:
    """Quản lý cập nhật tiến độ xử lý video lên tin nhắn Telegram có throttle chống rate-limit."""
    def __init__(self, message, loop):
        self.message = message
        self.loop = loop
        self.last_update_time = 0.0
        self.last_text = ""

    def update(self, current_step: int, total_steps: int, desc: str):
        now = time.time()
        # Cập nhật nếu qua 1.5s hoặc là bước cuối
        if (now - self.last_update_time >= 1.5) or (current_step == total_steps):
            self.last_update_time = now
            progress_bar = "▓" * current_step + "░" * (total_steps - current_step)
            text = (
                f"⚙️ **Đang lồng tiếng video cho con vợ...**\n\n"
                f"[{progress_bar}] Bước {current_step}/{total_steps}\n"
                f"▶ **{desc}**\n\n"
                f"⏳ Vui lòng chờ vài giây nhé!"
            )
            if text != self.last_text:
                self.last_text = text
                asyncio.run_coroutine_threadsafe(self._safe_edit(text), self.loop)

    async def _safe_edit(self, text: str):
        try:
            await self.message.edit_text(text, parse_mode=constants.ParseMode.MARKDOWN)
        except Exception:
            pass


async def handle_video_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Xử lý khi người dùng gửi video hoặc file video."""
    message = update.message
    if not message:
        return

    # Lấy thông tin video từ message.video hoặc message.document
    video_obj = message.video
    if not video_obj and message.document:
        mime = message.document.mime_type or ""
        if mime.startswith("video/") or (message.document.file_name and message.document.file_name.lower().endswith((".mp4", ".mov", ".mkv", ".webm", ".avi"))):
            video_obj = message.document

    if not video_obj:
        await message.reply_text("🤔 Con vợ gửi file gì đấy? Gửi video định dạng MP4/MOV/MKV vào đây anh mới lồng tiếng được chứ ha ha!")
        return

    # Kiểm tra giới hạn 20MB của Telegram Bot API
    file_size = getattr(video_obj, "file_size", 0) or 0
    if file_size > MAX_FILE_SIZE_BYTES:
        size_mb = file_size / (1024 * 1024)
        await message.reply_text(
            f"⚠️ **Video quá lớn ({size_mb:.1f} MB)!**\n\n"
            f"Telegram Bot API giới hạn nhận file qua bot tối đa **20 MB**.\n"
            f"Con vợ vui lòng nén video hoặc cắt ngắn dưới 20 MB rồi gửi lại cho anh nhé!",
            parse_mode=constants.ParseMode.MARKDOWN
        )
        return

    settings = get_user_settings(context)
    user_id = update.effective_user.id
    timestamp = int(time.time())
    original_filename = getattr(video_obj, "file_name", None) or f"video_{timestamp}.mp4"
    safe_base = Path(original_filename).stem.replace(" ", "_")
    input_file_path = DOWNLOAD_DIR / f"{user_id}_{timestamp}_{safe_base}.mp4"

    status_msg = await message.reply_text("📥 **Đang tải video về server...** Chờ anh xíu nhé!", parse_mode=constants.ParseMode.MARKDOWN)

    try:
        tg_file = await video_obj.get_file()
        await tg_file.download_to_drive(custom_path=str(input_file_path))
    except Exception as e:
        logger.error(f"Lỗi tải video: {e}")
        await status_msg.edit_text(f"❌ Không tải được video từ Telegram: {e}")
        return

    # Thông báo hàng đợi nếu bot đang bận
    if PROCESSING_LOCK.locked():
        await status_msg.edit_text(
            "⏳ **Video đã vào hàng đợi!**\n\n"
            "Server đang xử lý một video khác, video của con vợ sẽ được xử lý ngay sau đó...",
            parse_mode=constants.ParseMode.MARKDOWN
        )

    # Thực hiện xử lý có Lock bảo vệ tài nguyên phần cứng
    async with PROCESSING_LOCK:
        loop = asyncio.get_running_loop()
        tracker = ProgressTracker(status_msg, loop)

        try:
            await status_msg.edit_text(
                f"🚀 **Bắt đầu lồng tiếng!**\n\n"
                f"🎙️ Giọng: `{settings['voice']}`\n"
                f"⚡ Tốc độ: `{settings.get('speed', 1.0)}x`\n"
                f"🎭 Phong cách: `{'Teen bựa' if settings['style'] == 'teen_bua' else 'Tiếng Việt chuẩn'}`\n"
                f"📝 Phụ đề: `{'BẬT' if settings['burn_subtitles'] else 'TẮT'}`\n\n"
                f"⏳ Đang khởi động AI...",
                parse_mode=constants.ParseMode.MARKDOWN
            )

            from auto_dub import process_video

            user_output_dir = OUTPUT_DIR / f"tg_{user_id}_{timestamp}"
            user_output_dir.mkdir(parents=True, exist_ok=True)

            out_video_path, srt_path, metadata_json_path = await asyncio.to_thread(
                process_video,
                input_video=str(input_file_path),
                output_dir=str(user_output_dir),
                whisper_model_size=settings["whisper_model"],
                tts_engine="vieneu",
                voice=settings["voice"],
                style=settings["style"],
                original_volume=settings.get("original_volume", 0.15),
                burn_subtitles=settings["burn_subtitles"],
                progress_callback=tracker.update
            )

            if out_video_path == "NO_AUDIO":
                await status_msg.edit_text(
                    "⚠️ **Video này của con vợ KHÔNG CÓ ÂM THANH!**\n\n"
                    "Video này (đặc biệt là video tải từ Facebook/TikTok qua các trang web download) bị thiếu luồng tiếng (không có Audio track).\n"
                    "Con vợ kiểm tra lại xem video gốc có phát ra tiếng không, hoặc gửi video khác có âm thanh người nói vào nhé ha ha!",
                    parse_mode=constants.ParseMode.MARKDOWN
                )
                return

            if not out_video_path or not os.path.exists(out_video_path):
                await status_msg.edit_text(
                    "⚠️ **Không phát hiện giọng nói nào trong video của con vợ!**\n"
                    "Con vợ kiểm tra lại xem video có âm thanh người nói không nhé ha ha!",
                    parse_mode=constants.ParseMode.MARKDOWN
                )
                return

            await status_msg.edit_text("📤 **Đang tải video thành phẩm lên Telegram...** Sắp xong rồi!")
            await context.bot.send_chat_action(chat_id=message.chat_id, action=constants.ChatAction.UPLOAD_VIDEO)

            caption = (
                f"🎉 **LỒNG TIẾNG THÀNH CÔNG!** ha ha ha\n\n"
                f"🎙️ Giọng đọc: **{settings['voice']}**\n"
                f"⚡ Tốc độ: **{settings.get('speed', 1.0)}x**\n"
                f"🎭 Phong cách: **{'Teen bựa (Xưng anh - Con vợ)' if settings['style'] == 'teen_bua' else 'Tiếng Việt chuẩn'}**\n"
                f"⚡ Sức mạnh VieNeu-TTS v3 Turbo 48kHz"
            )

            with open(out_video_path, "rb") as vf:
                await message.reply_video(
                    video=vf,
                    caption=caption,
                    parse_mode=constants.ParseMode.MARKDOWN,
                    supports_streaming=True
                )

            # Gửi kèm file phụ đề SRT
            if srt_path and os.path.exists(srt_path):
                with open(srt_path, "rb") as sf:
                    await message.reply_document(
                        document=sf,
                        caption="📄 File phụ đề tiếng Việt (.srt) đi kèm đây nhé con vợ!"
                    )

            await status_msg.edit_text("✅ **Đã hoàn tất 100%!** Chúc con vợ thưởng thức video vui vẻ ha ha ha!")

        except Exception as err:
            logger.exception("Lỗi trong quá trình lồng tiếng")
            await status_msg.edit_text(f"❌ Có lỗi xảy ra trong lúc lồng tiếng: `{err}`", parse_mode=constants.ParseMode.MARKDOWN)

        finally:
            try:
                if input_file_path.exists():
                    input_file_path.unlink()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description="Telegram Bot tự động lồng tiếng phong cách Adam bựa")
    parser.add_argument("--token", default=os.getenv("TELEGRAM_BOT_TOKEN"), help="Token của Telegram Bot")
    args = parser.parse_args()

    token = args.token
    if not token or token == "YOUR_TELEGRAM_BOT_TOKEN_HERE":
        print("\n" + "=" * 65)
        print("❌ LỖI: Chưa cấu hình TELEGRAM_BOT_TOKEN!")
        print("=" * 65 + "\n")
        return

    print("\n" + "=" * 65)
    print("🦜 KHỞI ĐỘNG TELEGRAM BOT LỒNG TIẾNG & TTS ADAM BỰA")
    print("=" * 65)
    print("✅ Đã nạp Token thành công!")
    print("🤖 Bot đang lắng nghe tin nhắn từ Telegram (Polling)...")
    print("=" * 65 + "\n")

    app = ApplicationBuilder().token(token).build()

    # Đăng ký các handler lệnh
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("settings", settings_command))
    app.add_handler(CommandHandler("status", status_command))
    app.add_handler(CommandHandler("tts", tts_command))
    app.add_handler(CallbackQueryHandler(callback_handler))

    # Đăng ký handler xử lý video & document
    app.add_handler(MessageHandler(filters.VIDEO | filters.Document.VIDEO | filters.Document.ALL, handle_video_message))

    # Đăng ký handler xử lý tin nhắn văn bản (Smart Text-to-Speech)
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_text_message))

    # Khởi chạy bot
    app.run_polling()


if __name__ == "__main__":
    main()
