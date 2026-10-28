import os
import sys
import time
import asyncio
import logging
import argparse
from pathlib import Path
from dotenv import load_dotenv
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

# Đảm bảo in tiếng Việt trên console Windows không lỗi font
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

# Cấu hình logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Nạp cấu hình từ .env
load_dotenv()

# Khóa xử lý đồng thời (Chạy 1 video tại một thời điểm để tối ưu CPU/RAM)
PROCESSING_LOCK = asyncio.Lock()

# Thư mục lưu trữ tạm thời
DOWNLOAD_DIR = Path("downloads")
OUTPUT_DIR = Path("output")
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Cấu hình mặc định cho người dùng
DEFAULT_SETTINGS = {
    "voice": "Adam bựa",
    "style": "teen_bua",
    "burn_subtitles": True,
    "whisper_model": "base",
    "original_volume": 0.15,
}

# Giới hạn kích thước file Telegram Bot API tải trực tiếp (20 MB)
MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024


def get_user_settings(context: ContextTypes.DEFAULT_TYPE) -> dict:
    """Lấy hoặc khởi tạo cấu hình riêng cho từng người dùng."""
    if "settings" not in context.user_data:
        context.user_data["settings"] = DEFAULT_SETTINGS.copy()
    return context.user_data["settings"]


def get_settings_keyboard(settings: dict) -> InlineKeyboardMarkup:
    """Tạo bàn phím nút bấm tương tác cho cài đặt."""
    style_text = "🎭 Teen bựa (Xưng anh - Con vợ)" if settings["style"] == "teen_bua" else "👔 Tiếng Việt chuẩn"
    sub_text = "✅ Đang BẬT" if settings["burn_subtitles"] else "❌ Đang TẮT"

    keyboard = [
        [
            InlineKeyboardButton(f"🎭 Phong cách: {style_text}", callback_data="toggle_style")
        ],
        [
            InlineKeyboardButton(f"🗣️ Giọng đọc: {settings['voice']}", callback_data="show_voices")
        ],
        [
            InlineKeyboardButton(f"📝 Phụ đề dán hình: {sub_text}", callback_data="toggle_sub")
        ],
        [
            InlineKeyboardButton("🔄 Đặt lại mặc định", callback_data="reset_settings"),
            InlineKeyboardButton("❌ Đóng menu", callback_data="close_settings"),
        ]
    ]
    return InlineKeyboardMarkup(keyboard)


def get_voice_keyboard() -> InlineKeyboardMarkup:
    """Tạo bàn phím chọn giọng đọc."""
    voices = [
        ("🦜 Adam bựa (Hot)", "voice_Adam bựa"),
        ("🎙️ Adam chuẩn", "voice_Adam"),
        ("🎙️ Hải Đăng (Nam)", "voice_Hải Đăng"),
        ("🎙️ Mai Anh (Nữ)", "voice_Mai Anh"),
    ]
    keyboard = []
    for text, cdata in voices:
        keyboard.append([InlineKeyboardButton(text, callback_data=cdata)])
    keyboard.append([InlineKeyboardButton("⬅️ Quay lại", callback_data="back_to_settings")])
    return InlineKeyboardMarkup(keyboard)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Xử lý lệnh /start."""
    welcome_text = (
        "👋 **Hế lô các con vợ ha ha ha!**\n\n"
        "Anh là **Bot lồng tiếng tự động phong cách Adam bựa** (VieNeu-TTS 48kHz)!\n\n"
        "🎬 **Quy trình hoạt động siêu tốc:**\n"
        "1️⃣ Con vợ chỉ cần gửi **1 file video** hoặc file tài liệu video vào đây.\n"
        "2️⃣ Anh tự động bóc băng lời thoại bằng AI (Faster-Whisper).\n"
        "3️⃣ Dịch sang tiếng Việt chuẩn phong cách **Teen Bựa** (Xưng anh, gọi các con vợ, chèn biểu cảm cười ha ha ha).\n"
        "4️⃣ Lồng tiếng mượt mà bằng công nghệ **VieNeu-TTS** (Giọng Adam bựa siêu cuốn).\n"
        "5️⃣ In phụ đề sắc nét và gửi trả lại video thành phẩm cho con vợ ngay!\n\n"
        "⚙️ **Cài đặt cá nhân:** Dùng lệnh /settings để đổi giọng đọc hoặc phong cách dịch.\n"
        "❓ **Trợ giúp:** Dùng lệnh /help.\n\n"
        "👉 *Giờ thì bắn ngay video vào đây để anh biểu diễn nào các con vợ!*"
    )
    await update.message.reply_text(welcome_text, parse_mode=constants.ParseMode.MARKDOWN)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Xử lý lệnh /help."""
    help_text = (
        "📖 **HƯỚNG DẪN SỬ DỤNG BOT LỒNG TIẾNG ADAM BỰA**\n\n"
        "🔹 **Cách gửi video:**\n"
        "- Gửi trực tiếp dạng Video hoặc Document (định dạng .mp4, .mov, .mkv, .webm).\n"
        "- Do giới hạn của Telegram Bot API thông thường, dung lượng file gửi qua bot tối đa là **20 MB**.\n"
        "- Mẹo: Nên gửi video ngắn dưới 1 - 2 phút để xử lý nhanh và không bị vượt dung lượng 20MB.\n\n"
        "🔹 **Các lệnh hỗ trợ:**\n"
        "- /start : Khởi động bot & xem giới thiệu\n"
        "- /settings : Tùy chỉnh giọng đọc, phong cách dịch, bật/tắt phụ đề\n"
        "- /help : Xem hướng dẫn này\n\n"
        "🎉 Chúc con vợ có những giây phút cười ra nước mắt ha ha ha!"
    )
    await update.message.reply_text(help_text, parse_mode=constants.ParseMode.MARKDOWN)


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

    elif data == "show_voices":
        await query.edit_message_reply_markup(reply_markup=get_voice_keyboard())

    elif data.startswith("voice_"):
        chosen_voice = data.replace("voice_", "")
        settings["voice"] = chosen_voice
        await query.edit_message_reply_markup(reply_markup=get_settings_keyboard(settings))

    elif data == "back_to_settings":
        await query.edit_message_reply_markup(reply_markup=get_settings_keyboard(settings))

    elif data == "reset_settings":
        context.user_data["settings"] = DEFAULT_SETTINGS.copy()
        await query.edit_message_reply_markup(reply_markup=get_settings_keyboard(context.user_data["settings"]))

    elif data == "close_settings":
        await query.edit_message_text("✅ Đã lưu cấu hình cài đặt của con vợ!")


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
                f"[{progress_bar}] {current_step}/{total_steps}\n"
                f"▶ **{desc}**\n\n"
                f"⏳ Vui lòng chờ vài giây nhé!"
            )
            if text != self.last_text:
                self.last_text = text
                # Gọi edit_message_text an toàn trên async loop
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
    is_doc = False
    if not video_obj and message.document:
        mime = message.document.mime_type or ""
        if mime.startswith("video/") or (message.document.file_name and message.document.file_name.lower().endswith((".mp4", ".mov", ".mkv", ".webm", ".avi"))):
            video_obj = message.document
            is_doc = True

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

    # Thông báo bắt đầu tải
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
                f"🎭 Phong cách: `{settings['style']}`\n"
                f"📝 Phụ đề: `{'BẬT' if settings['burn_subtitles'] else 'TẮT'}`\n\n"
                f"⏳ Đang khởi động AI...",
                parse_mode=constants.ParseMode.MARKDOWN
            )

            # Nạp pipeline xử lý auto_dub
            from auto_dub import process_video

            # Chạy pipeline xử lý đồng bộ trên thread riêng
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
                original_volume=settings["original_volume"],
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

            # Cập nhật trạng thái gửi video
            await status_msg.edit_text("📤 **Đang tải video thành phẩm lên Telegram...** Sắp xong rồi!")
            await context.bot.send_chat_action(chat_id=message.chat_id, action=constants.ChatAction.UPLOAD_VIDEO)

            # Gửi video thành phẩm
            caption = (
                f"🎉 **LỒNG TIẾNG THÀNH CÔNG!** ha ha ha\n\n"
                f"🎙️ Giọng đọc: **{settings['voice']}**\n"
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

            # Gửi kèm file phụ đề SRT nếu người dùng cần
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
            # Dọn dẹp file tải về ban đầu để tiết kiệm dung lượng ổ cứng
            try:
                if input_file_path.exists():
                    input_file_path.unlink()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description="Telegram Bot tự động lồng tiếng phong cách Adam bựa")
    parser.add_argument("--token", default=os.getenv("TELEGRAM_BOT_TOKEN"), help="Token của Telegram Bot (lấy từ @BotFather)")
    args = parser.parse_args()

    token = args.token
    if not token or token == "YOUR_TELEGRAM_BOT_TOKEN_HERE":
        print("\n" + "=" * 65)
        print("❌ LỖI: Chưa cấu hình TELEGRAM_BOT_TOKEN!")
        print("=" * 65)
        print("👉 Hướng dẫn lấy Token miễn phí trong 1 phút:")
        print("1. Mở Telegram và tìm kiếm bot: @BotFather")
        print("2. Gửi lệnh: /newbot và làm theo hướng dẫn để đặt tên bot")
        print("3. Sao chép chuỗi HTTP API Token nhận được (dạng: 123456789:ABCdef...)")
        print("4. Điền Token vào file .env:")
        print("   TELEGRAM_BOT_TOKEN=123456789:ABCdef...")
        print("   hoặc chạy lệnh: python telegram_bot.py --token <TOKEN_CUA_BAN>")
        print("=" * 65 + "\n")
        return

    print("\n" + "=" * 65)
    print("🦜 KHỞI ĐỘNG TELEGRAM BOT LỒNG TIẾNG ADAM BỰA")
    print("=" * 65)
    print("✅ Đã nạp Token thành công!")
    print("🤖 Bot đang lắng nghe tin nhắn từ Telegram (Polling)...")
    print("👉 Hãy mở Telegram, tìm bot của bạn và gửi /start hoặc gửi video!")
    print("=" * 65 + "\n")

    app = ApplicationBuilder().token(token).build()

    # Đăng ký các handler lệnh
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("settings", settings_command))
    app.add_handler(CallbackQueryHandler(callback_handler))

    # Đăng ký handler xử lý video & document
    app.add_handler(MessageHandler(filters.VIDEO | filters.Document.VIDEO | filters.Document.ALL, handle_video_message))

    # Khởi chạy bot
    app.run_polling()


if __name__ == "__main__":
    main()
