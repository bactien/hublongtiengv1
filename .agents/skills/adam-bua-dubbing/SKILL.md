---
name: adam-bua-dubbing
description: >-
  Tự động nhận diện lời thoại, dịch sang phong cách teen bựa (xưng anh, gọi các con vợ, chèn biểu cảm ha ha ha...) và lồng tiếng bằng giọng "Adam bựa" (VieNeu-TTS v3 Turbo 48kHz) kèm phụ đề hardsub/softsub.
---

# 🦜 Skill: Lồng tiếng & Dịch phong cách "Adam bựa" (Xưng anh - Gọi các con vợ)

Skill này kích hoạt quy trình tự động hóa chuyển đổi video ngoại ngữ sang video thuyết minh tiếng Việt với phong cách **hài hước, bựa, lầy lội**, xưng **anh** và gọi khán giả là **các con vợ**, chèn tiếng cười **ha ha ha...** hoặc tag cảm xúc `[cười]`, được đọc bằng chất giọng **Adam bựa** của **VieNeu-TTS (48 kHz)**.

---

## 🚀 1. Lệnh thực thi nhanh (Quick Run)

### Cách 1: Sử dụng script `dub_bua.bat`
```powershell
# Chạy với video bất kỳ:
.\dub_bua.bat "duong_dan_video.mp4"

# Ví dụ chạy với video phỏng vấn mẫu trong thư mục:
.\dub_bua.bat samples\jobinterview.mp4
```

### Cách 2: Chạy trực tiếp qua Python CLI
```powershell
.\.venv\Scripts\python.exe auto_dub.py --input_file "duong_dan_video.mp4" --voice "Adam bựa" --style teen_bua --burn_subtitles
```

### Cách 3: Chạy Telegram Bot (Gửi video qua bot -> Nhận video lồng tiếng tự động)
```powershell
# Chạy file khởi động nhanh:
.\run_telegram_bot.bat

# Hoặc truyền token trực tiếp:
.\.venv\Scripts\python.exe telegram_bot.py --token <TOKEN_BOT_CUA_BAN>
```

---

## 🎭 2. Quy tắc dịch phong cách "Teen Bựa"

1. **Xưng hô:**
   - Người nói luôn xưng: **Anh** / **Anh đây** / **Bọn anh**
   - Khán giả / Đối phương luôn gọi là: **Các con vợ** / **Con vợ**
2. **Chào hỏi & Biểu cảm:**
   - Mở đầu bằng các câu chào tếu táo: *"Hế lô các con vợ ha ha ha"*, *"Chào cả lò nhà các con vợ nhá ha ha"*
   - Tự động chèn các đoạn cười duyên: `ha ha ha...`, `[cười]`, `[hắng giọng]`
3. **Tiếng lóng thời thượng (Gen Z / Mạng xã hội):**
   - *Yêu thích, mê* ➔ *Mê vãi chưởng / Mê tít thò lò ha ha*
   - *Tuyệt vời, xuất sắc* ➔ *Hết nước chấm luôn ha ha / Đỉnh nóc kịch trần*
   - *Kinh nghiệm* ➔ *Trình độ thì cứ gọi là thừa mứa*
   - *Làm việc chăm chỉ* ➔ *Cày bục cả mặt*
   - *Cảm ơn* ➔ *Đa tạ các con vợ nhá [cười]*
   - *Tạm biệt* ➔ *Anh lượn đây các con vợ ha ha ha*

---

## ✍️ 3. Tùy chỉnh câu bựa theo ý muốn (Interactive Scripting)

Khi chạy xong bước 2 và 3, hệ thống sẽ sinh ra một file kịch bản JSON:
`output\<ten_video>_metadata.json`

Bạn có thể mở file này ra, tự do chỉnh sửa câu thoại theo chất riêng của mình (ví dụ thêm câu chửi yêu, tag biểu cảm `[cười]`, `[thở dài]`), sau đó chạy lại lệnh:
```powershell
.\dub_bua.bat "duong_dan_video.mp4"
```
Hệ thống sẽ **ngay lập tức đọc theo đúng từng chữ bạn đã sửa trong file JSON** mà không dịch đè lại!

---

## ⚙️ 4. Các tham số nâng cao

| Tham số | Ý nghĩa | Mặc định |
| :--- | :--- | :--- |
| `--input_file` | Đường dẫn file video cần lồng tiếng | `e2e-tests/englishvideo.mp4` |
| `--voice` | Giọng đọc VieNeu-TTS | `"Adam bựa"` (hoặc `"Adam"`, `"Hải Đăng"`, `"Mai Anh"`) |
| `--style` | Phong cách dịch | `teen_bua` (hoặc `normal` nếu muốn dịch nghiêm túc) |
| `--original_volume` | Âm lượng nhạc nền gốc (0.15 = 15%) | `0.15` |
| `--burn_subtitles` | In phụ đề cứng trực tiếp lên hình | `True` |
| `--whisper_model` | Model nhận diện giọng nói | `base` (hoặc `small`, `medium`) |
