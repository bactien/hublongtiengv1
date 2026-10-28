import sys
import os
import json
import argparse
from auto_dub import convert_to_teen_bua

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

def main():
    parser = argparse.ArgumentParser(description="Công cụ biên kịch & chuyển đổi content phong cách Adam bựa (Xưng anh - Gọi các con vợ)")
    parser.add_argument("--text", type=str, help="Đoạn văn bản cần chuyển đổi sang phong cách bựa")
    parser.add_argument("--json_file", type=str, help="Đường dẫn file metadata JSON (ví dụ output/video_metadata.json) cần rewrite toàn bộ câu")
    
    args = parser.parse_args()
    
    if args.text:
        result = convert_to_teen_bua(args.text)
        print("\n================== KẾT QUẢ CONTENT ADAM BỰA ==================")
        print(f"Gốc:   {args.text}")
        print(f"Bựa:   {result}")
        print("==============================================================\n")
    elif args.json_file:
        if not os.path.exists(args.json_file):
            print(f"Lỗi: Không tìm thấy file {args.json_file}")
            return
        with open(args.json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            base_vi = item.get("vi_text") or item.get("text", "")
            item["vi_text"] = convert_to_teen_bua(base_vi)
        with open(args.json_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"\n✅ Đã cập nhật toàn bộ {len(data)} câu trong {args.json_file} sang phong cách Adam bựa!\n")
    else:
        # Chế độ tương tác nhanh
        print("\nNhập câu bạn muốn chuyển thể (hoặc gõ 'exit' để thoát):")
        while True:
            try:
                line = input("\n[Nhập]: ").strip()
                if not line or line.lower() == "exit":
                    break
                print("[Adam bựa]:", convert_to_teen_bua(line))
            except (KeyboardInterrupt, EOFError):
                break

if __name__ == "__main__":
    main()
