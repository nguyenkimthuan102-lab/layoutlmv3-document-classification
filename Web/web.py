import os
import sys
import torch
import numpy as np
from PIL import Image, ImageDraw

# Thư viện giao diện Web
import gradio as gr

# Thư viện Model AI
from transformers import AutoProcessor, LayoutLMv3ForSequenceClassification

# ==============================================================================
# CẤU HÌNH MÔI TRƯỜNG WINDOWS
# ==============================================================================
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['FLAGS_use_mkldnn'] = '0'

#  QUAN TRỌNG: BẠN PHẢI SỬA ĐƯỜNG DẪN NÀY TRỎ VÀO THƯ MỤC MODEL TRÊN MÁY BẠN
PATH_BEST_CHECKPOINT = r"D:\\Study\\Computer Vision\\Đồ Án\\Web\\checkpoint-11976"

#  Token ngrok của bạn (lấy tại https://dashboard.ngrok.com/get-started/your-authtoken)
NGROK_AUTH_TOKEN = "paste_your_token_here."

# ==============================================================================
# 1. KHỞI TẠO PADDLEOCR
# ==============================================================================
print(" Đang khởi tạo PaddleOCR...")
try:
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(
        use_textline_orientation=True,
        lang='en',
        enable_mkldnn=False
    )
    print(" Khởi tạo PaddleOCR thành công!")
except Exception as e:
    print(f" Lỗi khởi tạo OCR: {e}")
    sys.exit("Dừng chương trình vì OCR không khởi tạo được.")

# ==============================================================================
# 2. KHỞI TẠO MODEL VÀ PROCESSOR
# ==============================================================================
if not os.path.isdir(PATH_BEST_CHECKPOINT):
    sys.exit(f" KHÔNG TÌM THẤY THƯ MỤC MODEL TẠI: {PATH_BEST_CHECKPOINT}\nHãy tải model về và sửa lại biến PATH_BEST_CHECKPOINT.")

print(" Đang tải mô hình LayoutLMv3...")
try:
    model = LayoutLMv3ForSequenceClassification.from_pretrained(
        PATH_BEST_CHECKPOINT, local_files_only=True
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
    print(f" Tải mô hình hoàn tất. Đang chạy trên: {device}")
except Exception as e:
    sys.exit(f" Lỗi tải model: {e}")

# ==============================================================================
# 3. HÀM XỬ LÝ LÕI (INFERENCE)
# ==============================================================================
def classify_document(input_image_np):
    if input_image_np is None:
        return None, "Vui lòng tải lên ảnh", "", None, ""

    try:
        img_raw = Image.fromarray(input_image_np).convert("RGB")
        img_w, img_h = img_raw.size

        # 1. Chạy OCR bằng API mới: predict() thay cho ocr() (đã deprecated ở paddleocr 3.x)
        ocr_results = ocr.predict(input_image_np)

        words, boxes, raw_boxes = [], [], []
        if ocr_results and len(ocr_results) > 0:
            res = ocr_results[0]
            texts = res.get('rec_texts', []) if hasattr(res, 'get') else res['rec_texts']
            scores = res.get('rec_scores', []) if hasattr(res, 'get') else res['rec_scores']
            # rec_polys ưu tiên (4 điểm xoay), fallback dt_polys
            polys = res.get('rec_polys', None) if hasattr(res, 'get') else res.get('rec_polys', None)
            if polys is None:
                polys = res.get('dt_polys', [])

            for text, score, poly in zip(texts, scores, polys):
                if score < 0.6:  # Lọc bỏ các từ có độ tin cậy thấp
                    continue
                words.append(text)
                pts = np.array(poly).tolist()
                raw_boxes.append(pts)
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                # Chuẩn hóa tọa độ về scale 0-1000 cho LayoutLMv3
                boxes.append([
                    int(min(max(0, min(xs) / img_w * 1000), 1000)),
                    int(min(max(0, min(ys) / img_h * 1000), 1000)),
                    int(min(max(0, max(xs) / img_w * 1000), 1000)),
                    int(min(max(0, max(ys) / img_h * 1000), 1000)),
                ])

        if not words:
            words, boxes, raw_boxes = ["empty"], [[0, 0, 0, 0]], []

        # 2. Vẽ viền đỏ (Bounding Box) lên ảnh demo
        img_annotated = img_raw.copy()
        draw = ImageDraw.Draw(img_annotated)
        for b in raw_boxes:
            pts = [(int(x[0]), int(x[1])) for x in b]
            draw.polygon(pts, outline=(255, 80, 80))

        # 3. Đẩy vào Model LayoutLMv3 dự đoán
        inputs = processor(
            img_raw, words, boxes=boxes,
            return_tensors="pt", truncation=True
        ).to(model.device)

        model.eval()
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            pred_idx = torch.argmax(logits, dim=-1).item()
            confidence = probs[0][pred_idx].item()

        label = model.config.id2label[pred_idx].upper()

        # 4. Trích xuất Top 5 dự đoán cho giao diện Web
        top5 = torch.topk(probs[0], k=5)
        confidences_dict = {
            model.config.id2label[idx.item()].upper(): float(score.item())
            for score, idx in zip(top5.values, top5.indices)
        }

        summary = f"Tokens nhận diện: {len(words)} | Xử lý bằng: {model.device.type.upper()}"

        return img_annotated, label, f"{round(confidence * 100, 2)}%", confidences_dict, summary

    except Exception as e:
        return None, "LỖI HỆ THỐNG", str(e), None, "Thất bại"

# ==============================================================================
# 4. THIẾT KẾ GIAO DIỆN WEB (UI) BẰNG GRADIO
# ==============================================================================
with gr.Blocks(title="LayoutLMv3 Document Classification") as demo:
    gr.Markdown("""
    # Hệ Thống Phân Loại Tài Liệu Đa Phương Thức LayoutLMv3
    Tích hợp **PaddleOCR** và **LayoutLMv3** để tự động phân loại các loại văn bản số hóa.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(label=" Tải lên ảnh tài liệu", type="numpy")
            btn_submit = gr.Button(" Phân Loại Tài Liệu", variant="primary", size="lg")
            annotated_img = gr.Image(label=" Không gian văn bản (OCR Bounding Boxes)", interactive=False)

        with gr.Column(scale=1):
            output_label = gr.Textbox(label="Kết Quả Dự Đoán Chính", text_align="center")
            output_conf = gr.Textbox(label="Độ Tin Cậy")
            output_top5 = gr.Label(num_top_classes=5, label="Phân bố xác suất Top 5")
            output_summary = gr.Textbox(label="System Metadata")

    btn_submit.click(
        fn=classify_document,
        inputs=input_img,
        outputs=[annotated_img, output_label, output_conf, output_top5, output_summary]
    )

# ==============================================================================
# 5. MỞ CỔNG TRUY CẬP (LAN + PUBLIC qua ngrok)
# ==============================================================================
PORT = 7860

if NGROK_AUTH_TOKEN and NGROK_AUTH_TOKEN != "DAN_TOKEN_NGROK_CUA_BAN_VAO_DAY":
    try:
        from pyngrok import ngrok
        ngrok.set_auth_token(NGROK_AUTH_TOKEN)
        public_url = ngrok.connect(PORT, "http")
        print(f" PUBLIC URL (truy cập từ mạng bất kỳ): {public_url}")
    except Exception as e:
        print(f" Không tạo được ngrok tunnel: {e}")
        print("   Vẫn chạy bình thường, chỉ truy cập được trong mạng LAN.")
else:
    print(" Chưa cấu hình NGROK_AUTH_TOKEN -> chỉ truy cập được qua LAN.")
    print("   Lấy token tại: https://dashboard.ngrok.com/get-started/your-authtoken")

print(f" Local:  http://localhost:{PORT}")
print(f" LAN:    http://<IP_LAN_cua_may_ban>:{PORT}  (xem bằng lệnh ipconfig)")

demo.launch(
    server_name="0.0.0.0",
    server_port=PORT,
    share=False,   # tắt share của Gradio vì dùng ngrok thay thế
    debug=False,
    theme=gr.themes.Soft(),
)