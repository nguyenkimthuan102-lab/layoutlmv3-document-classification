# layoutlmv3-document-classification
Phân loại tài liệu trên dataset RVL-CDIP với kiến trúc LayoutLMv3

Cấu trúc thư mục:
-Web: Demo web với gradio
-Document_AI: Chứa file cài đặt, training, test.

Dataset: 40k ảnh lấy từ tập RVL-CDIP (đã đảm bảo lấy riêng - không trộn lẫn từ 3 tập train/val/test gốc và lấy với tỉ lệ cân bằng như tập gốc).
Link gg driver chứa dataset và file json đã trích xuất ocr (paddle ocr):
https://drive.google.com/drive/folders/1r8TTA_ap3mlKE3OFXvh5Tm2lLmgDun0K?usp=sharing

Link gg driver chứa checkpoint lưu trọng số model được finetune tốt nhất (checkpoint 11976):
https://drive.google.com/drive/folders/1QcUWOFiw8a8W_fnWsU513PZY21Izjlwk?usp=sharing

Hướng dẫn chạy demo trên local:
1. Clone repo về máy
2. Tải folder checkpoint về máy từ link gg driver ở trên
3. Sửa lại file web.py phù hợp (token ngrok, đường dẫn tới checkpoint)
4. Tải các thư viện từ file requirements.txt
5. Chạy file web.py