# layoutlmv3-document-classification
Phân loại tài liệu trên dataset RVL-CDIP với kiến trúc LayoutLMv3

Cấu trúc thư mục:
-Web: Demo web với gradio
-Document_AI: Chứa file cài đặt, training, test của LayoutLMv3.
-Baseline: Chứa file training 2 model baseline: ResNet và BERT (folder code chứa code tiền xử lý cho 2 model và file "baseline_datasets" để đọc dữ liệu và cấp dữ liệu cho model khi train)

Lưu ý: Code chạy trên gg colab, demo chạy trên local

Dataset: 40k ảnh lấy từ tập RVL-CDIP (đã đảm bảo lấy riêng - không trộn lẫn từ 3 tập train/val/test gốc và lấy với tỉ lệ cân bằng như tập gốc).

Link gg driver chứa dataset và file json đã trích xuất ocr (paddle ocr) của LayoutLMv3:
https://drive.google.com/drive/folders/1r8TTA_ap3mlKE3OFXvh5Tm2lLmgDun0K?usp=sharing

Link gg driver chứa checkpoint lưu trọng số model LayoutLMv3 được finetune tốt nhất (checkpoint 11976):
https://drive.google.com/drive/folders/1QcUWOFiw8a8W_fnWsU513PZY21Izjlwk?usp=sharing

Link gg driver chứa checkpoint lưu trọng số model baseline:
https://drive.google.com/drive/folders/1BLf3fkvp8LhbCM_Zbffi-hIfRGvcPuP3?usp=sharing

Link gg driver chứa toàn bộ dữ liệu đã được tiền xử lý của model baseline:
https://drive.google.com/drive/folders/15JGqBqcS_Jmwgo8kf7Uig0Wk2-CNjSAa?usp=sharing


Hướng dẫn chạy demo trên local:
1. Clone repo về máy
2. Tải folder checkpoint về máy từ link gg driver ở trên
3. Sửa lại file web.py phù hợp (token ngrok, đường dẫn tới checkpoint)
4. Tải các thư viện từ file requirements.txt
5. Chạy file web.py