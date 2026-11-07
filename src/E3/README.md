# E3 — LSTM/GRU trên ảnh

Mỗi hàng hoặc patch 4×4 là một bước thời gian. Mạng hồi quy không xử lý câu và không nhận đặc trưng pretrained.

```bash
python -m scripts.run_suite --exercise E3 --group required --seeds 42 43 44 --device cuda
```

Bài nộp gồm bốn cấu hình LSTM/GRU đọc theo hàng hoặc patch 4×4, cùng hai đối chứng MLP/CNN từ E1. `diagnose.py` khai triển phương trình PyTorch trên validation để tính gradient theo bước và so sánh thứ tự gốc, đảo ngược và hoán vị. Phương trình và kết quả được trình bày trong phần E3 của báo cáo chung.
