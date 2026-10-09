# E3 — LSTM/GRU trên ảnh

Mỗi hàng hoặc patch 4×4 là một bước thời gian. Mạng hồi quy không xử lý câu và không nhận đặc trưng pretrained.

```bash
python -m scripts.run_suite --exercise E3 --group all --seeds 42 43 44 --device cuda
```

Bốn cấu hình bắt buộc và bảy cấu hình mở rộng dùng chung training loop. `diagnose.py` unroll đúng phương trình PyTorch trên validation để đọc gradient-state thật. Phân tích toán và ablation ở chương mở rộng trong báo cáo; hướng dẫn ở `docs/EXPERIMENTS.md`.
