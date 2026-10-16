# Cấu trúc repository

Mã Python nằm trong `src/`: `common/` dùng chung, `E1/`, `E2/`, `E3/` cho bài tập, `A1/` cho bài tập lớn. Cài `python -m pip install --no-deps -e .` để dùng các module từ checkout. Config vẫn nằm trong `configs/`.

Hai báo cáo độc lập ở `reports/exercises/` và `reports/assignment1/`; mỗi thư mục có LaTeX, PDF, assets và generated evidence. Landing source ở `site/`; `python scripts/build_site.py` tạo `build/site/` và giữ URL PDF công khai cũ `reports/exercises.pdf`, `reports/assignment1.pdf`.

Artifact cục bộ: `runs/exercises/`, `runs/assignment1/`, `data/`, `artifacts/logs/`. Không đưa checkpoint và raw data vào Git. `scripts/` giữ các CLI, `tests/` kiểm tra mô hình/protocol/delivery, `docs/training-evidence/` giữ provenance.

## Kết quả đã đo trước refactor

Commit `1250e569c6ce055391b5670e2f9a31faedb44b20` chứa mã nguồn thực sự dùng cho 60 lượt E1–E3 và 24 lượt A1. `source-snapshot.json`, backup manifest và JSON/CSV thực nghiệm giữ nguyên byte và chữ ký cũ. `training-evidence/layout-migration.json` ánh xạ đường dẫn trước/sau.

`python -m scripts.verify_archive --checkpoints` kiểm tra Git blobs của nguồn đã đo và toàn bộ 991 tệp sao lưu ở vị trí mới. Không có checkpoint mới hoặc lượt train thêm.

Xuất lại số liệu cũ:

```bash
python -m scripts.export_report
python -m A1.export_report --archive
python -m scripts.write_findings --archive
python -m scripts.build_report
python -m scripts.build_report --report-root reports/assignment1 --jobname assignment1
```

`--archive` chỉ dành cho đọc bằng chứng của commit cố định, kiểm tra snapshot và từng source blob. Training/resume không dùng cơ chế này. Mã sau refactor có source identity mới: không resume checkpoint cũ bằng code mới. Muốn tái lập đúng nguồn cũ, checkout commit trên vào thư mục khác; muốn chạy mã hiện tại, chọn output root mới.

## Dùng lại cho CO5177

Phần Image phù hợp với CIFAR-100 của A1; E1–E3 không phù hợp làm dataset chính vì CO5177 loại Fashion-MNIST. Code/data protocol và kết quả A1 được chuyển có manifest nguồn sang repository `CO5177-PFDAV`. Đó là cùng các lượt chạy, không phải một phép lặp độc lập. Báo cáo Image có phạm vi và yêu cầu riêng, gồm giải thích ảnh gốc 32px và input 96px thấp hơn mức 128px được khuyến nghị.
