# Cấu trúc repository

Năm notebook trong `notebooks/` trình bày quy trình thực nghiệm của E1, E2, E3, A1 và A2. Mã mô hình, dữ liệu và các phép đo dùng chung nằm trong `src/`. Cấu hình nằm trong `configs/`.

E1–E3 dùng chung `reports/exercises/`. A1 và A2 có thư mục báo cáo riêng `reports/assignment1/` và `reports/assignment2/`. Mỗi thư mục có LaTeX, PDF, hình và số liệu theo lượt. Video A2 nằm cạnh PDF. Lệnh `python scripts/build_site.py` dựng `build/site/` từ `site/` với các URL PDF `reports/exercises.pdf`, `reports/assignment1.pdf` và `reports/assignment2.pdf`.

Notebook mặc định ghi kết quả trong `runs/notebooks/`. Các tệp nhẹ để đối chiếu bảng và hình nằm cùng từng báo cáo. Checkpoint và dữ liệu gốc được lưu cục bộ. Các tệp hỗ trợ biên dịch và kiểm tra ở `scripts/` và `tests/`.

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

Kho CO5177-PFDAV giữ hồ sơ tái sử dụng kết quả A1 trước đây. Notebook Image hiện tại của CO5177 huấn luyện riêng sau khi loại ảnh CIFAR-100 trùng pixel. Hai bộ kết quả có cách chia tập và hồ sơ thực nghiệm riêng.
