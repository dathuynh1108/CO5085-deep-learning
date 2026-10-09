# Kiểm tra bản bàn giao — 09/10/2026

## Đã thực hiện trên CPU

- `python -m pytest -q tests`: **37 passed**. Các phép thử dùng tensor hoặc JSON/checkpoint fixture trong thư mục tạm; không train dữ liệu thật. Đã chạy lại đủ 37 test trước khi đưa mã nguồn lên GitHub.
- `python -m compileall -q common E1 E2 E3 scripts`: không có lỗi cú pháp.
- CLI help của train/evaluate/diagnose/publish và `run_suite --list`: hoạt động; danh sách gồm 20 cấu hình.
- Export khi chưa có run: 20 dòng, các accuracy là `null`, không có seed train/test. Test một-seed kiểm tra SD là chưa xác định, không tự đặt bằng 0.
- XeLaTeX chạy hai lượt tạo `reports/exercises.pdf`: 15 trang trong bản ZIP ban đầu. Đã render để xem bìa, mục lục, bảng và công thức; không có cảnh báo overfull box hoặc ký tự thiếu. Có cảnh báo underfull ở một URL tài liệu tham khảo, không mất nội dung.
- Site render offline bằng Chromium ở 1440×1000 và 390×844: không tràn ngang, không JavaScript error. File PDF/JSON đích có trong thư mục build. Kiểm tra này không phải kiểm tra site đã deploy.

Môi trường kiểm thử bản ZIP: Python 3.13.5, torch 2.10.0+cpu, torchvision 0.25.0+cpu, NumPy 2.3.5, Matplotlib 3.10.8, Pillow 12.3.0, pytest 9.0.2. PDF dùng fallback Liberation Serif/Sans vì môi trường này không có bản OpenType của font TeX Gyre mặc định.

## GitHub

Repository đích là `dathuynh1108/CO5085-deep-learning`. Kết quả CI và lần triển khai Pages được ghi trong tab Actions; không coi kiểm tra offline là bằng chứng Pages đang hoạt động. Workflow build báo cáo chỉ biên dịch LaTeX, không huấn luyện mô hình.

## Chưa thực hiện

Chưa tải Fashion-MNIST, chạy training, đánh giá dữ liệu thật, đo GPU throughput hoặc xác nhận một lượt train end-to-end. Không có checkpoint, log train, accuracy hay hình thực nghiệm nào trong bản bàn giao. Các fixture trong test chỉ kiểm tra contract phần mềm và không được xuất vào kết quả báo cáo.

Chưa đọc được nội dung hai đề A1/A2; chưa có hiện thực cho chúng. E1–E3 cũng chưa đạt trạng thái bản nộp cuối cùng vì vẫn thiếu thực nghiệm và phần nhận xét của người học.
