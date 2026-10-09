# Kiểm tra bản bàn giao — 09/10/2026

## Kiểm tra trên GitHub Actions

Run [Initialize coursework artifacts #2](https://github.com/dathuynh1108/CO5085-deep-learning/actions/runs/37900439749) đã hoàn tất thành công:

- Checksum của 36 file code, test, config, LaTeX và trang giới thiệu khớp với gói ZIP đã kiểm thử.
- `python -m compileall -q common E1 E2 E3 scripts`: không lỗi cú pháp.
- `python -m pytest -q`: **37 passed in 2.99s**.
- Exporter xuất 20 cấu hình, **0 lượt train/test**. Các chỉ số chưa đo để `null` hoặc `--`.
- XeLaTeX biên dịch hai lượt, tạo PDF **15 trang A4**. Đã tải bản PDF từ artifact và kiểm tra trang bìa sau khi chuẩn hóa logo sang RGB trên nền trắng. Log cuối không có cảnh báo overfull box hoặc ký tự thiếu; còn một underfull box ở URL tài liệu tham khảo và cảnh báo microtype không ảnh hưởng nội dung.
- `scripts/build_site.py` dựng được trang, PDF và JSON trong cùng thư mục phát hành.
- Đề E1–E3 tải lại từ bản cố định trong repo tham khảo, được kiểm tra SHA256 trùng file giảng viên đã cung cấp. Không lấy code hoặc kết quả bài mẫu.
- PDF, bảng sinh tự động và đề gốc đã được commit lên nhánh `main` trong commit `dfe609850b3b15ddd23fabffafffa9c81016da6f`.

Môi trường CI: Ubuntu 24.04, Python 3.12.15, torch 2.10.0+cpu, torchvision 0.25.0+cpu. Các test dùng tensor hoặc JSON/checkpoint fixture, không chạy huấn luyện với dataset thật. Workflow khởi tạo đã được bỏ sau khi hoàn tất; repo giữ các workflow CPU unit tests, Build report và Pages.

## Kiểm tra bản ZIP trước khi đưa lên GitHub

Đã chạy lại đủ 37 test trên CPU và kiểm tra cú pháp. PDF đã được render để xem bìa, mục lục, bảng và công thức. Trang giới thiệu đã được xem bằng Chromium ở 1440×1000 và 390×844: không tràn ngang, không JavaScript error. Kiểm tra offline này không phải bằng chứng site đã deploy; trạng thái Pages nằm trong tab Actions.

## Chưa thực hiện

Chưa tải Fashion-MNIST, chạy training, đánh giá dữ liệu thật, đo GPU throughput hoặc xác nhận một lượt train end-to-end. Không có checkpoint, log train, accuracy hay hình thực nghiệm nào trong bản bàn giao. Các fixture trong test chỉ kiểm tra contract phần mềm và không được xuất vào kết quả báo cáo.

Chưa đọc được nội dung hai đề A1/A2; chưa có hiện thực cho chúng. E1–E3 cũng chưa đạt trạng thái bản nộp cuối cùng vì vẫn thiếu thực nghiệm và phần nhận xét của người học.
