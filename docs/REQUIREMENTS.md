# Đối chiếu đề E1–E3

Nguồn: `exercise-vne.pdf`, 6 trang, học kỳ 261. Số trang dưới đây tính cả trang bìa. Không suy diễn yêu cầu A1/A2 từ đề E1–E3.

| Đề | Cách hiện thực | Minh chứng cần sau khi train |
|---|---|---|
| Trang 3: cùng tập ảnh nhỏ | Fashion-MNIST; split và normalization ở `common/data.py` | `run.json`, split JSON và file indices |
| Trang 3: tự viết training loop | `common/engine.py`: forward, CE, backward, clip, optimizer.step | `history.csv`, checkpoint và log chạy |
| Trang 3: một PDF chung | `reports/report.tex` → `reports/exercises.pdf` | PDF đã cập nhật bảng và nhận xét |
| Trang 3: Pages có link repo/PDF | `site/`, `scripts/build_site.py`, workflow Pages | URL Pages hoạt động sau khi publish |
| Trang 4: E1 softmax | `E1/models.py::SoftmaxClassifier` | Accuracy, CE, params, lỗi phân loại |
| Trang 4: E1 MLP | `MLPClassifier`, flatten → 256 → 128 → 10 | Như trên |
| Trang 4: E1 CNN | `SmallCNN`, hai conv blocks và head | Như trên |
| Trang 4: so sánh E1 | 3 preset E1; exporter tự tổng hợp | Loss/accuracy, confusion, mẫu sai |
| Trang 5: MSA tự viết | `E2/models.py::ManualSelfAttention` | Test forward và gradient; kết quả train |
| Trang 5: bản PyTorch | `TorchSelfAttention`; copy cùng khởi tạo | So sánh cùng tokenizer, seed và config |
| Trang 5: ít nhất 2 tokenizer | Có 3: hàng, patch 4×4, CNN-stem | Accuracy, params, thời gian |
| Trang 5: MSA trong classifier | 2 encoder blocks, pre-LN, FFN, position, mean pool | 6 preset E2 |
| Trang 6: cả LSTM và GRU | Native single-layer recurrent models | 4 preset E3 bắt buộc |
| Trang 6: ảnh thành chuỗi | Hàng: T=28,D=28; patch: T=49,D=16 | Giải thích thứ tự quét trong report |
| Trang 6: baseline E1 | Dùng lại run MLP/CNN cùng split | Bảng E3 chứa hai baseline này |
| Trang 6: phụ thuộc chuỗi | Phân tích thứ tự không gian; diagnostic validation | Gradient, đảo/tráo thứ tự, giới hạn diễn giải |
| Trang 6: tái lập | Config, seed, source hash, dataset hash, checkpoint RNG | Lưu toàn bộ artifact của từng seed |

Phần mở rộng E3 là thí nghiệm thêm, không thay thế các mục bắt buộc. Có baseline, thay đổi từng thành phần và hai ablation của tổ hợp. Hiệu quả chỉ được kết luận sau khi chạy.

## Phần chưa thể hoàn tất

Chưa chạy huấn luyện, đánh giá hay đo thời gian trên GPU. Vì vậy các tiêu chí về **số liệu và phân tích kết quả thật** còn chờ, dù code xuất artifact đã có. Chưa có trang Pages live.

Hai tên `assignment/assignment1-vne.pdf` và `assignment/assignment2-vne.pdf` được nhìn thấy trong cây repo tham khảo, nhưng chưa đọc được nội dung. Hai file này không có trong bộ attachment nhận được. A1/A2 chưa có code hoặc báo cáo; cần đề thật trước khi chọn mô hình, dataset, giới hạn pretrained và cách đánh giá.

## Mốc trong đề

Khung GitHub Pages: 19/10/2026. E1 và E2: 02/11/2026. E3: 09/11/2026. Hạn ghi trong đề là 23:59 GMT+7; khi có thông báo cập nhật từ giảng viên thì đối chiếu lại LMS. Không tự tạo tag “đã nộp” trước khi có kết quả.
