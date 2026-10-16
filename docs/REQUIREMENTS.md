# Đối chiếu đề E1–E3

Nguồn: `exercise-vne.pdf`, 6 trang, học kỳ 261. Số trang dưới đây tính cả trang bìa. Không suy diễn yêu cầu A1/A2 từ đề E1–E3.

| Đề | Cách hiện thực | Minh chứng cần sau khi train |
|---|---|---|
| Trang 3: cùng tập ảnh nhỏ | Fashion-MNIST; split và normalization ở `src/common/data.py` | `run.json`, split JSON và file indices |
| Trang 3: tự viết training loop | `src/common/engine.py`: forward, CE, backward, clip, optimizer.step | `history.csv`, checkpoint và log chạy |
| Trang 3: một PDF chung | `reports/exercises/report.tex` → `reports/exercises/exercises.pdf` | PDF đã cập nhật bảng và nhận xét |
| Trang 3: Pages có link repo/PDF | `site/`, `scripts/build_site.py`, workflow Pages | URL Pages hoạt động sau khi publish |
| Trang 4: E1 softmax | `src/E1/models.py::SoftmaxClassifier` | Accuracy, CE, params, lỗi phân loại |
| Trang 4: E1 MLP | `MLPClassifier`, flatten → 256 → 128 → 10 | Như trên |
| Trang 4: E1 CNN | `SmallCNN`, hai conv blocks và head | Như trên |
| Trang 4: so sánh E1 | 3 preset E1; exporter tự tổng hợp | Loss/accuracy, confusion, mẫu sai |
| Trang 5: MSA tự viết | `src/E2/models.py::ManualSelfAttention` | Test forward và gradient; kết quả train |
| Trang 5: bản PyTorch | `TorchSelfAttention`; copy cùng khởi tạo | So sánh cùng tokenizer, seed và config |
| Trang 5: ít nhất 2 tokenizer | Có 3: hàng, patch 4×4, CNN-stem | Accuracy, params, thời gian |
| Trang 5: MSA trong classifier | 2 encoder blocks, pre-LN, FFN, position, mean pool | 6 preset E2 |
| Trang 6: cả LSTM và GRU | Native single-layer recurrent models | 4 preset E3 bắt buộc |
| Trang 6: ảnh thành chuỗi | Hàng: T=28,D=28; patch: T=49,D=16 | Giải thích thứ tự quét trong report |
| Trang 6: baseline E1 | Dùng lại run MLP/CNN cùng split | Bảng E3 chứa hai baseline này |
| Trang 6: phụ thuộc chuỗi | Phân tích thứ tự không gian; diagnostic validation | Gradient, đảo/tráo thứ tự, giới hạn diễn giải |
| Trang 6: tái lập | Config, seed, source hash, dataset hash, checkpoint RNG | Lưu toàn bộ artifact của từng seed |

Phần mở rộng E3 là thí nghiệm thêm, không thay thế các mục bắt buộc. Có baseline, thay đổi từng thành phần và hai ablation của tổ hợp. Hiệu quả chỉ được kết luận sau khi chạy.

## Đề A1/A2 được bổ sung ngày 14/10/2026

Nguồn A1: `assignments/assignment1-vne.pdf` (5 trang cả bìa), được người học cung cấp. A1 có code riêng ở `src/A1/`, báo cáo riêng `reports/assignment1/report.tex` → `reports/assignment1/assignment1.pdf`.

| Tiêu chí A1 | Protocol triển khai | Evidence sau chạy |
|---|---|---|
| Tập công khai, khó hơn E1; không MNIST/Fashion-MNIST/CIFAR-10 | CIFAR-100 đầy đủ, 100 lớp; 45.000 train/5.000 validation/10.000 test; split phân tầng seed 2026 | Metadata nguồn/split và chỉ số ảnh; không dùng subset |
| Nêu license, split, lý do chọn | Nguồn tác giả Toronto công khai CIFAR-100; dataset card của uoft-cs ghi license unknown; không tự gán MIT/CC0 và không đưa raw data lên Git | A1 README và chương dữ liệu của báo cáo ghi rõ tình trạng license |
| CNN và Transformer | ResNet18 và Swin-T từ torchvision | Config/source/model fingerprint |
| Scratch vs pretrained cùng split | Cả hai họ có một scratch và ba pretrained IMAGENET1K_V1 | Tám config × ba seed |
| Freeze backbone / một phần / toàn bộ fine-tune | Head-only; layer4+head với ResNet / stage cuối+norm+head với Swin; full | Danh sách tham số trainable, tests freeze và running statistics |
| Accuracy/F1, thời gian, trainable params, lỗi | Checkpoint theo validation, test độc lập sau khóa protocol; BF16 cùng chế độ A1, ảnh resize 96×96 | Summary, curves, confusion, per-class metrics, prediction/errors |
| A1.1 và A1.2 | A1.1: dataset + protocol; A1.2: đủ so sánh và phân tích thật | Report riêng và trang A1 riêng trong bản build Pages |

Giới hạn đã công bố trước chạy: ảnh gốc CIFAR-100 chỉ 32×32, resize không tạo thêm thông tin; 96×96 khác kích thước 224×224 phổ biến của pretrained; ngân sách 20 epoch không chứng minh scratch đã hội tụ. Báo cáo so sánh dưới setting này, không khái quát thành kiến trúc luôn tốt hơn.

A2: đã lưu đề gốc nhưng chưa chọn chủ đề/paper; **chưa triển khai** theo yêu cầu người học. Không tạo report A2 giả hoặc chọn paper thay người học.

## Trạng thái thực nghiệm

Đã hoàn tất60/60 lượt E1–E3, mỗi lượt30 epoch, và24/24 lượt A1, mỗi lượt20 epoch; tất cả đã đánh giá held-out test sau khi cố định protocol. Manifest, source/data/checkpoint hashes và lịch sử sao lưu được ghi trong `docs/TRAINING_EXECUTION.md`, `reports/exercises/generated/` và `reports/assignment1/generated/`.

## Mốc trong đề

Khung GitHub Pages: 19/10/2026. E1 và E2: 02/11/2026. E3: 09/11/2026. Hạn ghi trong đề là 23:59 GMT+7; khi có thông báo cập nhật từ giảng viên thì đối chiếu lại LMS. Không tự tạo tag “đã nộp” trước khi có kết quả.
