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

E3 đối chiếu bốn cấu hình hồi tiếp, gồm LSTM và GRU trên chuỗi hàng và chuỗi ô ảnh 4×4. Hai đối chứng MLP và CNN được dùng lại từ E1. Phép đảo chuỗi và hoán vị cố định trên tập validation khảo sát độ nhạy của checkpoint với thứ tự quét.

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

Ảnh gốc CIFAR-100 có kích thước 32×32. Nội suy lên 96×96 giữ giới hạn chi tiết của ảnh gốc. Lịch 20 epoch cho thấy Swin-T khởi tạo ngẫu nhiên học kém. Đối chứng bổ sung cho hai bản Swin-T dùng chung lịch 40 epoch và cùng siêu tham số, nhằm đo ảnh hưởng của cách khởi tạo khi bản học từ đầu đã cải thiện.

## Đối chiếu đề A2

Nguồn là `assignments/assignment2-vne.pdf`. Chủ đề là phân đoạn ngữ nghĩa trên Oxford-IIIT Pet, dựa trên cơ chế trộn theo biên trong PIDNet.

| Tiêu chí | Sản phẩm |
|---|---|
| Chủ đề khác A1 và paper liên quan | Phân đoạn theo trimap. Xu, Xiong và Bhattacharyya, PIDNet, IEEE/CVF CVPR 2023, tr. 19529–19539, DOI 10.1109/CVPR52729.2023.01871. |
| Tập công khai, EDA và thiết lập | Oxford-IIIT Pet, CC BY-SA 4.0. Chia 3.310/370/3.669 ảnh. Phân tích tỷ lệ nhãn và phân bố dải chuyển tiếp. |
| Hai biến thể huấn luyện và đánh giá | Trộn cố định 50/50 và trộn theo biên, giữ bộ mã hóa ResNet18 và giám sát biên. Ba seed mỗi biến thể, 12 epoch, mIoU, macro-Dice và IoU theo vùng. |
| Phân tích paper và đối chiếu cài đặt | Báo cáo A2 phân tích mạng hai nhánh, phép tương tự PID, ba nhánh, các mô-đun, hàm mục tiêu và thí nghiệm. Bảng đối chiếu mức thay đổi mIoU với các phép ablation công bố. |
| Notebook, PDF, trang A2 và video | `notebooks/A2.ipynb`, `reports/assignment2/` và `site/a2.html`. Báo cáo có trích dẫn, DOI và liên kết toàn văn. |

Mốc trong đề là 23:59 GMT+7 ngày 02/11/2026 cho A2.1 và ngày 16/11/2026 cho A2.2.

## Trạng thái thực nghiệm

Báo cáo E1–E3 hiện trình bày 39 lượt từ 13 cấu hình theo đề, ba seed và 30 epoch trên Fashion-MNIST 54.000/6.000/10.000 ảnh. Các biến thể LSTM mở rộng trong lượt lịch sử không nằm trong phạm vi hiện tại. A1 hoàn tất lượt notebook mới gồm 24 lượt × 20 epoch và sáu lượt Swin-T × 40 epoch, tổng cộng 720 epoch; kết quả chính đã được đối chiếu với checkpoint và dự đoán test. A2 dùng sáu lượt notebook × 12 epoch trên RTX 5090, với đủ 3.310/370/3.669 ảnh; bằng chứng chính nằm ở `reports/assignment2/notebook-evidence/`. Các lượt RTX 3090 trước đây được giữ riêng trong `reports/assignment2/generated/`. Các lượt chọn checkpoint bằng validation trước khi đánh giá test. Năm notebook là điểm bắt đầu, với cấu hình, seed và lịch sử thực thi được lưu cùng từng báo cáo.

## Mốc trong đề

Khung GitHub Pages: 19/10/2026. E1 và E2: 02/11/2026. E3: 09/11/2026. Hạn ghi trong đề là 23:59 GMT+7; khi có thông báo cập nhật từ giảng viên thì đối chiếu lại LMS. Không tự tạo tag “đã nộp” trước khi có kết quả.

## Đối chiếu bổ sung ngày 07/11/2026

- E1–E3 có đủ yêu cầu thuật toán và thực nghiệm trong 39 lượt hiện tại. Đề không quy định phần sáng tạo 10% cho các bài này.
- A1 có đủ 24 lượt cơ bản và sáu lượt đối chứng 40 epoch. Số worker của cả 30 lượt là bốn; báo cáo đã sửa thống nhất với metadata. Trang A1 có notebook chính và link thư mục mã nguồn A1 theo mục 7.1.2 của đề.
- A2 yêu cầu pipeline chạy được, ít nhất hai phương pháp hoặc biến thể, phân tích paper và liên hệ với cài đặt; không bắt buộc tái hiện toàn bộ PIDNet. Hai biến thể trộn và phần phân tích paper có bằng chứng tương ứng.
- Mục 1.6 của đề E1–E3 chỉ cho phép AI hỗ trợ, không nộp sản phẩm do AI tạo phần lớn và yêu cầu hiểu, bảo vệ được code cùng số liệu. Kiểm tra kỹ thuật hoặc thay đổi câu chữ khai báo không xác nhận được điều kiện này.
- Theo PDF, A2.1 hạn ngày 02/11/2026; commit A2 đầu tiên trong kho mã ngày 06/11/2026. Chưa có thông báo gia hạn được cung cấp. Người học cho biết LMS chưa mở submission và yêu cầu tiếp tục push; lịch sử commit được giữ nguyên.
