# Thiết kế so sánh

Các notebook `notebooks/E1.ipynb`, `notebooks/E2.ipynb` và `notebooks/E3.ipynb` là điểm bắt đầu của thực nghiệm hiện tại: 13 cấu hình bắt buộc × ba seed, tổng cộng 39 lượt, mỗi lượt 30 epoch. Cả ba bài dùng đầy đủ Fashion-MNIST với 54.000 ảnh train, 6.000 validation và 10.000 test, cùng split seed 2026 và normalization tính trên train. Seed mô hình là 42/43/44. AdamW, batch 256, weight decay 1e-4, 30 epoch, cosine LR đến 1% LR đầu. E1/E3 dùng LR 1e-3; E2 dùng 3e-4. So sánh giữa họ mô hình vì thế là so sánh dưới protocol đã công bố, không chứng minh mô hình nào luôn tối ưu. Có thể model còn cần thêm tuning; không tuning bằng test.

Softmax và MLP nhận vector 784. CNN học trực tiếp ảnh 1×28×28. E2 có D=64, 4 head, 2 block, FFN=128, learned positions và mean pooling. E3 dùng hidden=128, một tầng một chiều, không encoder pretrained. Cổng trong PyTorch có hai vector bias cộng lại. E3 hiện tại gồm LSTM và GRU trên chuỗi hàng hoặc patch 4×4, hai đối chứng MLP/CNN từ E1 và các phép kiểm tra thứ tự chuỗi trên validation.

## Các đối chứng

| So sánh | Cố định | Thay đổi |
|---|---|---|
| E1 | data, split, seed, training budget | softmax / MLP / CNN |
| E2 manual/library | cả tokenizer, tensor khởi tạo, D, head, depth, optimizer | attention backend |
| E2 tokenization | attention architecture, budget | 28 row tokens / 49 patch tokens / 49 CNN tokens |
| E3 LSTM/GRU | representation, hidden=128, budget | loại cell; số tham số khác nhau |
| E3 rows/patches | loại cell, hidden, budget | T=28,D=28 và T=49,D=16 |
| E3/E1 | split, metrics, seed | inductive bias; không cố cân bằng capacity |

E2 xây cả model manual trước rồi copy attention weights sang bản thư viện. Không chỉ reset seed và kỳ vọng hai cách gọi API tự có cùng khởi tạo. Dropout/kernel khác nhau vẫn có thể tạo quỹ đạo train khác; unit equivalence kiểm tra khi dropout tắt. Không dùng kết quả unit test làm thay cho so sánh thực nghiệm.

## Mở rộng LSTM theo hàng trong lượt lịch sử

Các biến thể dưới đây thuộc lượt thực nghiệm trước, được lưu riêng để truy nguyên. Chúng không nằm trong 13 cấu hình bắt buộc, bảng kết quả notebook hiện tại hoặc phạm vi kết luận E3 hiện tại. Không gộp các lượt này vào 39 lượt đang dùng trong báo cáo.

| ID (bỏ prefix `e3_lstm_rows_`) | Trực giao ban đầu | λ trực giao | Bias chrono cố định | Readout |
|---|---|---:|---|---|
| Baseline `e3_lstm_rows` | Không | 0 | Không | Last |
| `orth_init` | Có | 0 | Không | Last |
| `orth_reg` | Có | 0.001 | Không | Last |
| `chrono` | Không | 0 | Có | Last |
| `mean` | Không | 0 | Không | Mean |
| `combined` | Có | 0.001 | Có | Mean |
| `no_reg` | Có | 0 | Có | Mean |
| `no_mean` | Có | 0.001 | Có | Last |

So `orth_init` với baseline để xét khởi tạo; `orth_reg` với `orth_init` để xét penalty; `chrono`/`mean` với baseline cho tác động riêng; `combined` với `no_reg`/`no_mean` cho tác động trong tổ hợp. Không gọi đây là thiết kế factorial đầy đủ: chưa đo mọi tương tác và chưa có tổ hợp bỏ riêng chrono.

Penalty là trung bình theo gate của `||WᵀW−I||²/H`. AdamW vẫn có weight decay trên mọi tham số, kể cả bias và ma trận hồi quy; đó là một lựa chọn công khai và cố định. Weight decay kéo về zero, penalty kéo về trực giao, nên trọng số không được bảo toàn trực giao tuyệt đối. Chrono chỉ áp dụng lúc khởi tạo, không khóa bias trong train.

Chuỗi hàng và patch chỉ dài 28 và 49 bước. Không suy ra kết luận về chuỗi dài hàng nghìn bước; gradient đầu vào nhỏ cũng có thể phản ánh pixel nền ít thông tin. Trong phạm vi E3 hiện tại, gradient trạng thái và các phép đảo/hoán vị thứ tự trên validation hỗ trợ diễn giải độ nhạy với cách quét ảnh.
