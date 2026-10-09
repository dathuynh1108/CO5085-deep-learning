# Thiết kế so sánh

Tất cả dùng Fashion-MNIST, cùng split seed 2026 và normalization tính trên train. Seed mô hình là 42/43/44. AdamW, batch 256, weight decay 1e-4, 30 epoch, cosine LR đến 1% LR đầu. E1/E3 dùng LR 1e-3; E2 dùng 3e-4. So sánh giữa họ mô hình vì thế là so sánh dưới protocol đã công bố, không chứng minh mô hình nào luôn tối ưu. Có thể model còn cần thêm tuning; không tuning bằng test.

Softmax và MLP nhận vector 784. CNN học trực tiếp ảnh 1×28×28. E2 có D=64, 4 head, 2 block, FFN=128, learned positions và mean pooling. E3 dùng hidden=128, một tầng một chiều, không encoder pretrained. Cổng trong PyTorch có hai bias cộng lại; chrono chỉ thay `bias_ih`, giữ `bias_hh=0`.

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

## Mở rộng LSTM theo hàng

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

Rows và patches ở đây chỉ dài 28 và 49 bước. Không suy ra kết luận về sequence dài hàng nghìn bước; input gradient nhỏ cũng có thể phản ánh pixel nền ít thông tin. Bổ sung gradient-state, singular values và kiểm tra thứ tự nhằm hỗ trợ diễn giải, không phải bằng chứng duy nhất.
