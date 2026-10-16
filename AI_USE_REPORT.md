# Báo cáo sử dụng AI

**Sinh viên:** Huỳnh Thành Đạt — 2570161. **Môn:** CO5085, học kỳ 261.

AI được sử dụng trong quá trình thực hiện E1–E3 và A1 để hỗ trợ triển khai, kiểm thử mã nguồn, thiết lập thí nghiệm, phân tích kết quả và biên soạn báo cáo LaTeX.

## Yêu cầu và định hướng đầu vào

Prompt xác định phạm vi thực hiện các exercise đã có mã nguồn, bổ sung A1 với báo cáo riêng, khai báo các thông số thí nghiệm và khảo sát hướng cải thiện E3. Manifold learning là một trong các hướng được đề cập. Các lựa chọn về kiến trúc, optimizer, learning rate, batch size, số epoch và seed được cụ thể hóa trong cấu hình trước khi chạy.

## Phạm vi hỗ trợ

| Nội dung | Công việc có hỗ trợ AI |
|---|---|
| Triển khai theo tài liệu | Chuyển yêu cầu trong prompt thành mã dựa trên tài liệu chính thức PyTorch/torchvision: mô hình, DataLoader, training loop, loss, optimizer và checkpoint. |
| E1–E3 | Rà soát mã hiện có, kiểm tra quy trình train/validation/test, chạy các cấu hình và tổng hợp kết quả theo seed. |
| A1 | Đề xuất CIFAR-100 và hiện thực so sánh ResNet18/Swin-T ở các chế độ scratch, head-only, partial và full fine-tuning. |
| Cải thiện E3 | Gợi ý và triển khai thí nghiệm với khởi tạo trực giao, penalty trực giao, bias cổng nhớ và pooling; xây dựng các đối chứng ablation. |
| Manifold learning | Gợi ý liên hệ giữa ma trận trực giao và Stiefel manifold. Hiện thực trong bài sử dụng penalty trực giao mềm trong không gian tham số thông thường. |
| Kiểm thử | Hỗ trợ viết và kiểm tra các test về attention, gradient, split, freeze backbone, augmentation và khôi phục optimizer/RNG khi resume. |
| Vận hành GPU | Thiết lập môi trường Vast.ai, xây dựng script chạy theo cấu hình, theo dõi log, khôi phục job gián đoạn và kiểm tra checksum khi sao lưu. |
| Báo cáo | Tổ chức LaTeX, sinh bảng và biểu đồ từ kết quả, hỗ trợ phân tích lỗi, trình bày ablation và rà soát bố cục PDF. |

## Đối chiếu kết quả

Đã hoàn tất 60 lượt E1–E3 và 24 lượt A1, với ba seed 42/43/44. Các bảng và biểu đồ gắn với cấu hình, seed, cách chia dữ liệu, source hash và checkpoint của từng lượt chạy. Validation được dùng để chọn checkpoint; test được đánh giá riêng sau khi cố định thiết lập. Log và thông tin tái lập được lưu trong các thư mục kết quả và `docs/TRAINING_EXECUTION.md`.

## Tài liệu sử dụng

- [PyTorch documentation](https://docs.pytorch.org/docs/stable/).
- [Torchvision models and pretrained weights](https://docs.pytorch.org/vision/stable/models.html).
- [CIFAR-100 — trang dữ liệu của tác giả](https://www.cs.toronto.edu/~kriz/cifar.html).
- [Fashion-MNIST — repository của tác giả](https://github.com/zalandoresearch/fashion-mnist).
- Đề bài trong thư mục `assignments/`.

## Tổ chức lại và dùng lại kết quả cho Image

AI hỗ trợ chuyển mã vào `src/`, tách hai báo cáo trong `reports/`, cập nhật packaging/CLI/CI và landing theo phong cách của CO5177. Các hash thực nghiệm gốc và 991 tệp sao lưu được đối chiếu sau khi chuyển. Mã sau refactor có identity mới; đọc kết quả cũ qua cơ chế archive tường minh, không đổi signature để resume.

Theo yêu cầu dùng lại bài phù hợp, AI chuyển A1 sang track Image của CO5177 cùng manifest nguồn, bổ sung notebook khám phá toàn bộ CIFAR-100 và phân tích prediction. Kiểm tra pixel phát hiện ảnh trùng giữa các split; giới hạn này và kiểm tra độ nhạy trên cùng checkpoint được bổ sung vào báo cáo A1. Không chạy thêm training hoặc thuê thêm GPU trong lần tổ chức này.
