# Báo cáo sử dụng AI và nguồn tham khảo

Báo cáo sử dụng AI được đính kèm tại [AI_USE_REPORT.md](../AI_USE_REPORT.md),
bao gồm hỗ trợ triển khai, kiểm thử, thiết lập thí nghiệm, vận hành GPU và biên soạn báo cáo.

Yêu cầu bài tập được đối chiếu với các PDF gốc trong assignments/.
Code sử dụng tài liệu PyTorch/torchvision; dữ liệu từ Fashion-MNIST của Zalando
và CIFAR-100 của Toronto. Nguồn và tình trạng license được ghi trong từng báo cáo.

Kết quả E1–E3 và A1 được đo trong lượt chạy riêng, gắn với cấu hình, seed,
source/data hashes và checkpoint. Validation chọn checkpoint; test được đánh giá
sau khi cố định protocol. Log thực thi và checksum sao lưu nằm trong
[TRAINING_EXECUTION.md](TRAINING_EXECUTION.md) và docs/training-evidence/.
