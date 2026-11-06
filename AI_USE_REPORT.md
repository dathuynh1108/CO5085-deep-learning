# Khai báo sử dụng AI

**Sinh viên** Huỳnh Thành Đạt · 2570161
**Học phần** CO5085 · Học kỳ 261

AI được sử dụng để tra cứu tài liệu và API của PyTorch, torchvision, hỗ trợ hiện thực và kiểm tra mã, đối chiếu số liệu thực nghiệm, và biên tập báo cáo. Phạm vi bài tập, yêu cầu thí nghiệm và định hướng phân đoạn hoặc phát hiện đối tượng cho A2 do sinh viên cung cấp. Các lựa chọn cụ thể về paper, dữ liệu và phép đối chứng A2 được phát triển với sự hỗ trợ của AI.

| Phần | Nội dung có hỗ trợ |
| --- | --- |
| E1–E3 | Tra cứu API mô hình, hàm mất mát và checkpoint. Hỗ trợ hiện thực và rà soát vòng lặp huấn luyện, đối chiếu attention tự viết với PyTorch và thiết kế các đối chứng cho LSTM. Kiểm tra cách diễn giải các phép đo thứ tự quét và gradient. |
| A1 | Hỗ trợ tra cứu API trọng số tiền huấn luyện, đóng băng tầng và lập lịch tốc độ học của PyTorch. Rà soát kết quả của thiết lập 20 epoch, hỗ trợ thiết kế và chạy đối chứng Swin-T có warmup, kiểm tra số liệu theo seed và biên tập phần thảo luận. |
| A2 | Tra cứu bài PIDNet và tài liệu Oxford-IIIT Pet, đề xuất đối chứng trộn đặc trưng theo biên, hỗ trợ viết pipeline PyTorch, vận hành thực nghiệm GPU và kiểm tra kết quả. Hỗ trợ soạn bản thảo phân tích paper, phân tích lỗi và trình bày bảng, hình, video. |
| Tổ chức bài nộp | Hỗ trợ chuyển quy trình thực nghiệm sang Jupyter, kiểm tra việc lưu cấu hình, seed và log, biên dịch LaTeX và cập nhật GitHub Pages. |

Các chỉ số được tính từ log, checkpoint và dự đoán của các lượt chạy. Thông tin về tập dữ liệu, cấu hình, seed và mã nguồn được lưu cùng kết quả để đối chiếu. Các số liệu trích từ PIDNet được ghi nguồn riêng trong báo cáo A2.

## Tài liệu đối chiếu

- [Đề bài CO5085](assignments/)
- [Tài liệu PyTorch](https://docs.pytorch.org/docs/stable/) và [mô hình torchvision](https://docs.pytorch.org/vision/stable/models.html)
- [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) và [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)
- [PIDNet, IEEE/CVF CVPR 2023](https://doi.org/10.1109/CVPR52729.2023.01871) và [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/)
