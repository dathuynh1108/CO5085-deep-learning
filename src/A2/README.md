# A2 · Phân đoạn ảnh với phép trộn đặc trưng theo biên

Hai biến thể dùng bộ mã hóa ResNet18 và cùng một hàm mất mát phụ cho đầu dự đoán biên. Bản cố định trộn đặc trưng chi tiết và ngữ cảnh theo tỷ lệ 50/50. Bản theo biên lấy trọng số trộn từ sigmoid của logit biên. Phép đối chứng khảo sát cơ chế Bag trong PIDNet trên trimap của Oxford-IIIT Pet.

- [Notebook thực nghiệm A2](../../notebooks/A2.ipynb) và [mở trên Colab](https://colab.research.google.com/github/dathuynh1108/CO5085-deep-learning/blob/main/notebooks/A2.ipynb)
- [PIDNet, IEEE/CVF CVPR 2023, tr. 19529–19539](https://doi.org/10.1109/CVPR52729.2023.01871)
- [Oxford-IIIT Pet, CC BY-SA 4.0](https://www.robots.ox.ac.uk/~vgg/data/pets/)
- [Báo cáo A2](../../reports/assignment2/assignment2.pdf) và [video minh họa](../../reports/assignment2/demo.mp4)
- [Cấu hình thí nghiệm](../../configs/a2_experiments.json) và [kết quả tổng hợp](../../reports/assignment2/generated/summary.json)

## Tái lập bằng notebook

Notebook nạp Oxford-IIIT Pet, khảo sát phân bố nhãn và trimap, thực thi vòng lặp huấn luyện và validation, rồi đánh giá trên tập test sau khi khóa checkpoint. Kết quả gồm mIoU, macro-Dice, IoU dải chuyển tiếp, đường cong học, ma trận nhầm lẫn theo pixel và ảnh dự đoán.

Trên máy cục bộ, mở notebook trong kernel Python 3.12 có PyTorch, torchvision và các gói trong `requirements.txt`. Trên Colab, chọn GPU hỗ trợ BF16 như L4 hoặc A100. Chế độ mặc định huấn luyện hai biến thể, mỗi biến thể 12 epoch với ba seed, và lưu bằng chứng tại `runs/notebooks/a2/`. Có thể đặt `CO5085_DEVICE=cpu` để chạy FP32. Chế độ `CO5085_SMOKE=1` kiểm tra luồng thực thi với một seed, một epoch và tập con, lưu riêng tại `runs/notebooks/a2-smoke/`.

Sau khi xem kết quả trên tập validation và khóa checkpoint, nhập `TEST` ở ô xác nhận để đánh giá trên tập test. Với phiên chạy tự động, đặt `CO5085_CONFIRM_TEST=1` trước khi khởi chạy kernel. Notebook kiểm tra biến này sau khi tạo và xác minh `frozen.json`. Có thể đặt `CO5085_OUTPUT_ROOT` để lưu một lượt mới vào thư mục riêng.

## Tái lập bằng dòng lệnh

Lượt đo trong báo cáo dùng Python 3.12.15, PyTorch 2.14.1+cu132, torchvision 0.29.1+cu132, NumPy 2.5.4, Pillow 13.0.0, CUDA 13.2 và RTX 3090. Hai biến thể dùng BF16 autocast trên GPU. Số liệu từng seed và thông tin môi trường nằm trong `reports/assignment2/generated/evidence/`.

Chạy từ thư mục gốc, trong môi trường đã cài PyTorch/torchvision khớp phiên bản CUDA, cùng các gói trong `requirements.txt`:

```bash
python -m pip install --no-deps -e .
python -m A2.data
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m A2.run train --device cuda --workers 6
python -m A2.run freeze
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m A2.run test --device cuda --workers 6 --confirm-test
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m A2.benchmark
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m A2.export_report --device cuda
python -m scripts.build_report --report-root reports/assignment2 --jobname assignment2
```

`A2.data` tải archive chính thức qua torchvision và chuẩn bị cache ảnh, mặt nạ và manifest trong `data/a2_prepared/`. Cần giữ hai archive gốc để đối chiếu checksum. Dùng thư mục cache mới nếu chuẩn bị lại dữ liệu. Giữ môi trường giải mã nhất quán khi đối chiếu SHA-256 tensor. Cache macOS với Pillow 12.3.0 đã cho cùng danh sách ảnh và hash tensor train/validation, nhưng hash tensor test khác cache Linux dùng trong lượt đo. Hai archive gốc khớp SHA-256.

`A2.run` lưu vào `runs/assignment2/`. Mỗi lượt hoàn tất có checkpoint tốt nhất, lịch sử theo epoch, định danh nguồn và checksum. Huấn luyện bỏ qua các lượt đã hoàn tất có cùng định danh. Với lượt dở dang, lưu riêng thư mục của lượt đó trước khi chạy lại từ seed ban đầu. Pipeline dùng checkpoint xác thực, khóa đủ sáu lượt rồi đánh giá test. Thay đổi cấu hình hoặc cách chuẩn bị dữ liệu cần một thư mục cache và thư mục lượt chạy riêng.

Các tệp thử nghiệm nhẹ và hình được lưu cùng báo cáo để biên dịch PDF và xây dựng Pages. Dữ liệu thô, cache và checkpoint nằm ngoài Git. Hình và video có sử dụng ảnh, trimap từ Oxford-IIIT Pet theo CC BY-SA 4.0.
