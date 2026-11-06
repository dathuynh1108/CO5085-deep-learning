# A2 · Phân đoạn ảnh với phép trộn đặc trưng theo biên

Hai biến thể dùng bộ mã hóa ResNet18 và cùng một hàm mất mát phụ cho đầu dự đoán biên. Bản cố định trộn đặc trưng chi tiết và ngữ cảnh theo tỷ lệ 50/50. Bản theo biên lấy trọng số trộn từ sigmoid của logit biên. Phép đối chứng khảo sát cơ chế Bag trong PIDNet trên trimap của Oxford-IIIT Pet.

- [PIDNet, IEEE/CVF CVPR 2023, tr. 19529–19539](https://doi.org/10.1109/CVPR52729.2023.01871)
- [Oxford-IIIT Pet, CC BY-SA 4.0](https://www.robots.ox.ac.uk/~vgg/data/pets/)
- [Báo cáo A2](../../reports/assignment2/assignment2.pdf) và [video minh họa](../../reports/assignment2/demo.mp4)
- [Cấu hình thí nghiệm](../../configs/a2_experiments.json) và [kết quả tổng hợp](../../reports/assignment2/generated/summary.json)

## Tái lập thực nghiệm

Lượt đo dùng Python 3.12.15, PyTorch 2.14.1+cu132, torchvision 0.29.1+cu132, NumPy 2.5.4, Pillow 13.0.0, CUDA 13.2 và RTX 3090. Hai biến thể dùng BF16 autocast trên GPU. Số liệu từng hạt giống và thông tin môi trường nằm trong `reports/assignment2/generated/evidence/`.

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

`A2.run` lưu vào `runs/assignment2/`. Mỗi lượt hoàn tất có checkpoint tốt nhất, lịch sử theo epoch, định danh nguồn và checksum. Huấn luyện bỏ qua các lượt đã hoàn tất có cùng định danh. Với lượt dở dang, lưu riêng thư mục của lượt đó trước khi chạy lại từ hạt giống ban đầu. Pipeline dùng checkpoint xác thực, khóa đủ sáu lượt rồi đánh giá test. Thay đổi cấu hình hoặc cách chuẩn bị dữ liệu cần một thư mục cache và thư mục lượt chạy riêng.

Các tệp thử nghiệm nhẹ và hình được lưu cùng báo cáo để biên dịch PDF và xây dựng Pages. Dữ liệu thô, cache và checkpoint nằm ngoài Git. Hình và video có sử dụng ảnh, trimap từ Oxford-IIIT Pet theo CC BY-SA 4.0.
