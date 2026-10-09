# Bàn giao chạy trên máy GPU

Mục tiêu là chạy đúng các cấu hình đã định, giữ dữ liệu và kết quả có thể kiểm tra lại. Bộ mã ban đầu chưa train. Không lấy accuracy của bài khác hoặc benchmark công khai để điền vào báo cáo.

## Trước lượt chạy chính

Đọc `README.md`, `docs/EXPERIMENTS.md`, `docs/REQUIREMENTS.md`. Tạo venv riêng, cài đúng cặp torch/torchvision và kiểm tra driver. Không thay mô hình bằng model zoo, pretrained backbone, `Trainer`, `Lightning` hay `trainer.fit`.

Không cần chạy một batch thật để “preflight” theo yêu cầu của người học. Unit test thuần tensor không tải dữ liệu đã có và có thể chạy trước. Lượt đầu có thể chạy riêng E1 CNN đủ 30 epoch để kiểm chứng toàn quy trình, nhưng đó phải là một **run chính thức**, không giả làm benchmark toàn bộ. Khi chuyển sang suite, dùng `--resume` để không ghi đè run đã có.

Lệnh chuẩn chạy tuần tự trên một GPU:

```bash
python -m scripts.prepare_data --data-root data
python -m scripts.run_suite --group all --seeds 42 43 44 --device cuda --num-workers 4
```

Windows native gặp vấn đề multiprocessing thì chọn `--num-workers 0` ngay từ đầu; giữ nhất quán khi đo thời gian. `--device cuda:0` chọn thiết bị cụ thể. Không chạy nhiều suite cùng ghi vào một `runs/`. `CUDA_VISIBLE_DEVICES` nên giữ nguyên khi resume.

FP32 được dùng cho cả ba họ mô hình. Không bật AMP, TF32 hoặc compile ở một nhóm rồi so tốc độ với nhóm còn lại. Nếu đổi chế độ để thử tăng tốc, dùng nghiên cứu riêng và ghi rõ khác biệt. `--deterministic` là lựa chọn từ đầu, không bật giữa chừng; code sẽ báo lỗi nếu toán tử không hỗ trợ chế độ đã chọn.

## Khi gián đoạn

Chạy lại cùng lệnh với `--resume`. Model, AdamW, cosine scheduler, shuffle generator và Python/NumPy/PyTorch RNG được khôi phục tại cuối epoch hoàn tất gần nhất. Epoch đang chạy dở sẽ chạy lại. Chưa hỗ trợ resume giữa batch.

`.train.lock` ngăn hai tiến trình ghi cùng run. Sau mất điện, kiểm tra tiến trình thực tế trước khi xóa khóa cũ. Script không tự coi mọi khóa là rác. Checkpoint `last.pt` chứa trạng thái Python/NumPy và được load với `weights_only=False`: chỉ dùng checkpoint tự tạo, không tải file lạ về để resume.

Đổi code, data, seed, hyperparameter hoặc môi trường torch/CUDA sẽ bị từ chối resume. Cần ghi rõ lý do, dùng output root mới và chạy lại đủ các đối chứng; không lách bằng sửa signature. Với cùng phần mềm/phần cứng, state RNG được phục hồi; không hứa bitwise reproducibility giữa GPU hoặc kernel khác nhau.

## Đánh giá sau khi chốt thiết kế

Mỗi epoch chỉ dùng train và validation. Chọn best bằng validation accuracy, hòa thì CE nhỏ hơn. Xem validation để phát hiện underfit/overfit và chọn hướng nghiên cứu; mọi thay đổi sau đó phải được ghi lại trước khi mở test.

```bash
python -m scripts.run_suite --group all --seeds 42 43 44 --phase test --confirm-test --device cuda
python -m E3.diagnose --run runs/e3_lstm_rows/seed_42 --device cuda
python -m E3.diagnose --run runs/e3_lstm_rows_combined/seed_42 --device cuda
```

Test đã đo được tái sử dụng nếu checkpoint không đổi. Không thử nhiều hyperparameter dựa vào điểm test. Diagnostic không train: dùng 256 validation images có chỉ số gốc nhỏ nhất, đo cùng subset cho các cách quét. Muốn mở rộng số mẫu thì thay `--max-samples` đồng nhất cho các đối chứng.

Mỗi run phải có `run.json`, `history.csv`, `best.pt`, `last.pt`, `metrics.json`. Sau test có thêm `evaluation_test.json`, predictions CSV, confusion và tối đa 8 lỗi có confidence cao nhất. Không có lỗi thì không sinh gallery giả.

## Hoàn thiện báo cáo

```bash
python -m scripts.export_report
python -m scripts.build_report
python scripts/build_site.py
```

Exporter kiểm tra cùng split, code/config, seed, checkpoint hash và test dataset. Nó không gộp trial override vào nghiên cứu chuẩn. Mean/std tính theo seed thực có; một seed thì SD chưa xác định. Tốc độ chỉ gộp trong một cấu hình khi môi trường đo giống nhau; vẫn phải đọc môi trường của các cấu hình trước khi so chéo.

Ảnh minh họa lấy từ seed nhỏ nhất, không chọn seed có test tốt nhất. Đường cong đối chiếu dùng seed chung nhỏ nhất. Hình thể hiện một seed không được gọi là trung bình ba seed. Bảng có phạm vi seed thực tế; không ghi “60 runs” nếu mới chạy một phần.

Sau export, người học cần cập nhật các đoạn nhận xét: khoảng cách train/validation; cặp lớp dễ nhầm; accuracy so với params và thời gian; manual vs library; khác biệt tokenizer; kết quả ablation; giới hạn của chuỗi 28/49 bước. Nếu improvement không tốt hơn, giữ kết quả đó và giải thích thận trọng.

Chỉ commit source, config, bảng/figures tổng hợp, PDF và log cần để tái lập. Không commit data/raw hoặc checkpoint nặng. Có thể lưu checkpoint ngoài Git và ghi checksum/đường dẫn được phép chia sẻ. Trước nộp cần mở lại PDF và các link Pages, kiểm tra bảng không bị tràn, không còn kết luận dự kiến viết như sự thật.

A1/A2 nằm ngoài gói hiện tại; không tự điền nội dung hai bài này khi chưa có đề.
