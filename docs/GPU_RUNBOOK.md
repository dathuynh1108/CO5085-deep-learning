# Quy trình chạy trên máy GPU

Các notebook trong `notebooks/` là điểm bắt đầu của quy trình thực nghiệm. E1–E3 hiện có 39 lượt hoàn tất từ 13 cấu hình bắt buộc × ba seed × 30 epoch, trên toàn bộ Fashion-MNIST 54.000/6.000/10.000 ảnh. A2 có sáu lượt notebook × 12 epoch trên RTX 5090. A1 hoàn tất 30 lượt notebook, gồm 24 lượt × 20 epoch và sáu đối chứng Swin-T × 40 epoch, tổng cộng 720 epoch. Toàn bộ checkpoint và 300.000 dự đoán test đã được đối chiếu.

Kết quả thực thi được lưu từ máy Vast.ai và các phiên Jupyter tương ứng; chưa xác nhận một phiên Google Colab mới chạy toàn bộ notebook bằng Run all. Phần lệnh bên dưới phục vụ tái lập bằng CLI. Lượt lịch sử 60 lượt E1–E3 và 24 lượt A1 được phân biệt trong [TRAINING_EXECUTION.md](TRAINING_EXECUTION.md).

## Trước lượt chạy chính

Đọc `README.md`, `docs/EXPERIMENTS.md`, `docs/REQUIREMENTS.md`. Tạo venv riêng, cài đúng cặp torch/torchvision, cài `python -m pip install --no-deps -e .` và kiểm tra driver. Với E1–E3, không thay mô hình bằng model zoo hoặc pretrained backbone. Mọi phần dùng training loop trực tiếp, không `Trainer`, `Lightning` hay `trainer.fit`; A1 cho phép torchvision/pretrained theo đề riêng.

Không cần chạy một batch thật để “preflight” theo yêu cầu của người học. Unit test thuần tensor không tải dữ liệu đã có và có thể chạy trước. Lượt đầu có thể chạy riêng E1 CNN đủ 30 epoch để kiểm chứng toàn quy trình, nhưng đó phải là một **run chính thức**, không giả làm benchmark toàn bộ. Khi chuyển sang suite, dùng `--resume` để không ghi đè run đã có.

Lệnh chuẩn chạy tuần tự trên một GPU:

```bash
python -m scripts.prepare_data --data-root data
python -m scripts.run_exercises_isolated --group required --seeds 42 43 44 --device cuda --num-workers 8
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
python -m scripts.run_exercises_isolated --group required --seeds 42 43 44 --phase test --confirm-test --device cuda --num-workers 8
python -m E3.diagnose --run runs/exercises/e3_lstm_rows/seed_42 --device cuda
```

Test đã đo được tái sử dụng nếu checkpoint không đổi. Không thử nhiều hyperparameter dựa vào điểm test. Diagnostic không train: dùng 256 validation images có chỉ số gốc nhỏ nhất, đo cùng subset cho các cách quét. Muốn mở rộng số mẫu thì thay `--max-samples` đồng nhất cho các đối chứng.

Mỗi run phải có `run.json`, `history.csv`, `best.pt`, `last.pt`, `metrics.json`. Sau test có thêm `evaluation_test.json`, predictions CSV, confusion và tối đa 8 lỗi có confidence cao nhất. Không có lỗi thì không sinh gallery giả.

## Hoàn thiện báo cáo

Notebook lưu và tổng hợp bằng chứng của lượt hiện tại. Các lệnh xuất bên dưới dành cho kết quả CLI trong `generated/`; chúng không thay thế việc kiểm tra và đồng bộ số liệu chính trong `notebook-evidence/`.

```bash
python -m scripts.export_report
python -m A1.export_report
python -m scripts.write_findings
python -m scripts.build_report
python -m scripts.build_report --report-root reports/assignment1 --jobname assignment1
python scripts/build_site.py
```

Exporter kiểm tra cùng split, code/config, seed, checkpoint hash và test dataset. Nó không gộp trial override vào nghiên cứu chuẩn. Mean/std tính theo seed thực có; một seed thì SD chưa xác định. Tốc độ chỉ gộp trong một cấu hình khi môi trường đo giống nhau; vẫn phải đọc môi trường của các cấu hình trước khi so chéo.

Ảnh minh họa lấy từ seed nhỏ nhất, không chọn seed có test tốt nhất. Đường cong đối chiếu dùng seed chung nhỏ nhất. Hình thể hiện một seed không được gọi là trung bình ba seed. Bảng ghi phạm vi seed thực tế; 39 lượt E1–E3 hiện tại không bao gồm các biến thể LSTM mở rộng của lượt lịch sử.

Phần phân tích đối chiếu các nội dung: khoảng cách train/validation; cặp lớp dễ nhầm; accuracy so với params và thời gian; manual vs library; khác biệt tokenizer; độ nhạy với thứ tự quét; giới hạn của chuỗi 28/49 bước. Nếu improvement không tốt hơn, giữ kết quả đó và giải thích thận trọng.

Chỉ commit source, config, bảng/figures tổng hợp, PDF và log cần để tái lập. Không commit data/raw hoặc checkpoint nặng. Có thể lưu checkpoint ngoài Git và ghi checksum/đường dẫn được phép chia sẻ. Trước nộp cần mở lại PDF và các link Pages, kiểm tra bảng không bị tràn, không còn kết luận dự kiến viết như sự thật.

A1 được bổ sung theo đề thật trong `assignments/assignment1-vne.pdf`. Code và báo cáo riêng ở `src/A1/`; xem `src/A1/README.md`. A2 có giao thức riêng trên Oxford-IIIT Pet, xem mục cuối tài liệu này.

## Pipeline Vast lịch sử ngày 14/10/2026

Đoạn này mô tả lượt lịch sử 60+24, không phải lệnh chạy notebook hiện tại. `COURSE_PYTHON=/venv/main/bin/python bash scripts/train_coursework_gpu.sh` đã chạy E1–E3 rồi A1 trên một GPU, mỗi suite hoàn tất train trước phase held-out test. Log ở `artifacts/logs/training.log`; shell ghi `coursework.exit` và chỉ ghi `coursework.complete` khi cả report/site đã build thành công. Không sửa source/config đã hash khi suite còn chạy.

Lượt Vast dùng `scripts.run_exercises_isolated` cho E1–E3: một child process cho mỗi config/seed ở cả train và test, cùng entrypoint và resume guards. Cách này tránh tích lũy file descriptor của pinned/persistent DataLoader qua nhiều run trong một process. Không đổi workers, precision, model, split hoặc epoch; lịch sử gián đoạn được ghi trong `docs/TRAINING_EXECUTION.md`.

Sau complete, `scripts/watch_vast_training.py` tải báo cáo, evidence và checkpoint về workspace. `scripts/vast_finalize.py` kiểm tra đủ60+24 run, checkpoint/test signatures, tensor/file hashes, SHA256 bản sao và đúng endpoint rồi mới xóa instance đã được người học cho phép. Nếu job lỗi hoặc checksum/endpoint sai, máy được giữ để xử lý. Không tự thuê máy mới hoặc suy ra permission xóa một instance khác.

Người học đã yêu cầu push sau khi train xong. Bước cuối kiểm tra PDF thực nghiệm, cập nhật code/PDF/LaTeX và evidence, rồi commit/push; raw data và checkpoint không vào Git. Heartbeat trong chat tiếp tục việc kiểm tra và phát hành khi run dài kết thúc.

## Thực nghiệm A2 hiện tại và lượt lịch sử

`notebooks/A2.ipynb` trình bày chuẩn bị dữ liệu, huấn luyện, khóa giao thức và đánh giá trên toàn bộ Oxford-IIIT Pet: 3.310 ảnh train, 370 validation và 3.669 test. Hai biến thể chạy ba seed, mỗi lượt 12 epoch trên RTX 5090. Kết quả chính nằm trong `reports/assignment2/notebook-evidence/`: trộn cố định đạt mIoU 78,00 ± 0,11%, trộn theo biên đạt 77,82 ± 0,03%; chênh lệch ghép cặp là −0,18 điểm phần trăm. Kết quả không cho thấy cải thiện của phép trộn theo biên trong thiết lập này.

Quy trình CLI được mô tả ở [hướng dẫn A2](../src/A2/README.md). Sáu lượt RTX 3090 trước đây và kết quả trong `reports/assignment2/generated/` là bằng chứng lịch sử được giữ nguyên, không thay cho số liệu notebook hiện tại.
