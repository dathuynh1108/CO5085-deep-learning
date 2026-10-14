# CO5085 — Học sâu và ứng dụng trong thị giác máy tính

**Huỳnh Thành Đạt · 2570161 · Học kỳ 261 (2026–2027)**  
Bài làm cá nhân. Giảng viên: Lê Thành Sách, Nguyễn Quốc Minh.

Repo gồm các bài E1–E3 trên Fashion-MNIST: phân loại bằng softmax/MLP/CNN, attention với nhiều cách chia token, và LSTM/GRU khi biểu diễn ảnh thành chuỗi. Tất cả mô hình được khởi tạo mới; không tải trọng số pretrained.

**Kết quả ngày 15/10/2026:** E1–E3 hoàn tất 60/60 lượt train và test; A1 hoàn tất 24/24 lượt train và test trên RTX 5090. CNN E1 đạt 92,69% ±0,15%; Swin-T full A1 đạt 86,63% ±0,22% test accuracy (trung bình ±SD mẫu, ba seed). A2 chờ chọn paper. Báo cáo sử dụng AI được đính kèm tại [AI_USE_REPORT.md](AI_USE_REPORT.md).

A1 có tám cấu hình ResNet18/Swin-T × scratch/head/partial/full, ba seed, 20 epoch; dùng toàn bộ CIFAR-100 với split45.000/5.000/10.000. Code ở [A1](A1/README.md), báo cáo tách riêng [A1 LaTeX](A1/reports/report.tex) và [A1 PDF](A1/reports/assignment1.pdf). PDF E1–E3 không chứa assignment.

[Báo cáo PDF](reports/exercises.pdf) · [LaTeX](reports/report.tex) · [Đối chiếu đề](docs/REQUIREMENTS.md) · [Hướng dẫn chạy GPU](docs/GPU_RUNBOOK.md)

## Nội dung

| Phần | Thí nghiệm |
|---|---|
| E1 | Softmax, MLP, CNN; cùng split và quy trình đánh giá |
| E2 | MSA tự viết và `nn.MultiheadAttention`; token theo hàng, patch 4×4, CNN-stem |
| E3 | LSTM và GRU; chuỗi hàng hoặc patch; đối chiếu lại MLP/CNN của E1 |
| Mở rộng E3 | Khởi tạo trực giao, phạt sai lệch trực giao, bias cổng nhớ theo độ dài chuỗi, mean pooling và ablation |

Có 13 cấu hình bắt buộc và 7 cấu hình mở rộng, mỗi cấu hình chạy đủ ba seed 42,43,44 và 30 epoch, không early stopping. Bảng thời gian thực đo, môi trường và phạm vi phép đo nằm trong báo cáo.

Phần mở rộng kiểm tra ảnh hưởng của khởi tạo và đường đọc trạng thái đối với LSTM trên chuỗi ảnh ngắn. Cả bảy biến thể có test accuracy trung bình thấp hơn baseline LSTM rows trong thiết lập đã đo. Báo cáo phân tích các ablation và phân biệt regularization trực giao mềm với tối ưu có ràng buộc trên đa tạp.

## Môi trường

Dùng Python 3.11–3.13 và môi trường ảo riêng. Lệnh dưới đây dành cho Linux/WSL; trên PowerShell thay lệnh kích hoạt bằng `.venv\Scripts\Activate.ps1`.

```bash
git clone https://github.com/dathuynh1108/CO5085-deep-learning.git
cd CO5085-deep-learning
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/torch-cuda.txt
python -m pip install -r requirements.txt -r requirements/dev.txt
```

File CUDA ghim `torch==2.10.0`, `torchvision==0.25.0`, wheel CUDA 13.0. Đây là cặp phiên bản từ [hướng dẫn PyTorch](https://pytorch.org/get-started/previous-versions/), không phải yêu cầu dùng bản mới nhất. Driver NVIDIA phải hỗ trợ wheel đã chọn. Với CUDA 12.8, cài cùng cặp phiên bản từ index `https://download.pytorch.org/whl/cu128`; không trộn wheel CPU và CUDA. Chạy CPU thì thay file torch bằng `requirements/torch-cpu.txt`.

E1–E3 sử dụng FP32, tắt TF32, AMP và compile; A1 sử dụng BF16 autocast với tham số/optimizer FP32. Cả 84 lượt train và test đã chạy trên PyTorch 2.10.0+cu130, torchvision 0.25.0+cu130, CUDA 13, RTX 5090, workers 8 và 4 CPU threads. Có 56 kiểm thử số học/protocol đã pass trên GPU trước lượt train; lịch sử vận hành và checksum ở [TRAINING_EXECUTION.md](docs/TRAINING_EXECUTION.md).

## Chạy thí nghiệm

Chạy từ thư mục gốc repo. Lệnh liệt kê không đọc dữ liệu:

```bash
python -m scripts.run_exercises_isolated --group all --seeds 42 43 44 --list
python -m scripts.prepare_data --data-root data
python -m scripts.run_exercises_isolated --group all --seeds 42 43 44 --device cuda --num-workers 8
```

Dữ liệu được lấy bằng loader torchvision, ưu tiên HTTPS từ repo chính thức của Fashion-MNIST và giữ kiểm tra checksum. Train/validation có 54.000/6.000 ảnh; test giữ nguyên 10.000 ảnh. Mean/std chỉ tính trên phần train. Split seed cố định là 2026, độc lập với seed khởi tạo mô hình.

Một cấu hình riêng:

```bash
python -m E2.train --experiment e2_patches_manual --seed 42 --device cuda
python -m E3.train --experiment e3_lstm_rows_combined --seed 42 --device cuda
```

Để tiếp tục sau gián đoạn, dùng lại lệnh suite với `--resume`. Checkpoint lưu trạng thái model, optimizer, scheduler và các bộ sinh số ngẫu nhiên. Resume yêu cầu cùng code/config/split/môi trường đã ghi. Không xóa `.train.lock` khi tiến trình cũ vẫn chạy. Thay batch size hoặc số epoch thì dùng thư mục output mới; kết quả thử cấu hình không trộn vào báo cáo chính.

## Đánh giá và cập nhật báo cáo

Checkpoint được chọn bằng validation accuracy, hòa thì xét validation cross-entropy. **Không dùng test để chọn mô hình hoặc chỉnh siêu tham số.** Sau khi chốt thiết kế và hoàn tất train:

```bash
python -m scripts.run_exercises_isolated --group all --seeds 42 43 44 --phase test --confirm-test --device cuda --num-workers 8
python -m E3.diagnose --run runs/e3_lstm_rows/seed_42 --device cuda
python -m E3.diagnose --run runs/e3_lstm_rows_combined/seed_42 --device cuda
python -m scripts.export_report
python -m A1.export_report
python -m scripts.write_findings
python -m scripts.build_report
python -m scripts.build_report --report-root A1/reports --jobname assignment1
```

`E3.diagnose` chỉ dùng validation, không cập nhật trọng số. Nó đo gradient trên các trạng thái hồi quy thật khi unroll cùng trọng số, không lấy gradient của tensor output fused rồi gọi đó là gradient BPTT.

Bước build PDF dùng XeLaTeX hoặc Tectonic với font TeX Gyre Termes (hoặc Liberation Serif/Sans), các gói `fontspec`, `geometry`, `titlesec`, `fancyhdr`, `booktabs`, `tabularx`, `longtable`, `amsmath`, `hyperref`. Trên Ubuntu có thể cài `texlive-xetex texlive-latex-extra texlive-fonts-recommended fonts-liberation fonts-dejavu-core`. Các bảng đã commit đủ để build PDF mà không tải dữ liệu hoặc checkpoint.

Bảng và hình được xuất từ log đã kiểm tra provenance. `scripts.write_findings` tính chênh lệch theo cặp seed và ghi phân tích từ đủ 60+24 lượt train/test, sau đó báo cáo được rà soát cùng đường cong và ma trận lỗi. SD là độ lệch chuẩn mẫu qua ba seed. Các kết quả thấp, bao gồm Swin-T scratch, được giữ trong bảng và thảo luận giới hạn.

## Cấu trúc và kiểm tra

```text
E1/, E2/, E3/       Mô hình, entry point và chẩn đoán hồi quy
common/             Split, training loop, checkpoint, evaluation, plots
configs/            20 cấu hình thí nghiệm
A1/                 Mã và báo cáo riêng A1
scripts/            Chạy suite, xuất kết quả, build PDF, chuẩn bị Pages
reports/            Một báo cáo E1–E3, mã LaTeX và kết quả được xuất
site/               Trang giới thiệu cho GitHub Pages
tests/              Kiểm thử số học và quy trình, không tải dataset
docs/               Mapping yêu cầu, thiết kế thí nghiệm, hướng dẫn GPU
```

```bash
python -m pytest -q
python -m compileall -q common E1 E2 E3 A1 scripts
```

Kiểm thử bao gồm forward/gradient của MSA tự viết so với PyTorch, thứ tự token, unroll LSTM/GRU, gradient của regularizer, split, resume shuffle, khóa thư mục run và xuất báo cáo khi chưa có kết quả. Chúng không thay thế kiểm chứng train end-to-end trên máy đích.

## GitHub Pages

Trang giới thiệu được dựng từ `site/`, kèm hai PDF riêng và JSON kết quả. [Trang chung](https://dathuynh1108.github.io/CO5085-deep-learning/) liên kết E1–E3 và [trang A1](https://dathuynh1108.github.io/CO5085-deep-learning/a1.html). Workflow Pages xuất các artifact đã commit.

Trong **Settings → Pages**, chọn **Source: GitHub Actions**. Sau đó chạy workflow `Pages` hoặc push thay đổi của trang/báo cáo lên `main`. Có thể xem trước bằng:

```bash
python scripts/build_site.py
python -m http.server 8000 --directory site-preview
```

Repo đã được tạo; không cần chạy `scripts/publish_github.py`. Script đó chỉ dành cho việc tạo một repository mới từ gói ZIP ban đầu, không cập nhật repo đang có.

## Nguồn và cách sử dụng

Yêu cầu thực hiện lấy từ các PDF của môn học trong `assignments/`. Repo tham khảo `dangtai111325/CO5085` chỉ được xem về cách tổ chức thư mục; các bảng, hình và kết luận thực nghiệm của bài này được tạo từ lượt chạy riêng.

Báo cáo sử dụng AI được đính kèm tại [AI_USE_REPORT.md](AI_USE_REPORT.md).
