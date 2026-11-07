# CO5085 — Học sâu và ứng dụng trong thị giác máy tính

**Huỳnh Thành Đạt · 2570161 · Học kỳ 261 (2026–2027)**  
Bài làm cá nhân. Giảng viên: Lê Thành Sách, Nguyễn Quốc Minh.

Repo gồm các bài E1–E3 trên Fashion-MNIST: phân loại bằng softmax/MLP/CNN, attention với nhiều cách chia token, và LSTM/GRU khi biểu diễn ảnh thành chuỗi. Tất cả mô hình được khởi tạo mới; không tải trọng số pretrained.

**Kết quả E1–E3.** Ba notebook đã huấn luyện và đánh giá 39 lượt trên RTX 5090. CNN đạt độ chính xác trên tập test trung bình 92,69% ± 0,06 điểm phần trăm qua ba seed. Phạm vi hỗ trợ của AI được ghi trong [AI_USE_REPORT.md](AI_USE_REPORT.md).

Thiết lập A1 gồm tám cấu hình ResNet18/Swin-T, ba seed và 20 epoch trên CIFAR-100 với phân hoạch 45.000/5.000/10.000 ảnh. Đối chứng bổ sung dùng chung lịch 40 epoch cho hai bản Swin-T. Bản khởi tạo ngẫu nhiên đạt 60,30% và bản ImageNet đạt 86,39%. Xem [notebook A1](notebooks/A1.ipynb) và [báo cáo A1](reports/assignment1/assignment1.pdf).

[Báo cáo PDF](reports/exercises/exercises.pdf) · [LaTeX](reports/exercises/report.tex) · [Đối chiếu đề](docs/REQUIREMENTS.md) · [Hướng dẫn chạy GPU](docs/GPU_RUNBOOK.md)

A2 khảo sát phân đoạn ngữ nghĩa trên Oxford-IIIT Pet, đối chiếu trộn cố định với trộn theo biên dựa trên [PIDNet, CVPR 2023](https://doi.org/10.1109/CVPR52729.2023.01871). Sáu lượt huấn luyện và đánh giá cho mIoU trung bình lần lượt 78,00% và 77,82%. Xem [notebook A2](notebooks/A2.ipynb), [báo cáo A2](reports/assignment2/assignment2.pdf) và [video minh họa](reports/assignment2/demo.mp4).

## Nội dung

| Notebook | Thí nghiệm |
|---|---|
| [E1](notebooks/E1.ipynb) | Softmax, MLP và CNN trên cùng tập ảnh và quy trình đánh giá |
| [E2](notebooks/E2.ipynb) | MSA tự viết và `nn.MultiheadAttention` với token theo hàng, patch 4×4 và đặc trưng CNN |
| [E3](notebooks/E3.ipynb) | LSTM và GRU trên chuỗi hàng hoặc patch, đối chiếu với MLP/CNN của E1 và khảo sát độ nhạy với thứ tự quét |
| [A1](notebooks/A1.ipynb) | ResNet18 và Swin-T trên CIFAR-100 với các chế độ tinh chỉnh và phép so sánh Swin-T ở lịch học 40 epoch |
| [A2](notebooks/A2.ipynb) | Đối chiếu trộn cố định và trộn theo biên cho phân đoạn trimap Oxford-IIIT Pet |

E1–E3 gồm 13 cấu hình theo yêu cầu của đề. Mỗi cấu hình dùng ba seed 42, 43 và 44, với 30 epoch cố định. Báo cáo trình bày chỉ số, môi trường và cách đo thời gian.

## Môi trường

Dùng Python 3.12 và môi trường ảo riêng. Lệnh dưới đây dành cho Linux/WSL. Trên PowerShell, lệnh kích hoạt là `.venv\Scripts\Activate.ps1`.

```bash
git clone https://github.com/dathuynh1108/CO5085-deep-learning.git
cd CO5085-deep-learning
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/torch-cuda.txt
python -m pip install --no-deps -e .
python -m pip install -r requirements.txt -r requirements/dev.txt
python -m pip install jupyterlab ipykernel nbconvert pandas
python -m ipykernel install --user --name co5085 --display-name "Python (CO5085)"
```

File CUDA ghim `torch==2.10.0`, `torchvision==0.25.0`, wheel CUDA 13.0. Đây là cặp phiên bản từ [hướng dẫn PyTorch](https://pytorch.org/get-started/previous-versions/), không phải yêu cầu dùng bản mới nhất. Driver NVIDIA phải hỗ trợ wheel đã chọn. Với CUDA 12.8, cài cùng cặp phiên bản từ index `https://download.pytorch.org/whl/cu128`; không trộn wheel CPU và CUDA. Chạy CPU thì thay file torch bằng `requirements/torch-cpu.txt`.

E1–E3 sử dụng FP32 với TF32, AMP và compile tắt. A1 sử dụng BF16 autocast với tham số và trạng thái optimizer FP32. Các lượt notebook E1–E3 và A1 dùng PyTorch 2.10.0+cu130, torchvision 0.25.0+cu130, CUDA 13 và RTX 5090. E1–E3 dùng bốn worker, A1 dùng tám. Lịch sử vận hành và checksum được ghi trong [TRAINING_EXECUTION.md](docs/TRAINING_EXECUTION.md).

## Chạy thí nghiệm bằng notebook

Mỗi notebook thực hiện lần lượt khảo sát và tiền xử lý dữ liệu, thiết lập đối chứng, huấn luyện, chọn checkpoint theo tập validation, đánh giá và tạo biểu đồ. Thư mục kết quả lưu cấu hình, seed, lịch sử huấn luyện, checkpoint, chỉ số và ma trận nhầm lẫn. Notebook là điểm bắt đầu của thí nghiệm. Các module Python và lệnh CLI tiếp tục cung cấp mã dùng chung và quy trình xuất báo cáo.

Từ thư mục gốc, mở JupyterLab, chọn notebook trong bảng trên, chọn kernel **Python (CO5085)** rồi chạy các ô từ đầu theo thứ tự.

```bash
jupyter lab
```

E1–E3 dùng cùng phân hoạch Fashion-MNIST với 54.000 ảnh huấn luyện, 6.000 ảnh validation và 10.000 ảnh test. Các thống kê chuẩn hóa được tính trên tập huấn luyện. Chạy E1 trước E3 với cùng thư mục đầu ra để E3 có thể kiểm tra và dùng lại hai đối chứng MLP/CNN. A1 và A2 có thể chạy độc lập. Chế độ GPU dùng BF16 của A1 và A2 cần phần cứng hỗ trợ BF16.

Ví dụ chạy E1 trên CUDA với seed 42 và lưu bản notebook đã thực thi.

```bash
mkdir -p artifacts/notebooks
CO5085_EXERCISES_SEEDS=42 CO5085_EXERCISES_DEVICE=cuda CO5085_EXERCISES_WORKERS=8 \
  python -m nbconvert --to notebook --execute notebooks/E1.ipynb \
  --ExecutePreprocessor.kernel_name=co5085 --ExecutePreprocessor.timeout=-1 \
  --output E1-seed42.ipynb --output-dir artifacts/notebooks
```

| Phần | Thư mục kết quả mặc định | Đánh giá trên tập test sau khi chốt giao thức |
|---|---|---|
| E1–E3 | `runs/notebooks/exercises/` | Đặt `CO5085_EXERCISES_CONFIRM_TEST=1` hoặc bật `CONFIRM_TEST` trong ô cấu hình |
| A1 | `runs/notebooks/assignment1/` và `runs/notebooks/a1_swin_followup/` | Đặt `CO5085_A1_CONFIRM_TEST=1` |
| A2 | `runs/notebooks/a2/` | Nhập `TEST` tại ô xác nhận hoặc đặt `CO5085_CONFIRM_TEST=1` khi chạy tự động |

E1–E3 và A1 hỗ trợ chạy theo seed và tiếp tục lượt bị gián đoạn với cấu hình và môi trường ban đầu. Các thiết lập nằm trong ô đầu của từng notebook. Lượt kiểm tra nhanh hoặc thay đổi giao thức sử dụng thư mục đầu ra riêng. Lệnh CLI và quy trình tái lập các lượt đã lưu được mô tả trong [hướng dẫn GPU](docs/GPU_RUNBOOK.md).

## Đánh giá và cập nhật báo cáo

E1–E3 chọn checkpoint theo độ chính xác trên tập validation, rồi xét cross-entropy trên tập validation khi hai checkpoint có cùng độ chính xác. Tập test dành cho phép đo cuối cùng sau khi chốt thiết kế. Các lệnh dưới đây thuộc quy trình CLI để đánh giá và cập nhật báo cáo từ các thư mục lượt chạy đã lưu.

```bash
python -m scripts.run_exercises_isolated --group required --seeds 42 43 44 --phase test --confirm-test --device cuda --num-workers 8
python -m E3.diagnose --run runs/exercises/e3_lstm_rows/seed_42 --device cuda
python -m scripts.export_report
python -m A1.export_report --archive
python -m scripts.write_findings --archive
python -m scripts.build_report
python -m scripts.build_report --report-root reports/assignment1 --jobname assignment1
```

`E3.diagnose` chỉ dùng validation, không cập nhật trọng số. Nó đo gradient trên các trạng thái hồi quy thật khi unroll cùng trọng số, không lấy gradient của tensor output fused rồi gọi đó là gradient BPTT.

Bước build PDF dùng XeLaTeX hoặc Tectonic với font TeX Gyre Termes (hoặc Liberation Serif/Sans), các gói `fontspec`, `geometry`, `titlesec`, `fancyhdr`, `booktabs`, `tabularx`, `longtable`, `amsmath`, `hyperref`. Trên Ubuntu có thể cài `texlive-xetex texlive-latex-extra texlive-fonts-recommended fonts-liberation fonts-dejavu-core`. Các bảng đã commit đủ để build PDF mà không tải dữ liệu hoặc checkpoint.

Bảng và hình được xuất từ log đã đối chiếu nguồn gốc. Chênh lệch giữa các cấu hình được tính theo cặp cùng seed. SD là độ lệch chuẩn mẫu qua ba seed. Các kết quả thấp, bao gồm Swin-T huấn luyện từ đầu, được giữ trong bảng để phân tích giới hạn của thiết lập.

## Cấu trúc và kiểm tra

```text
notebooks/           Điểm chạy E1, E2, E3, A1 và A2 trong Jupyter
src/                 Mã Python: common, E1, E2, E3, A1, A2
configs/             Protocol và cấu hình thí nghiệm
reports/exercises/   PDF, LaTeX, hình và evidence E1–E3
reports/assignment1/  PDF, LaTeX, hình và dữ liệu thực nghiệm A1
reports/assignment2/  PDF, LaTeX, hình, video và dữ liệu thực nghiệm A2
assignments/         Đề gốc
site/                HTML, CSS, JavaScript của landing page
scripts/             Lệnh train, export, build và kiểm tra
tests/              Unit và protocol tests
docs/               Quy trình, yêu cầu và provenance
data/               Dữ liệu cục bộ (không đưa vào Git)
runs/               Kết quả notebook và các lượt trước (không đưa vào Git)
artifacts/          Log và trạng thái vận hành (không đưa vào Git)
build/site/         Bản build Pages (không đưa vào Git)
```


```bash
python -m pytest -q
python -m compileall -q src scripts
```

Kiểm thử bao gồm forward/gradient của MSA tự viết so với PyTorch, thứ tự token, unroll LSTM/GRU, split, resume shuffle, khóa thư mục run và xuất báo cáo khi chưa có kết quả. Chúng không thay thế kiểm chứng train end-to-end trên máy đích.

## GitHub Pages

Mã trang chung, trang A1 và trang A2 nằm trong `site/`. Bản dựng gồm PDF, dữ liệu kết quả và video minh họa A2. Workflow Pages xuất bản các tệp từ nhánh `main` lên [trang học phần](https://dathuynh1108.github.io/CO5085-deep-learning/).

Trong **Settings → Pages**, chọn **Source: GitHub Actions**. Sau đó chạy workflow `Pages` hoặc push thay đổi của trang/báo cáo lên `main`. Có thể xem trước bằng:

```bash
python scripts/build_site.py
python -m http.server 8000 --directory build/site
```

Repo đã được tạo; không cần chạy `scripts/publish_github.py`. Script đó chỉ dành cho việc tạo một repository mới từ gói ZIP ban đầu, không cập nhật repo đang có.

## Nguồn và cách sử dụng

Yêu cầu thực hiện lấy từ các PDF của môn học trong `assignments/`. Repo tham khảo `dangtai111325/CO5085` chỉ được xem về cách tổ chức thư mục; các bảng, hình và kết luận thực nghiệm của bài này được tạo từ lượt chạy riêng.

Báo cáo sử dụng AI được đính kèm tại [AI_USE_REPORT.md](AI_USE_REPORT.md).

## Đường dẫn và kết quả trước khi tổ chức lại

Xem [cấu trúc repository](docs/REPOSITORY_LAYOUT.md). Cài editable package trước khi chạy các module: `python -m pip install --no-deps -e .`. Các module `E1`, `E2`, `E3`, `A1` và `A2` nằm dưới `src/`.

Các lượt GPU lịch sử tháng 10 giữ chữ ký nguồn trước refactor tại commit `1250e56`; các lượt notebook hiện tại lưu mã nguồn và kết quả riêng theo từng báo cáo. Kiểm tra bằng `python -m scripts.verify_archive --checkpoints`. Để xuất lại A1 đã đo, dùng `python -m A1.export_report --archive`; nhận xét dùng `python -m scripts.write_findings --archive`. Không resume các checkpoint này bằng mã sau refactor; chạy mới phải dùng output root mới.
