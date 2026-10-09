# CO5085 — Học sâu và ứng dụng trong thị giác máy tính

**Huỳnh Thành Đạt · 2570161 · Học kỳ 261 (2026–2027)**  
Bài làm cá nhân. Giảng viên: Lê Thành Sách, Nguyễn Quốc Minh.

Repo gồm các bài E1–E3 trên Fashion-MNIST: phân loại bằng softmax/MLP/CNN, attention với nhiều cách chia token, và LSTM/GRU khi biểu diễn ảnh thành chuỗi. Tất cả mô hình được khởi tạo mới; không tải trọng số pretrained.

**Trạng thái:** mã nguồn và bản thảo báo cáo đã có; chưa tải dữ liệu, chưa huấn luyện và chưa có kết quả thực nghiệm. Số tham số trong báo cáo được đếm từ kiến trúc, không phải số liệu train. Phần A1/A2 chưa triển khai vì chưa có nội dung hai đề tương ứng trong bộ tài liệu được cung cấp. Đây chưa phải bản nộp cuối cùng.

[Báo cáo PDF](reports/exercises.pdf) · [LaTeX](reports/report.tex) · [Đối chiếu đề](docs/REQUIREMENTS.md) · [Hướng dẫn chạy GPU](docs/GPU_RUNBOOK.md)

## Nội dung

| Phần | Thí nghiệm |
|---|---|
| E1 | Softmax, MLP, CNN; cùng split và quy trình đánh giá |
| E2 | MSA tự viết và `nn.MultiheadAttention`; token theo hàng, patch 4×4, CNN-stem |
| E3 | LSTM và GRU; chuỗi hàng hoặc patch; đối chiếu lại MLP/CNN của E1 |
| Mở rộng E3 | Khởi tạo trực giao, phạt sai lệch trực giao, bias cổng nhớ theo độ dài chuỗi, mean pooling và ablation |

Có 13 cấu hình bắt buộc và 7 cấu hình mở rộng. Chạy đủ ba seed 42, 43, 44 tương ứng 60 lượt huấn luyện. Cấu hình mặc định là 30 epoch/lượt, không early stopping. Chưa có ước lượng thời gian vì chưa đo trên GPU đích.

Phần mở rộng không phải một phương pháp mới được chứng minh tốt hơn. Đây là một giả thuyết cần kiểm tra: thay đổi khởi tạo và đường đọc trạng thái có giúp LSTM học ổn định hơn trên chuỗi ảnh ngắn hay không. Báo cáo phân biệt rõ regularization trực giao mềm với tối ưu có ràng buộc trên đa tạp.

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

Mã sử dụng FP32, tắt TF32, chưa bật AMP hoặc `torch.compile`. Lựa chọn này giúp phép so sánh giữa các cách hiện thực dễ kiểm soát hơn. Unit test đã chạy trên PyTorch 2.10.0+cpu / torchvision 0.25.0+cpu; chưa xác nhận một lượt train hoàn chỉnh trên CUDA.

## Chạy thí nghiệm

Chạy từ thư mục gốc repo. Lệnh liệt kê không đọc dữ liệu:

```bash
python -m scripts.run_suite --group all --seeds 42 43 44 --list
python -m scripts.prepare_data --data-root data
python -m scripts.run_suite --group all --seeds 42 43 44 --device cuda --num-workers 4
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
python -m scripts.run_suite --group all --seeds 42 43 44 --phase test --confirm-test --device cuda
python -m E3.diagnose --run runs/e3_lstm_rows/seed_42 --device cuda
python -m E3.diagnose --run runs/e3_lstm_rows_combined/seed_42 --device cuda
python -m scripts.export_report
python -m scripts.build_report
```

`E3.diagnose` chỉ dùng validation, không cập nhật trọng số. Nó đo gradient trên các trạng thái hồi quy thật khi unroll cùng trọng số, không lấy gradient của tensor output fused rồi gọi đó là gradient BPTT.

Bước build PDF cần XeLaTeX: TeX Live hoặc MiKTeX với font TeX Gyre Termes (hoặc Liberation Serif/Sans làm fallback), các gói `fontspec`, `geometry`, `titlesec`, `fancyhdr`, `booktabs`, `tabularx`, `longtable`, `amsmath`, `hyperref`. Trên Ubuntu có thể cài `texlive-xetex texlive-latex-extra texlive-fonts-recommended fonts-liberation fonts-dejavu-core`.

Bảng và hình tự cập nhật từ log; phần nhận xét vẫn phải đọc và viết lại theo kết quả thật. Không có code sinh nhận xét thành tích hoặc tự điền accuracy. Một seed không có độ lệch chuẩn; từ hai seed mới tính sample standard deviation. Các ô chưa đo là `null` trong JSON và `--` trong PDF.

## Cấu trúc và kiểm tra

```text
E1/, E2/, E3/       Mô hình, entry point và chẩn đoán hồi quy
common/             Split, training loop, checkpoint, evaluation, plots
configs/            20 cấu hình thí nghiệm
scripts/            Chạy suite, xuất kết quả, build PDF, chuẩn bị Pages
reports/            Một báo cáo E1–E3, mã LaTeX và kết quả được xuất
site/               Trang giới thiệu cho GitHub Pages
tests/              Kiểm thử số học và quy trình, không tải dataset
docs/               Mapping yêu cầu, thiết kế thí nghiệm, hướng dẫn GPU
```

```bash
python -m pytest -q
python -m compileall -q common E1 E2 E3 scripts
```

Kiểm thử bao gồm forward/gradient của MSA tự viết so với PyTorch, thứ tự token, unroll LSTM/GRU, gradient của regularizer, split, resume shuffle, khóa thư mục run và xuất báo cáo khi chưa có kết quả. Chúng không thay thế kiểm chứng train end-to-end trên máy đích.

## GitHub Pages

Trang giới thiệu được dựng từ `site/`, kèm PDF tại `reports/exercises.pdf`. Workflow `Pages` chỉ xuất trang và báo cáo đang có; không train mô hình.

Trong **Settings → Pages**, chọn **Source: GitHub Actions**. Sau đó chạy workflow `Pages` hoặc push thay đổi của trang/báo cáo lên `main`. Có thể xem trước bằng:

```bash
python scripts/build_site.py
python -m http.server 8000 --directory site-preview
```

Repo đã được tạo; không cần chạy `scripts/publish_github.py`. Script đó chỉ dành cho việc tạo một repository mới từ gói ZIP ban đầu, không cập nhật repo đang có.

## Nguồn và cách sử dụng

Yêu cầu thực hiện lấy từ `exercise-vne.pdf` của môn học. Repo tham khảo `dangtai111325/CO5085` chỉ được xem để tham khảo cách tổ chức thư mục; không dùng code, hình hoặc số liệu thực nghiệm của repo đó. Bố cục LaTeX dựa trên mẫu Diffusion Maps được cung cấp; nội dung báo cáo viết riêng cho E1–E3.

Bộ mã và bản thảo ban đầu có mức hỗ trợ AI đáng kể. Đề môn học chỉ cho phép AI hỗ trợ và yêu cầu khai báo; do đó không nên nộp nguyên trạng như một sản phẩm tự thực hiện. Người học cần tự kiểm tra, sửa và giải thích được cách làm, chạy thí nghiệm thật, viết nhận xét của mình và làm rõ mức hỗ trợ với giảng viên. Chi tiết ở [Academic integrity](docs/ACADEMIC_INTEGRITY.md).
