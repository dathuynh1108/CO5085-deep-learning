# CO5085 — Học sâu và ứng dụng trong thị giác máy tính

**Huỳnh Thành Đạt · 2570161 · Học kỳ 261 (2026–2027)**

Bài làm cá nhân gồm các notebook thực nghiệm, báo cáo và đề gốc. [Trang học phần](https://dathuynh1108.github.io/CO5085-deep-learning/) tổng hợp nội dung và kết quả.

| Phần | Notebook | Báo cáo | Đề |
|---|---|---|---|
| E1–E3: phân loại Fashion-MNIST bằng softmax, MLP, CNN, attention, LSTM và GRU | [E1](notebooks/E1.ipynb) · [E2](notebooks/E2.ipynb) · [E3](notebooks/E3.ipynb) | [PDF](reports/exercises/exercises.pdf) | [PDF](assignments/exercise-vne.pdf) |
| A1: ResNet18 và Swin-T trên CIFAR-100 | [A1](notebooks/A1.ipynb) | [PDF](reports/assignment1/assignment1.pdf) | [PDF](assignments/assignment1-vne.pdf) |
| A2: trộn cố định và trộn theo biên trên Oxford-IIIT Pet | [A2](notebooks/A2.ipynb) | [PDF](reports/assignment2/assignment2.pdf) · [Video](reports/assignment2/demo.mp4) | [PDF](assignments/assignment2-vne.pdf) |

Các notebook chứa thiết lập, mã thực nghiệm và kết quả thực thi. Mã dùng chung nằm trong `src/`, cấu hình trong `configs/`; báo cáo gồm PDF, LaTeX và hình minh họa. Giữ các thư mục này cùng notebook khi chạy lại.

Dùng Python 3.12, cài môi trường và mở JupyterLab từ thư mục gốc:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/torch-cuda.txt
python -m pip install -r requirements.txt -r requirements/dev.txt
python -m pip install --no-deps -e .
python -m pip install jupyterlab ipykernel nbconvert pandas
jupyter lab
```

Có thể dùng `requirements/torch-cpu.txt` khi chạy trên CPU. Chạy E1 trước E3 để dùng lại đối chứng MLP/CNN. Các mô hình chọn checkpoint bằng validation; chỉ xác nhận đánh giá trên tập test sau khi chốt giao thức trong notebook.

Kiểm tra mã và dựng trang từ các kết quả đã lưu:

```bash
python -m pytest -q
python scripts/build_site.py
```

Phạm vi hỗ trợ AI được trình bày trong [AI_USE_REPORT.md](AI_USE_REPORT.md).
