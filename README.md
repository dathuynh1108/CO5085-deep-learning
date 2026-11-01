# CO5085 — Học sâu và ứng dụng trong thị giác máy tính

**Huỳnh Thành Đạt · 2570161 · Học kỳ 261 (2026–2027)**

Bài làm cá nhân gồm các notebook thực nghiệm, báo cáo và đề gốc. [Trang học phần](https://dathuynh1108.github.io/CO5085-deep-learning/) tổng hợp nội dung và kết quả.

| Phần | Notebook | Báo cáo | Đề |
|---|---|---|---|
| E1–E3: phân loại Fashion-MNIST bằng softmax, MLP, CNN, attention, LSTM và GRU | [E1](notebooks/E1.ipynb) · [E2](notebooks/E2.ipynb) · [E3](notebooks/E3.ipynb) | [PDF](reports/exercises/exercises.pdf) | [PDF](assignments/exercise-vne.pdf) |
| A1: ResNet18 và Swin-T trên CIFAR-100 | [A1](notebooks/A1.ipynb) | [PDF](reports/assignment1/assignment1.pdf) | [PDF](assignments/assignment1-vne.pdf) |
| A2: PIDNet-S và giám sát theo biên trên CamVid | [A2](notebooks/A2.ipynb) | [PDF](reports/assignment2/assignment2.pdf) · [Video](reports/assignment2/demo.mp4) | [PDF](assignments/assignment2-vne.pdf) |

Mỗi notebook là một thực nghiệm độc lập, viết bằng tiếng Anh và chứa đầy đủ mã chuẩn bị dữ liệu, mô hình, huấn luyện, đánh giá và trực quan hóa. Có thể mở riêng từng notebook trên Jupyter hoặc Google Colab rồi chạy lần lượt các ô từ đầu đến cuối; không cần tải mã nguồn khác trong kho. E3 tự huấn luyện các đối chứng MLP và CNN cùng với LSTM và GRU.

Môi trường thực nghiệm dùng Python 3.12 và PyTorch 2.10. Có thể mở notebook trên Colab hoặc khởi động JupyterLab tại thư mục gốc:

```bash
python -m pip install jupyterlab
python -m jupyter lab
```

Mở notebook cần chạy và chọn **Run all**. Chọn môi trường GPU cho các lượt huấn luyện đầy đủ. Ô cài đặt đầu notebook chuẩn bị các thư viện cần thiết. Dữ liệu được tải và chia tập trong notebook. Tập test được đánh giá sau khi hoàn tất huấn luyện và lựa chọn checkpoint bằng validation.

E1–E3 sử dụng Fashion-MNIST. A1 nghiên cứu ResNet18 và Swin-T trên CIFAR-100. A2 giữ nguyên kiến trúc PIDNet-S, gồm ba nhánh P/I/D, Pag, PAPPM và Light-Bag, trên CamVid 960×720. Hai biến thể giữ nguyên kiến trúc, so sánh việc dùng đủ loss của tác giả với việc bỏ riêng BAS-Loss.

Phạm vi hỗ trợ AI được trình bày trong [AI_USE_REPORT.md](AI_USE_REPORT.md).
