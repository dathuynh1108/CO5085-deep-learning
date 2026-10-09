# E2 — MSA và tokenization

Có sáu preset: `{patches,rows,cnn}` × `{manual,torch}`. `ManualSelfAttention` chỉ dùng các phép toán cơ bản. `TorchSelfAttention` dùng `nn.MultiheadAttention`. Cả model được khởi tạo cùng tensor trước khi thay backend, kiểm chứng ở `tests/test_core.py`.

```bash
python -m scripts.run_suite --exercise E2 --seeds 42 43 44 --device cuda
```

Token rows dài 28; patches/CNN dài 49. CNN-stem được học từ đầu và có thêm tham số; không gọi sự khác biệt này là hiệu quả tokenization thuần túy.
