# E1 — Softmax, MLP, CNN

`models.py` chứa ba classifier, output là logits. Cross-entropy trong training loop đã có log-softmax nên không gọi softmax trước loss. Xác suất chỉ được tính lúc đánh giá.

```bash
python -m scripts.run_suite --exercise E1 --seeds 42 43 44 --device cuda
```

Kết quả dùng lại làm baseline cho E3, không đổi split.
