# A1 — CNN vs Transformer trên CIFAR-100

A1 so sánh ResNet18 và Swin-T trên CIFAR-100. Thiết lập 20 epoch có 24 lượt
huấn luyện và đánh giá, với ba seed 42, 43 và 44. Swin-T tinh chỉnh
toàn bộ đạt độ chính xác trên tập test 86,63% ± 0,22 điểm phần trăm, so với
78,97% ± 0,04 của ResNet18 cùng chế độ.

Ở lịch 20 epoch, Swin-T từ đầu chỉ đạt 2,76% ± 1,26 điểm phần trăm. Thí
nghiệm bổ sung so sánh hai cách khởi tạo Swin-T dưới cùng lịch 40 epoch,
có năm epoch tăng dần tốc độ học và sau đó giảm theo cosine. Hai thiết lập
dùng chung cách chia CIFAR-100 và được trình bày riêng trong [báo cáo A1](../../reports/assignment1/assignment1.pdf).

## Dữ liệu và nguồn

Dùng toàn bộ CIFAR-100: 100 fine classes, RGB 32×32, 500 ảnh train và 100 ảnh
test mỗi lớp. Từ 50.000 ảnh official train, NumPy default_rng(seed=2026) shuffle
indices từng lớp, lấy 50 ảnh/lớp làm validation: **45.000 train / 5.000 val**.
Official 10.000 test là split riêng, không tham gia train/chọn checkpoint.
Không lấy subset lớp hay ảnh. Train/val indices và hash ảnh/labels được lưu.

Nguồn chính: [Toronto CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html),
Alex Krizhevsky, Vinod Nair, Geoffrey Hinton; trích dẫn Krizhevsky (2009),
*Learning Multiple Layers of Features from Tiny Images*. Trang nguồn không nêu
license tường minh. [Official uoft-cs HF card](https://huggingface.co/datasets/uoft-cs/cifar100)
ghi **license: unknown**. Không gán MIT hoặc suy ra quyền phân phối từ license
code torchvision. Đề A1 liệt kê CIFAR-100; công bố chính xác tình trạng license
này ở A1.1. Đây không phải xác nhận license riêng cho bộ ảnh.

Dữ liệu được chuẩn bị ở `<data-root>/cifar-100-python/{train,test,meta}` bằng
nguồn official; CLI không tải ảnh ngầm. Trọng số pretrained chuẩn được
Torchvision tải/cache một lần mỗi họ khi chưa có cache; resume và test dựng
kiến trúc với weights=None rồi nạp checkpoint riêng, không tải pretrained lại.
Không sao chép pretrained weights vào mỗi run. Cặp torch/torchvision và driver
phải tương thích; phiên bản thực được lưu trong metadata. Môi trường E1–E3
đã dùng torch 2.10.0+cu130 / torchvision 0.25.0+cu130 trên CUDA 13.

## Thiết lập 20 epoch

| Thuộc tính | Giá trị |
|---|---|
| Kiến trúc | torchvision ResNet18 / Swin-T, head Linear → 100 |
| Chế độ mỗi họ | scratch full; pretrained head; pretrained partial; pretrained full |
| Pretrained | IMAGENET1K_V1, nêu exact weight enum trong config |
| Partial ResNet | layer4 + fc trainable |
| Partial Swin | features.7 (last stage) + norm + head trainable; merging trước last stage vẫn freeze |
| Budget | 20 epoch, batch 128, không drop ảnh cuối batch |
| Seeds | 42, 43, 44; fixed split seed 2026 độc lập seed train |
| Train input | RandomCrop32 padding4 → horizontal flip p=0.5 → bilinear resize96 antialias → tensor → normalize |
| Val/test input | bilinear resize96 antialias → tensor → normalize |
| Normalize | ImageNet mean [0.485,0.456,0.406], std [0.229,0.224,0.225] cho cả scratch/pretrained |
| Optimization | manual loop, AdamW β=(0.9,0.999), eps=1e-8, WD=1e-4; CE không label smoothing; global clip1 |
| LR | scratch/head 1e-3; pretrained partial/full 1e-4 |
| Scheduler | cosine per completed epoch, T_max20, eta_min0 |
| Compute | BF16 autocast tất cả chế độ; FP32 parameters/optimizer; không GradScaler, TF32 off, không compile |
| Best | val accuracy lớn nhất, hòa thì CE nhỏ nhất, hòa cả hai giữ epoch trước |

Ảnh96 giảm chi phí so với pipeline pretrained224 nhưng upscale không tạo thông
tin mới từ ảnh32. Không dùng `weights.transforms()` vì đó là recipe khác.
Kết quả chỉ có ý nghĩa trong budget/input này; **không khẳng định scratch đã
hội tụ** hoặc Transformer kém hơn nói chung. BF16 CUDA cần GPU hỗ trợ; lỗi nếu
không có, không fallback FP32. CPU BF16 chỉ phục vụ kiểm tra logic, không được
gọi là benchmark CUDA. `--deterministic` là lựa chọn cố định từ đầu.

BN buffers và stochastic depth của module frozen luôn eval khi train;
trainable subtrees mới bật train. Fingerprint frozen parameters/buffers được
kiểm tra sau mỗi epoch và khi resume. Full/scratch dùng train toàn bộ backbone.

## Notebook

Mở [A1.ipynb](../../notebooks/A1.ipynb) bằng kernel Python đã cài PyTorch và torchvision. Notebook mặc định thực hiện cả tám cấu hình 20 epoch và đối chứng khởi tạo Swin-T 40 epoch. Các ô đầu trình bày cách chia dữ liệu, cấu hình và ba seed. Các ô sau huấn luyện, chọn checkpoint trên tập validation và vẽ đường cong cùng ma trận nhầm lẫn.

Có thể chạy từng seed với `CO5085_A1_SEEDS`. Đặt `CO5085_A1_PROTOCOLS=assignment1` cho thiết lập 20 epoch hoặc `a1_swin_followup` cho đối chứng 40 epoch. Mặc định notebook xuất kết quả trên tập validation. Sau khi chốt giao thức, `CO5085_A1_CONFIRM_TEST=1` bật bước đánh giá trên tập test. Kết quả lưu dưới `runs/notebooks/`.

## Artifact và resume

Mỗi run lưu atomic `last.pt` (model+AdamW+cosine+RNG Python/NumPy/torch/CUDA+
train/val loader generators+history), `best.pt` (**model-only**, selection và
signature), `history.csv`, `metadata.json`, `split.json`, `curves.png`.
`last.pt` là commit record của epoch: interruption giữa các artifact writes
được phục hồi bằng `--resume`; epoch chưa commit chạy lại. Nếu best write bị
ngắt sau last write, best được tái tạo từ last khi last chính là epoch selected.
Không resume giữa batch; augmentation dùng seed SHA256 theo train seed/epoch/official image index
và khôi phục RNG CPU bao quanh transform; không phụ thuộc worker scheduling.
Worker không persistent; loader generators cho shuffle/worker initialization
được lưu riêng. Không tuyên bố đã capture trạng thái RNG của worker. Checkpoint tự tạo load bằng
`weights_only=False`, chỉ dùng file đáng tin cậy.

Signature chứa config, source A1 training path + common/runtime, dataset/split,
environment, seed và worker count. Git commit/dirty lưu để đối chiếu nhưng
unrelated E1–E3/report edits không làm đổi signature. Không ghi đè run khác
signature hoặc run có artifact mà thiếu last. `.a1.lock` bảo vệ train/test cùng
run; sau crash cần kiểm tra process trước khi dọn lock. Không hứa bitwise
reproducibility giữa GPU hoặc kernels khác nhau.

Sau test có `evaluation_test.json`, predictions, confusion, per-class CSV,
confusion figure và tối đa 8 lỗi confidence cao nhất (tie original index).
Không có lỗi thì không tạo gallery. Metrics accuracy/macroF1 trên đủ 100 lớp;
F1 undefined = 0. Best được chọn bằng val, không bằng test. Training time là
train pass bao gồm load/augmentation, synchronize trước/sau; epoch time thêm
validation; không gồm model setup/checkpoint/export. GPU memory là peak
**allocated** trong train+val, không phải reserved/system memory. Metadata ghi
total/trainable params và toàn bộ epoch measurements.

### Dung lượng và provenance

Không lưu best optimizer, snapshot từng epoch hay backbone copy riêng.
Ước lượng kiến trúc (không phải số đo file): ResNet18 khoảng11,23M tham số,
Swin-T khoảng27,60M sau head100. Best+last xấp xỉ `2*model_bytes +
2*trainable_parameter_bytes` (FP32 Adam moments, chưa tính serialization/RNG).
Cận trên khi mọi run đều trainable toàn bộ: 24 runs khoảng **7,5 GB decimal**
(thực tế head/partial ít optimizer state hơn), cộng temporary atomic file tối đa khoảng
0,33GB và cached weights/data. Lượt chạy sử dụng disk 70 GB và RTX 5090 với 32 GB VRAM.
Checkpoint và dữ liệu được lưu ngoài Git.

Exporter xác minh config/source/split/selection/test artifact checksums và
phạm vi seed, không trộn changed trials. `manifest.json` ghi missing/rejected
runs rõ ràng; `summary.json`, `summary.csv`, `comparison_table.tex` chỉ lấy dữ
liệu có chứng cứ. SD chỉ với ≥2 seeds; timing không gộp khi môi trường khác
nhau. Figure dùng seed nhỏ nhất của preset, không seed có điểm test cao nhất.
JSON/CSV nhẹ được copy vào `reports/assignment1/generated/evidence/<id>/seed_<n>` cùng
checksum, không copy checkpoint. Metadata/evidence phục vụ tái xuất;
checkpoint được sao lưu riêng và đối chiếu SHA256.

## Báo cáo

`reports/assignment1/report.tex` là báo cáo A1 riêng, cùng logo/fonts/layout với báo cáo
E1–E3. Exporter kiểm tra và tổng hợp kết quả; `scripts.write_findings` tính
so sánh theo cặp seed, tách bảng chất lượng/chi phí và ghi phân tích từ kết quả.
XeLaTeX hoặc Tectonic build PDF từ `reports/assignment1/`; logo nằm tại
`reports/assignment1/assets/hcmut-logo.png`. Bảng đã commit đủ để build lại PDF độc lập.
Báo cáo sử dụng AI được đính kèm tại [AI_USE_REPORT.md](../../AI_USE_REPORT.md).
