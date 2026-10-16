> Layout hiện tại: xem [REPOSITORY_LAYOUT.md](REPOSITORY_LAYOUT.md). Các đường dẫn trong phần lịch sử bên dưới là đường dẫn tại thời điểm chạy.

# Thực thi GPU ngày 14/10/2026

## Phạm vi và protocol

Người học cho phép clone và chỉnh repo, train E1–E3 đã có code và thực hiện A1. A2 chờ chọn paper. Người học đã yêu cầu cập nhật code/report/PDF/LaTeX, commit và push sau khi train xong; không tự nộp LMS. Sau train: sao lưu checkpoint/log/kết quả vào workspace, kiểm tra checksum rồi xóa instance Vast.ai.

- E1–E3: giữ nguyên 20 config × seed 42/43/44, mỗi run 30 epoch; Fashion-MNIST 54.000/6.000/10.000; FP32, không AMP/TF32/compile/pretrained. Source thuộc `common/`, `E1/`, `E2/`, `E3/` được khóa trước lượt đầu.
- A1: 8 config × seed 42/43/44, mỗi run 20 epoch; CIFAR-100 đầy đủ, 45.000/5.000/10.000; ảnh96, BF16, tắt TF32/compile. Scratch và ba chế độ pretrained cho cả ResNet18/Swin-T.
- Validation chọn best; held-out test thực hiện một lần cho từng checkpoint đã khóa. Không chỉnh hyperparameter dựa trên test.

## Máy

Instance cũ `55937255` không khởi động được do host hết tài nguyên. Người học yêu cầu xóa và thuê máy khác; CLI xác nhận sau đó chỉ còn instance mới.

Instance mới `55938548`: một NVIDIA GeForce RTX 5090, 32.607 MiB VRAM, driver 580.159.03, disk70GB. Giá khi thuê và cập nhật disk khoảng 0,55093 USD/giờ, chưa bao gồm traffic. Runtime/software và thời gian đo sẽ lấy từ artifact thật của từng run.

## GitHub Pages

Pages đã được bật với `build_type=workflow` ngày14/10/2026, sửa nguyên nhân HTTP404 ở configure-pages. Run37901183673 attempt2 đã success; landing và CSS/PDF trảHTTP200. Deployment đã cập nhật landing/favicon ở commit `6180055`; nội dung A1 và kết quả mới cập nhật cùng đợt push sau train. Evidence và phạm vi ở `docs/PAGES_DEPLOYMENT.md`.

## Gián đoạn và khôi phục vận hành

Ngày14/10 lúc15:49 UTC, suite dừng sau29 run hoàn tất30 epoch, ở validation epoch đầu của `e3_lstm_rows/seed_44`: `OSError: [Errno 24] Too many open files`. Giới hạn soft NOFILE là1024. Kiểm tra DataLoader tổng hợp không dùng dữ liệu thật, với pinned memory/persistent workers, ghi nhận số descriptor sau cleanup tăng46→66 qua6 loader. Chạy nhiều run trong cùng tiến trình giữ lại handle của worker; đây là lỗi tài nguyên của orchestration, không phải lỗi mô hình hoặc config.

`scripts/run_exercises_isolated.py` gọi entrypoint train/test hiện có trong một tiến trình mới cho mỗi config/seed, chạy tuần tự và dừng ngay khi child lỗi. Giữ đủ20 config×3 seed, workers8, OMP/MKL4, precision, split và32 source/config hash. Các run hoàn tất đi qua resume guards, không chạy thêm epoch. Bốn test vận hành kiểm tra isolation, resume chỉ khi có checkpoint, quyền mở test và fail-fast đã pass; Bash/shellcheck/shfmt cũng pass.

Run seed44 chưa có checkpoint/history/metrics; `run.json` của lần bị lỗi được lưu cùng log trước khi khởi động lại từ seed ban đầu. Không có epoch hoàn tất bị bỏ hoặc điểm test được dùng để chỉnh protocol. Chi tiết và hash lần gián đoạn lưu trong `logs/fd-recovery.json`; log lỗi ban đầu giữ trong `logs/training-fd-failure-20261014.log`. Sau restart chỉ tiến trình của một run giữ DataLoader handles; process exit giải phóng chúng trước run tiếp theo.

Kiểm tra sau restart: seed44 đã hoàn tất30 epoch, signature không đổi; SHA256 của `last.pt`, `history.csv` và `run.json` ở29 run trước vẫn nguyên. Suite tiếp tục config tiếp theo; runner giữ3 descriptor, tiến trình train đang chạy giữ112, dưới giới hạn1024. Evidence nhẹ: `docs/training-evidence/fd-recovery.json`. Bộ theo dõi sao lưu/xóa instance đã khởi động lại.

## Khôi phục tải CIFAR-100

Torchvision dừng ở file13.336.000 byte với lỗi `File not found or corrupted`; MD5 đã ngăn archive thiếu đi vào training. A1 chưa có metadata/checkpoint ở thời điểm đó. Log gốc giữ ở `logs/training-cifar-failure-20261014.log`, receipt ở `logs/cifar-job-recovery.json` và bản nhẹ `docs/training-evidence/cifar-job-recovery.json`.

`scripts/prefetch_cifar100.py` tải tiếp tuần tự512KiB mỗi request từ đúng URL Toronto của torchvision, kiểm tra status206, Content-Range, tổng độ dài và số byte. Retry lỗi mạng tối đa3 lần mỗi chunk, giới hạn toàn lượt2 giờ, giữ `.partial` khi lỗi và khóa một downloader. Chỉ atomic promote thành archive sẵn dùng sau khi đủ byte và MD5 chính thức `eb9058c3a382ffc7106e4002c42a8d85` khớp. Torchvision tiếp tục kiểm tra và giải nén theo đường cũ. Bốn test resume/integrity/range/retry đã pass; shellcheck/shfmt và cú pháp shell pass.

Master đã khởi động lại, đưa các E run hoàn tất qua resume guards và tái sử dụng test đã khóa; không train thêm epoch hoặc mở test để tuning. Source/config32 hash giữ nguyên. Cơ chế tải chỉ nằm trong script vận hành, không sửa `A1/`, `common/` hoặc experiment config. Audit tiến độ/MD5/SHA256 ở `logs/cifar-download-recovery.json`. Bộ theo dõi sao lưu/xóa instance đã khởi động lại.

## Khôi phục sao lưu sau khi hoàn tất

Đường SSH trực tiếp nhiều lần bị host ngắt, rsync exit255; bộ thu kết quả dừng trước bước xóa. Đã thêm SSH keepalive, giữ phần tải dở và giới hạn retry chỉ cho lỗi SSH255; sáu kiểm thử cleanup/transfer đã pass. Máy không có OOM kill. Probe trực tiếp8MiB không xong trong100 giây; proxy Vast của cùng instance truyền đủ trong14,69 giây và khớp SHA256. Chuyển phần thu cuối sang proxy, giữ nguyên kiểm tra identity và toàn bộ checksum. Chi tiết ở `docs/training-evidence/transfer-recovery.json`.

## Trạng thái hoàn tất

E1–E3:60/60 lượt train đủ30 epoch và60 lượt held-out test. A1:24/24 lượt train đủ20 epoch và24 lượt held-out test. Cả hai dùng seed42/43/44, workers8 và4 CPU threads. Môi trường GPU: Python3.12.14, torch2.10.0+cu130, torchvision0.25.0+cu130, CUDA13; RTX5090. Có56 kiểm thử số học/protocol pass trên GPU trước train; kiểm thử vận hành và xuất bản được chạy riêng khi thay đổi các script.

Source/config32 hash giữ nguyên trong toàn bộ lượt đo. Diagnostic baseline/combined E3 sử dụng256 ảnh validation cho ba seed. Các metrics accuracy và macro-F1 đã được tính lại độc lập từ ma trận lỗi của đủ84 lượt test; audit ở `docs/training-evidence/numerical-audit.json`.

Đã tải và đối chiếu SHA256 đủ991 tệp, gồm checkpoint best/last của60+24 lượt chạy, metadata, kết quả test, thông tin split và log. Manifest ở `docs/training-evidence/backup-manifest.json`. Checkpoint/log nằm trong `runs/`, `A1/runs/`, `logs/` ở workspace và bị Git ignore. Dữ liệu thô không đưa vào Git.

Instance55938548 đã được xóa sau khi đủ các kiểm tra; CLI xác nhận không còn instance này. Instance cũ55937255 cũng đã được xóa trước lượt chạy thay thế.

Báo cáo E1–E3 và A1 có PDF/LaTeX riêng, dùng kết quả thực đo và văn phong học thuật trung tính, hạn chế đại từ. Mỗi PDF chỉ dẫn một câu đến `AI_USE_REPORT.md`. Bản LaTeX và bảng/figures đã lưu trong repository để build lại PDF mà không tải checkpoint. Kết quả QA và hash PDF cuối ở `docs/training-evidence/final-pdf-qa.json`; Git history và các workflow CI/Pages ghi nhận bản phát hành.
