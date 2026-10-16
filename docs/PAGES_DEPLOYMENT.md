> Layout hiện tại: xem [REPOSITORY_LAYOUT.md](REPOSITORY_LAYOUT.md). Các đường dẫn trong phần lịch sử bên dưới là đường dẫn tại thời điểm chạy.

# GitHub Pages — khắc phục ngày14/10/2026

Landing: https://dathuynh1108.github.io/CO5085-deep-learning/

## Nguyên nhân và sửa

Run [Pages37901183673](https://github.com/dathuynh1108/CO5085-deep-learning/actions/runs/37901183673), commit `2dfb685dbf5c5cd3eabb2a82b7404632230b90b8`, fail ở `actions/configure-pages@v5`: `Get Pages site failed` / HTTP404. Bước `python scripts/build_site.py --output build/site` đã pass. API Pages cũng trả404 dù account có quyền admin repo: Pages chưa được tạo/bật.

Đã tạo Pages bằng API với `build_type=workflow`, rồi rerun failed job. Attempt2 hoàn tất **success** lúc22:11 ngày14/10/2026 (GMT+7). Configure, upload artifact và deploy đều pass. Không sửa source/config training hoặc ghi đè các thay đổi đang chuẩn bị cho A1.

Lệnh khởi tạo (chỉ dùng một lần khi Pages chưa có, bằng account được phép quản lý repo):

```bash
gh api --method POST repos/dathuynh1108/CO5085-deep-learning/pages -f build_type=workflow
gh run rerun 37901183673 --repo dathuynh1108/CO5085-deep-learning --failed
```

Các lần sau workflow dùng `pages: write`, `id-token: write`, build thư mục tĩnh và deploy artifact; không cần token cá nhân trong workflow hoặc tự bật Pages ở mỗi lượt.

## Phạm vi nội dung đã publish

Deployment đã xác minh dùng commit đã có trên main ngày09/10/2026: landing E1–E3 và PDF bản thảo chưa có số thực nghiệm. Các đoạn “chưa huấn luyện” thuộc snapshot báo cáo này, không đại diện trạng thái job GPU đang chạy ngày14/10.

Trang A1 riêng, khai báo AI và báo cáo mới đang được chuẩn bị trong workspace. Người học yêu cầu push chúng cùng code/PDF/LaTeX sau khi đủ train và kiểm tra kết quả. Bản build site mới đã có mục A1/A2; A2 ghi chờ chọn paper. Không khai báo số thực nghiệm trước khi artifact đầy đủ.

## Bản sửa site và kiểm tra live

Đã push riêng commit `6180055e8b6776e1e17467d548ca5f68e6bb04d9`: thêm favicon tương đối `favicon.svg` và mô tả trạng thái theo artifact báo cáo, không nói toàn bộ job chưa train. Các thay đổi A1/report còn lại giữ trong workspace cho đợt publish sau train.

Pages [run38588169554](https://github.com/dathuynh1108/CO5085-deep-learning/actions/runs/38588169554) và CPU tests [run38588169653](https://github.com/dathuynh1108/CO5085-deep-learning/actions/runs/38588169653) đều success trên đúng commit; CI có37 test pass. Hai test delivery local cũng pass.

Kiểm tra bằng Chrome có sẵn qua Playwright tại1440×1000 và390×844: đúng URL/title, có nội dung, không tràn ngang, không console/page error, click mục Bài tập chuyển đến `#exercises`. Landing/CSS/favicon/PDF/JSON hoạt động; PDF và JSON trảHTTP200 ở cả hai viewport. Đã xem screenshot desktop/mobile. Chỉ kiểm tra snapshot landing hiện tại; trang A1 mới sẽ được kiểm tra lại sau full publish.
