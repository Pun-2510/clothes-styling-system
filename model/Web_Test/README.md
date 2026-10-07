# Website gợi ý sản phẩm

Tìm sản phẩm bằng mô tả (SBERT), ảnh (ResNet18) hoặc kết hợp (RRF).

## Cách chạy

Cài Python 3.11. Mở PowerShell tại thư mục `web_git`:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:WEB_CPU_THREADS = "4"
.\.venv\Scripts\python.exe app.py --warmup combined
```

Mở **http://127.0.0.1:8000**. Nhấn `Ctrl+C` để dừng.
Các lần sau chỉ cần chạy lại lệnh khởi động web.

Khi mở web, bảng **Kiểm thử tự động ResNet18** chạy trước trên cùng một tập
ảnh cho ba biến thể baseline, pretrained và trained. Có thể đổi số ảnh kiểm thử
bằng biến `WEB_RESNET_BENCHMARK_SAMPLES` (mặc định 120). Baseline và pretrained
dùng classification head khởi tạo ngẫu nhiên; checkpoint trained là model được
Combine sử dụng. Bảng cũng lấy tối đa 20 ảnh mỗi class từ
`../ResNet_Model/custom_data` để class `Cheongsam` được kiểm tra sau khi train.

Ảnh người dùng tải lên được xóa nền bằng `rembg` trước khi ResNet phân loại và
tạo embedding. Mặc định web dùng model nhẹ `u2netp`; có thể đổi bằng biến
`WEB_BACKGROUND_MODEL`. Đặt `WEB_REMOVE_BACKGROUND=0` nếu cần tắt bước này để
đối chứng.

## Lưu ý

- Giữ nguyên thư mục `models/`. Nếu clone từ Git, cài Git LFS và chạy `git lfs install`, `git lfs pull` để lấy trọng số.
- Lần đầu cần Internet để tải SBERT và dữ liệu; chờ model chuẩn bị xong trước khi tìm kiếm.
- Nếu cổng 8000 bận, thêm `--port 8001` rồi mở http://127.0.0.1:8001.
