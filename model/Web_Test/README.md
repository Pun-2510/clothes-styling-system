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

## Lưu ý

- Giữ nguyên thư mục `models/`. Nếu clone từ Git, cài Git LFS và chạy `git lfs install`, `git lfs pull` để lấy trọng số.
- Lần đầu cần Internet để tải SBERT và dữ liệu; chờ model chuẩn bị xong trước khi tìm kiếm.
- Nếu cổng 8000 bận, thêm `--port 8001` rồi mở http://127.0.0.1:8001.