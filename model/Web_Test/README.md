# Website kiểm thử SBERT + ResNet18

Website hỗ trợ ba chế độ trên cùng catalog sản phẩm:

- **Mô tả:** SBERT mã hóa câu truy vấn và tìm bằng cosine similarity.
- **Ảnh:** ResNet18 mã hóa ảnh tải lên và tìm bằng cosine similarity.
- **Kết hợp:** Weighted Reciprocal Rank Fusion (RRF) hợp nhất thứ hạng SBERT
  và ResNet18. Thanh trượt trên giao diện điều chỉnh trọng số hai model.

Model BERT classifier trong `models/bert_fashion_model/` không thuộc pipeline
hai model này và không được bật trên giao diện. Bộ artifact hiện tại chỉ chứa
config/tokenizer, không chứa trọng số BERT hoàn chỉnh.

## Cài đặt và chạy

Cài Python 3.11. Mở PowerShell tại `model/Web_Test`:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:WEB_CPU_THREADS = "4"
.\.venv\Scripts\python.exe app.py --warmup combined
```

Mở **http://127.0.0.1:8000**. `--warmup combined` nạp cả SBERT và ResNet18,
sau đó tạo hai index embedding trong nền. Có thể mở web ngay; trạng thái chuẩn
bị model được hiển thị trên giao diện. Nhấn `Ctrl+C` để dừng.

Muốn tạo và kiểm tra cache trước rồi thoát:

```powershell
.\.venv\Scripts\python.exe app.py --prepare-only --warmup combined
```

Chạy self-test bằng model thật cho đủ ba đường text/image/combined:

```powershell
.\.venv\Scripts\python.exe app.py --self-test
```

Lệnh trả JSON, kiểm tra model đã tải, score hữu hạn, số kết quả và vị trí của
sản phẩm truy vấn trong top-k. Exit code khác `0` khi self-test thất bại.
Endpoint `/api/status` cũng trả trạng thái từng dependency và cho biết SBERT,
ResNet18 đã được nạp hay chưa.

## Cấu hình hai model

Các biến môi trường quan trọng:

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `WEB_SBERT_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Hugging Face ID hoặc đường dẫn SBERT local |
| `WEB_RESNET_MODEL` | `models/resnet_outfit.pth` | Checkpoint ResNet18 đã train |
| `WEB_DEVICE` | `auto` | `auto`, `cpu` hoặc `cuda` |
| `WEB_MAX_PRODUCTS` | `3000` | Số sản phẩm lập index, tối đa 3000 |
| `WEB_BATCH_SIZE` | `32` | Batch khi sinh embedding |
| `WEB_CACHE_DIR` | `.cache` | Nơi lưu index embedding theo fingerprint model/catalog |
| `WEB_DISK_CACHE` | `1` | Đặt `0` để không đọc/ghi cache embedding |
| `WEB_TEXT_MULTIVIEW` | `1` | Trung bình embedding mô tả đầy đủ và tên/category ngắn |
| `WEB_RRF_K` | `60` | Hằng số làm trơn của weighted RRF |
| `WEB_IMAGE_TTA` | `1` | Dùng horizontal-flip TTA nhất quán cho catalog và ảnh query |
| `WEB_COLOR_WEIGHT` | `0.10` | Trọng số màu trong image search |
| `WEB_CATEGORY_WEIGHT` | `0.15` | Soft prior category của ResNet18 |
| `WEB_GENDER_WEIGHT` | `0.05` | Soft prior giới tính khi confidence đủ cao |
| `WEB_REMOVE_BACKGROUND` | `0` | Đặt `1` để bật rembg cho ảnh upload |
| `WEB_PATTERN_SEARCH` | `0` | Đặt `1` để index nhiều crop nhằm giữ chi tiết họa tiết |

Ví dụ chạy CPU cân bằng tốc độ/RAM:

```powershell
$env:WEB_DEVICE = "cpu"
$env:WEB_CPU_THREADS = "4"
$env:WEB_BATCH_SIZE = "32"
$env:WEB_MAX_PRODUCTS = "3000"
.\.venv\Scripts\python.exe app.py --warmup combined
```

Ví dụ chạy CUDA:

```powershell
$env:WEB_DEVICE = "cuda"
$env:WEB_BATCH_SIZE = "64"
.\.venv\Scripts\python.exe app.py --warmup combined
```

Khi thay checkpoint, SBERT, catalog hoặc cấu hình pattern, ứng dụng tạo cache
mới theo fingerprint thay vì dùng nhầm embedding cũ. Không cần xóa `.cache`
trong trường hợp bình thường.

SBERT dùng câu catalog có nhãn trường (`Product`, `Category`, `Colour`,
`Gender`, `Brand`, `Usage`, `Season`) và bổ sung alias tiếng Anh cho các thuộc
tính thời trang tiếng Việt phổ biến. ResNet giữ tỷ lệ ảnh khi resize và dùng
cùng quy tắc TTA cho ảnh catalog lẫn ảnh upload; hai thay đổi này làm cache cũ
tự hết hiệu lực.

## Kiểm tra ResNet18

Khi mở web, bảng **Kiểm thử tự động ResNet18** chạy trên tập audit đã lưu nếu
có. Có thể đổi số ảnh benchmark bằng `WEB_RESNET_BENCHMARK_SAMPLES` (mặc định
120). Checkpoint trained dùng trong Combine là file do `WEB_RESNET_MODEL` chỉ
định.

Ảnh tải lên được chuẩn hóa orientation, transparency và kích thước. Nếu bật
`WEB_REMOVE_BACKGROUND=1`, ảnh còn được xóa nền trước khi ResNet phân loại và
tạo embedding. Mặc định dùng model nhẹ `u2netp`; đổi bằng
`WEB_BACKGROUND_MODEL`.

## Lưu ý

- Giữ nguyên `models/resnet_outfit.pth` và `models/resnet_gender.pth`. Nếu clone
  từ Git, cài Git LFS rồi chạy `git lfs install` và `git lfs pull`.
- Lần đầu cần Internet để tải SBERT và dataset
  `ashraq/fashion-product-images-small`; những lần sau dùng cache local.
- Chế độ kết hợp cần cả mô tả và ảnh. Text-only và image-only vẫn hoạt động độc
  lập nếu chỉ muốn nạp một model.
- Nếu cổng 8000 bận, thêm `--port 8001` và mở
  **http://127.0.0.1:8001**.
