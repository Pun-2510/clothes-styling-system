# Dataset thống nhất: Mini Fashion của project

Các bộ nạp trong `SBERT_Model`, `ResNet_Model`, `combine_model`, `Web_Test`, `compare_models.py` và `color_benchmark.py` dùng cấu hình chung tại `mini_fashion.py`. Không tự chuyển sang `ashraq/fashion-product-images-small` hoặc `nreimers/fashion-dataset` nếu thiếu dữ liệu.

## Chọn thư mục dữ liệu

Mặc định dùng `project/` bên cạnh `model/`, với cấu trúc giống project CLIP:

```text
project/
  data/processed/products.csv
  data/data/<ảnh sản phẩm>
  runs/clip_finetune_3epochs_local/dataset.csv
  runs/clip_finetune_3epochs_local/best/
```

Nếu dataset đang ở bản DACNTT, cấu hình một lần từ thư mục gốc repository:

```powershell
& C:/DACNTT/model/.venv/Scripts/python.exe -B model/mini_fashion.py --project-dir C:/DACNTT/project --save-local
```

Lệnh kiểm tra catalog và ảnh rồi ghi `model/mini_fashion.local.json` (đã ignore). Nó **dùng chung dữ liệu tại đường dẫn này**, không sao chép hàng nghìn ảnh hay sửa CSV gốc. Trên máy hiện tại đã cấu hình đường dẫn này.

Hoặc đặt biến môi trường có ưu tiên cao hơn cấu hình local:

```powershell
$env:MINI_FASHION_PROJECT = 'C:/DACNTT/project'
```

Máy mới có dữ liệu đầy đủ ở `project/` thì không cần cấu hình. Nếu chưa có dữ liệu, chuẩn bị theo README của project; adapter chỉ nhận CSV đã xử lý, không nhận trực tiếp `data.csv` thô. Thiếu file/ảnh sẽ báo lỗi rõ ràng.

## Dữ liệu từng luồng

| Luồng | Nguồn |
| --- | --- |
| Demo SBERT, Combine, Web_Test | Toàn bộ `data/processed/products.csv` |
| Training ResNet/BERT classifier | Chỉ split `train` trong `runs/.../dataset.csv`; script có thể chia validation nội bộ từ phần train này |
| Benchmark chung và thử tag màu | Mặc định split `test` của cùng `dataset.csv` |

Catalog project hiện có 5.015 sản phẩm/142 danh mục. Thí nghiệm đã lưu loại 11 ảnh trùng, còn 4.036 train + 484 validation + 484 test. Không tạo lại split khi đổi đường dẫn. ResNet training vẫn giới hạn sáu lớp theo kiến trúc cũ; catalog tìm kiếm dùng toàn bộ danh mục. Bộ lọc danh mục từ classifier sáu lớp trong Combine mặc định tắt vì không bao phủ toàn bộ Mini Fashion; có thể bật lại bằng `FILTER_BY_CATEGORY=true` khi phù hợp.

Web_Test mặc định dùng toàn catalog; `WEB_MAX_PRODUCTS` có thể giới hạn để demo nhanh. Combine dùng toàn catalog; `MAX_IMAGES` có thể giới hạn. SBERT mặc định không cắt còn 5.000 dòng. Nếu chủ động đặt giới hạn khác nhau thì các demo không còn cùng gallery; dùng benchmark để so sánh công bằng.

Adapter giữ nguyên product ID và bổ sung alias cho code cũ:

| Cột project | Alias tương thích |
| --- | --- |
| `product_name` | `productDisplayName` |
| `category` | `articleType` |
| `image_path` | `image` (đường dẫn ảnh local) |
| `color_tags` nếu có | `baseColour` |

Không đoán brand/gender/màu còn thiếu. Tag màu tự động vẫn được tạo riêng bởi `color_benchmark.py`, không ghi đè CSV project. Đường dẫn ảnh cũ trên máy khác được định vị lại bằng tên file trong `data/data/`.

## Chạy thử tag màu bằng nguồn mặc định mới

```powershell
& C:/DACNTT/model/.venv/Scripts/python.exe -B model/color_benchmark.py --prepare-only
```

Chạy đủ các model, chỉ cần bổ sung checkpoint ResNet nếu nó nằm ngoài repository:

```powershell
& C:/DACNTT/model/.venv/Scripts/python.exe -B model/color_benchmark.py `
  --resnet-checkpoint C:/DACNTT/model/Web_Test/models/resnet_outfit.pth
```

CLIP fine-tune mặc định lấy checkpoint từ project đã chọn. `--csv`, `--image-dir`, `--clip-checkpoint` vẫn có thể ghi đè. Với CSV tùy chỉnh, truyền `--image-dir` nếu đường dẫn ảnh trong CSV không còn dùng được.

**Lần chạy `color_tags_pilot` trước khi đổi loader đã dùng chính Mini Fashion test này**, nên không cần đổi tên dataset hay tạo lại số liệu đó. Đây là thay đổi nguồn mặc định cho các script cũ. Checkpoint cũ không tự trở thành model đã train trên Mini Fashion; muốn kết luận như vậy cần train lại và ghi nhận nguồn checkpoint mới.

SBERT dùng tên cache mới kèm hash danh sách ID/văn bản/model; Web_Test đưa catalog và hash ảnh vào cache key. Embedding cũ được giữ nguyên nhưng không được dùng cho catalog mới. `combine_model/test.py` là thử nghiệm cũ có giao thức riêng; dùng `color_benchmark.py --models ... combine combine_trained` cho so sánh màu có kiểm soát. Việc đổi loader không sửa toàn bộ giao diện gọi hàm cũ trong `combine_search.py`.

Xem [COLOR_BENCHMARK.md](COLOR_BENCHMARK.md) về cách review nhãn và đọc kết quả. Bước chuyển dataset không tự chạy training, không tải thêm dataset từ Internet.
