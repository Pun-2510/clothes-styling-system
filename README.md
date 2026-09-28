# Hệ thống gợi ý sản phẩm thời trang

Đự Án Công nghệ thông tin

## Giới thiệu

Dự án xây dựng hệ thống gợi ý sản phẩm thời trang đa phương thức bằng cách kết
hợp thông tin hình ảnh và mô tả văn bản. Hệ thống đề xuất các sản phẩm phù hợp
hoặc tương tự dựa trên ảnh hay nội dung mô tả do người dùng cung cấp.

## Công nghệ sử dụng

- Python
- PyTorch
- OpenCLIP
- Sentence-BERT
- ResNet50

## Quy trình chạy dự án sau khi tải hoặc cập nhật mã nguồn

Các lệnh dưới đây chạy bằng PowerShell trên Windows.

> **Lưu ý:** `model.safetensors` không được lưu trên GitHub vì có dung lượng
> khoảng 577 MB. Người dùng mới tải kho mã nguồn cần tinh chỉnh lại mô hình
> trước khi xây dựng và chạy hệ thống bằng Docker.

### 1. Tải mới hoặc cập nhật mã nguồn

Tải kho mã nguồn lần đầu:

```powershell
git clone https://github.com/Pun-2510/clothes-styling-system.git
Set-Location .\clothes-styling-system\project
```

Nếu kho mã nguồn đã có trên máy, chạy từ thư mục gốc của kho mã nguồn:

```powershell
git pull origin main
Set-Location .\project
```

Tất cả lệnh tiếp theo phải được chạy trong thư mục `project/`.

### Tùy chọn: pipeline theo số lượng sản phẩm

Đây là luồng thí nghiệm độc lập, không phải bước bắt buộc để chạy website.
Sau khi chuẩn bị dataset và Python environment, code và số liệu compare nằm tại
`project/comparisons/`; hướng dẫn đầy đủ ở [comparisons/README.md](project/comparisons/README.md).

Chọn đúng 4.000 sản phẩm, dùng checkpoint đã fine-tune, sinh embeddings cho cả
pretrained CLIP và fine-tuned CLIP rồi so sánh Precision, Recall, F1:

```powershell
.\.venv\Scripts\python.exe -m comparisons.run_pipeline `
  --num-products 4000 `
  --reuse-run runs/clip_finetune_3epochs_local `
  --output-dir comparisons/results/clip_4000
```

Run được dùng lại phải có checkpoint và metadata đầy đủ. Nếu cần huấn luyện mới,
bỏ `--reuse-run runs/clip_finetune_3epochs_local`, thêm `--epochs 3` và chọn thư mục
kết quả mới. Có thể thêm `--categories "Backpacks" "Handbags"` (tên phải khớp dữ
liệu), `--seed 42` hoặc `--dry-run` để chỉ xem lệnh, chưa thực thi.

4.000 là số sản phẩm trong catalog thí nghiệm. Compare chỉ dùng tập test chưa tham gia huấn luyện;
khi dùng lại run, đó là phần giao của test cũ với catalog mới, không phải cả 4.000.
Kết quả và log nằm trong `comparisons/results/clip_4000/`; không ghi đè run cũ.

Nếu muốn xem thử catalog thí nghiệm trên web (tùy chọn, không phải mặc định):

```powershell
docker compose --env-file comparisons/results/clip_4000/website.env up -d --build
```

Vẫn chỉ một cấu hình Docker Compose. Không truyền `--env-file` thì đường dẫn mặc
định ở các bước bên dưới vẫn được dùng như trước.

### 2. Chuẩn bị bộ dữ liệu

Tải [Bộ dữ liệu ảnh và văn bản sản phẩm thời trang thu gọn](https://www.kaggle.com/datasets/nirmalsankalana/mini-product-image-and-text-dataset)
và giải nén theo cấu trúc:

```text
project/
└── data/
    ├── data.csv
    └── data/
        ├── 3238.jpg
        └── ...
```

Kiểm tra bộ dữ liệu:

```powershell
Test-Path .\data\data.csv
Test-Path .\data\data
```

Cả hai lệnh phải trả về `True`.

### 3. Tạo môi trường Python

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Nếu `.venv` đã tồn tại thì không cần tạo lại.

### 4. Tiền xử lý dữ liệu

```powershell
.\.venv\Scripts\python.exe -m src.prepare_dataset
```

Tệp kết quả:

```text
data/processed/products.csv
```

### 5. Tinh chỉnh CLIP trong 3 vòng lặp huấn luyện

Kho mã nguồn có lưu kết quả báo cáo của lần chạy trước nhưng không chứa tệp
trọng số lớn. Khi cần tái tạo mô hình từ đầu, xóa thư mục kết quả mẫu đã tải:

```powershell
Remove-Item -Recurse -Force .\runs\clip_finetune_3epochs_local
```

Chạy tinh chỉnh toàn bộ mô hình trong 3 vòng lặp huấn luyện:

```powershell
.\.venv\Scripts\python.exe -m src.finetune_clip `
  --run-dir runs/clip_finetune_3epochs_local `
  --trainable full `
  --epochs 3 `
  --batch-size 16 `
  --learning-rate 1e-6 `
  --weight-decay 0.01 `
  --seed 42
```

Tham số `--epochs 3` yêu cầu chương trình chạy ba vòng huấn luyện. Sau mỗi vòng,
mô hình được đánh giá trên tập xác thực. Bộ trọng số có trung bình Recall@1 tốt
nhất được lưu tại:

```text
runs/clip_finetune_3epochs_local/best/
```

Nếu GPU thiếu bộ nhớ, giảm `--batch-size` xuống `4` hoặc `2`. Nếu không có
CUDA, chương trình tự động sử dụng CPU nhưng thời gian huấn luyện sẽ lâu hơn.

### 6. So sánh mô hình tiền huấn luyện và mô hình đã tinh chỉnh

```powershell
.\.venv\Scripts\python.exe -m src.compare_clip `
  --run-dir runs/clip_finetune_3epochs_local `
  --ks 1 5 10
```

Kết quả được lưu tại:

```text
runs/clip_finetune_3epochs_local/comparison.csv
runs/clip_finetune_3epochs_local/comparison.json
```

### 7. Tạo đặc trưng nhúng bằng bộ trọng số đã tinh chỉnh

```powershell
.\.venv\Scripts\python.exe -m src.generate_clip_embeddings `
  --model runs/clip_finetune_3epochs_local/best `
  --output-dir embeddings_finetuned
```

Các tệp được tạo:

```text
embeddings_finetuned/image_embeddings.npy
embeddings_finetuned/image_embeddings.npy.json
embeddings_finetuned/text_embeddings.npy
embeddings_finetuned/text_embeddings.npy.json
```

### 8. Kiểm tra đầy đủ tệp trước khi chạy trang web

```powershell
Test-Path .\data\processed\products.csv
Test-Path .\runs\clip_finetune_3epochs_local\best\model.safetensors
Test-Path .\embeddings_finetuned\image_embeddings.npy
Test-Path .\embeddings_finetuned\text_embeddings.npy
```

Tất cả lệnh phải trả về `True`.

### 9. Xây dựng và chạy trang web bằng Docker

Chạy từ `project/`: `docker-compose up -d --build` (hoặc `docker compose up -d --build`).
Compose tự đọc `.env` để chọn bộ artifact, không cần thêm `--env-file` mỗi lần.
Mặc định `.env` trỏ tới `data/processed/`, `embeddings_finetuned/` và checkpoint
đã fine-tune trong `runs/`, độc lập với `comparisons/`. File `.env` không lưu trên Git; khi clone sang máy khác,
có thể sao chép `.env.example` thành `.env` rồi chỉnh đường dẫn tới artifact
đã chuẩn bị. Đây chỉ là chọn dữ liệu chạy web, không tự huấn luyện/sinh embeddings.

Compose hiện gồm frontend, Nginx và hai API instance (`backend`, `backend_2`).
Nginx nhận cổng 8000, phân phối theo số kết nối đang hoạt động và thử instance
còn lại khi gặp lỗi kết nối/502/503/504. Frontend vẫn ở cổng 5173.
Mỗi backend nạp model/embeddings riêng vào RAM; không đảm bảo nhanh gấp đôi
khi cùng dùng CPU một máy. Chưa thêm Redis/cache truy vấn.

Code chạy web đã gom tại `project/web/` (backend, frontend, Nginx). Compose và
`.env` vẫn ở `project/` để giữ nguyên lệnh chạy; các lệnh ML `python -m src...` không đổi.
Xem [hướng dẫn web](project/web/README.md), [hướng dẫn project](project/README.md)
và [Nginx, thử dự phòng](project/web/nginx/README.md).
Lệnh `--env-file .../website.env` vẫn áp dụng cho cả hai backend.

Khởi động Docker Desktop, sau đó kiểm tra bộ máy Docker và Docker Compose:

```powershell
docker version
docker compose version
docker compose config --quiet
```

Xây dựng và khởi động giao diện người dùng cùng máy chủ API:

```powershell
docker compose up -d --build
```

Những lần chạy tiếp theo, nếu không thay đổi Dockerfile hoặc các thư viện phụ thuộc:

```powershell
docker compose up -d
```

Kiểm tra vùng chứa và theo dõi nhật ký máy chủ API:

```powershell
docker compose ps
docker compose logs -f nginx backend backend_2
```

Các địa chỉ truy cập:

- Trang web: <http://localhost:5173>
- Tài liệu API (Swagger UI): <http://localhost:8000/docs>
- Trạng thái hoạt động: <http://localhost:8000/api/health>
- Trạng thái sẵn sàng: <http://localhost:8000/api/health/ready>

Dừng hệ thống:

```powershell
docker compose down
```

Lệnh Docker đúng là `docker compose up -d --build`, trong đó `--build` dùng
hai dấu gạch ngang ASCII.
