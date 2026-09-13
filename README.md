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
py -3.12 -m venv .venv
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
docker compose logs -f backend
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
