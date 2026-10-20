# HƯỚNG DẪN CHẠY HỆ THỐNG GỢI Ý SẢN PHẨM THỜI TRANG

Tất cả câu lệnh trong tài liệu này phải được chạy từ thư mục `project/`.

## 1. WORKFLOW TỔNG QUÁT

Quy trình xây dựng artifact đầy đủ:

```text
Dataset gốc
   ↓
Prepare dataset + audit category tự động
   ↓
Fine-tune CLIP và chọn checkpoint tốt nhất
   ↓
So sánh pretrained CLIP với fine-tuned CLIP (tùy chọn)
   ↓
Sinh image/text embeddings bằng checkpoint fine-tuned
   ↓
Cấu hình .env
   ↓
docker compose up -d --build
```

Website không lấy dữ liệu từ kết quả thí nghiệm. Website sử dụng trực tiếp:

```text
data/processed/products.csv
data/data/
embeddings/
runs/clip_finetune_3epochs/best/
```

## 2. ĐIỀU KIỆN TIÊN QUYẾT

- Python 3.12.
- Docker Desktop và Docker Compose.
- Internet trong lần đầu tải thư viện, Docker image và model Hugging Face.
- Dataset Mini Fashion Product Images and Text Dataset:
  https://www.kaggle.com/datasets/nirmalsankalana/mini-product-image-and-text-dataset

Đặt dataset theo cấu trúc:

```text
project/
└── data/
    ├── data.csv
    └── data/
        ├── 3238.jpg
        └── ...
```

Kiểm tra:

```powershell
Test-Path .\data\data.csv
Test-Path .\data\data
```

Cả hai lệnh phải trả về `True`.

## 3. TẠO MÔI TRƯỜNG PYTHON

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Các lệnh sau gọi trực tiếp Python trong `.venv`, vì vậy không bắt buộc chạy
`Activate.ps1`.

## 4. CHẠY NHANH BẰNG ARTIFACT ĐÃ CHUẨN BỊ

Nếu chỉ cần chạy website, có thể tải model fine-tuned và các file cần thiết từ
Google Drive thay vì tự fine-tune:

```text
https://drive.google.com/file/d/1_xiQjMBlf5EMZVRflDwC2G5kbgEkwy00/view?usp=drive_link
```

Gói artifact cần cung cấp các thành phần đồng bộ:

```text
data/processed/products.csv
embeddings/
├── image_embeddings.npy
├── image_embeddings.npy.json
├── text_embeddings.npy
└── text_embeddings.npy.json
runs/clip_finetune_3epochs/
└── best/
    ├── config.json
    ├── model.safetensors
    ├── processor_config.json
    ├── tokenizer.json
    └── tokenizer_config.json
```

Gói artifact không nhất thiết chứa ảnh catalog. Vẫn phải đặt ảnh trong
`data/data/` để website hiển thị sản phẩm. Không chạy lại prepare nếu muốn sử
dụng embeddings tải sẵn, vì thay đổi `products.csv` sẽ làm checksum không còn
khớp.

Sau khi giải nén đúng cấu trúc, chuyển đến mục 9 để cấu hình `.env`, rồi chạy
Docker ở mục 10.

## 5. PREPARE DATASET VÀ AUDIT CATEGORY

Ví dụ chuẩn bị đúng 5.000 sản phẩm và cố gắng lấy tối thiểu 10 sản phẩm cho mỗi
category:

```powershell
.\.venv\Scripts\python.exe -m src.prepare_dataset `
  --csv data/data.csv `
  --image-dir data/data `
  --num-products 5000 `
  --min-per-category 10 `
  --seed 42 `
  --output data/processed/products.csv
```

Prepare tự động tạo:

```text
data/processed/products.csv
data/processed/products.preparation.json
data/processed/products.category_audit.csv
data/processed/products.category_audit.json
```

`products.csv` là dữ liệu đầu vào chính và đã chứa cột `category`. Hai file
`category_audit` là báo cáo phân bố, không phải input riêng của CLIP. Category có
ít dữ liệu nguồn hơn mức tối thiểu sẽ được ghi cảnh báo; chương trình không tạo
ảnh giả để bù dữ liệu.

Xem các category chưa đạt mức tối thiểu:

```powershell
Import-Csv data/processed/products.category_audit.csv |
  Where-Object status -ne "ok" |
  Sort-Object selected_count |
  Format-Table category,source_count,selected_count,status,warnings
```

Audit lại một catalog có sẵn mà không chạy prepare:

```powershell
.\.venv\Scripts\python.exe -m src.audit_dataset `
  --dataset-csv data/processed/products.csv `
  --minimum 10
```

## 6. FINE-TUNE CLIP

Mỗi `run-dir` chỉ đại diện cho một lần thí nghiệm và không được ghi đè. Nếu thư
mục `runs/clip_finetune_3epochs` đã tồn tại, hãy sao lưu hoặc đổi tên nó trước
khi huấn luyện lại.

```powershell
.\.venv\Scripts\python.exe -m src.finetune_clip `
  --csv data/processed/products.csv `
  --run-dir runs/clip_finetune_3epochs `
  --trainable full `
  --epochs 3 `
  --batch-size 16 `
  --learning-rate 1e-6 `
  --weight-decay 0.01 `
  --category-balance-power 0.5 `
  --max-category-sample-weight 5 `
  --seed 42
```

Trong đó:

- `category-balance-power 0.5` dùng cân bằng căn bậc hai theo tần suất category.
- `max-category-sample-weight 5` giới hạn trọng số lấy mẫu tối đa ở mức 5 lần.
- Các tham số này chỉ điều chỉnh sampling trong train, không tạo thêm ảnh.
- Validation chọn checkpoint tốt nhất theo Recall@1 trung bình của
  text-to-image và image-to-text.
- Test không tham gia cập nhật trọng số hoặc chọn epoch.

Kết quả chính:

```text
runs/clip_finetune_3epochs/
├── dataset.csv
├── dataset.category_audit.csv
├── dataset.category_audit.json
├── experiment.json
├── training.json
└── best/
```

`dataset.csv` chứa dữ liệu và cột `split` cho train/validation/test. Các file
`dataset.category_audit` dùng để kiểm tra phân bố của từng split.

Nếu GPU thiếu bộ nhớ, giảm `--batch-size` xuống `8`, `4` hoặc `2`. Khi không có
CUDA, chương trình tự sử dụng CPU nhưng thời gian huấn luyện sẽ lâu hơn.

## 7. SO SÁNH MODEL (TÙY CHỌN)

Đây là bước đánh giá phục vụ báo cáo, không phải dữ liệu đầu vào của
website. Pipeline so sánh thống nhất nằm ngoài `project/`, do đó hãy chạy lệnh sau
từ thư mục gốc của repository:

```powershell
Set-Location ..
.\project\.venv\Scripts\python.exe -m experiments.model_comparison.compare_full_gallery `
  --run-dir .\project\runs\clip_finetune_3epochs `
  --catalog-csv .\project\data\processed\products.csv `
  --expected-gallery-size 5000 `
  --resnet-source project-trained `
  --device cpu `
  --batch-size 16 `
  --ks 1 5 10 `
  --clip-image-weight 0.5 `
  --clip-text-image-weight 0.5 `
  --output-dir .\experiments\model_comparison\results\full_gallery_5000_v3
Set-Location .\project
```

Kết quả được lưu trong `experiments/model_comparison/results/`. Pipeline dùng cùng
catalog 5.000 sản phẩm, test query, giá trị K và checkpoint cho các nhóm so sánh.
Xem chi tiết tại `experiments/model_comparison/README.md` ở cấp repository.

## 8. SINH EMBEDDINGS CHO WEBSITE

Sau khi fine-tuning hoàn tất, sinh lại cả image và text embeddings bằng đúng
checkpoint và đúng `products.csv`:

```powershell
.\.venv\Scripts\python.exe -m src.generate_clip_embeddings `
  --csv data/processed/products.csv `
  --model runs/clip_finetune_3epochs/best `
  --output-dir embeddings `
  --batch-size 16
```

Kết quả:

```text
embeddings/image_embeddings.npy
embeddings/image_embeddings.npy.json
embeddings/text_embeddings.npy
embeddings/text_embeddings.npy.json
```

Không xóa các file `.npy.json`; backend dùng metadata này để phát hiện trường
hợp model, catalog và embeddings không đồng bộ.

## 9. CẤU HÌNH `.env`

Tạo `.env` từ file mẫu nếu chưa có:

```powershell
Copy-Item .env.example .env
```

Cấu hình chính:

```env
FASHION_CATALOG_DIR=./data/processed
FASHION_IMAGE_DIR=./data/data
FASHION_EMBEDDINGS_DIR=./embeddings
FASHION_CHECKPOINT_DIR=./runs/clip_finetune_3epochs/best
CATEGORY_CONFIDENCE_FLOOR=0.50
CATEGORY_CONFIDENCE_THRESHOLD=0.60
CATEGORY_BONUS=0.05
CATEGORY_TOP_PER_CLASS=3
TEXT_RETRIEVAL_IMAGE_WEIGHT=0.50
COMBINED_IMAGE_WEIGHT=0.50
COMBINED_CATEGORY_WEIGHT=0.10
COMBINED_ATTRIBUTE_WEIGHT=0.25
```

`FASHION_CATALOG_DIR` và `FASHION_IMAGE_DIR` phải được giữ lại. Chúng lần lượt
cung cấp catalog và ảnh sản phẩm cho backend.

## 10. CHẠY WEBSITE BẰNG DOCKER

Kiểm tra artifact trước khi chạy:

```powershell
Test-Path .\data\processed\products.csv
Test-Path .\data\data
Test-Path .\embeddings\image_embeddings.npy
Test-Path .\embeddings\text_embeddings.npy
Test-Path .\runs\clip_finetune_3epochs\best\model.safetensors
docker compose config --quiet
```

Khởi động toàn bộ hệ thống bằng một lệnh:

```powershell
docker compose up -d --build
```

Các thành phần được chạy:

```text
web/frontend       ReactJS
web/nginx          Nginx reverse proxy/load balancer
backend            FastAPI instance 1
backend_2          FastAPI instance 2
```

Địa chỉ:

```text
Frontend:    http://localhost:5173
API qua Nginx: http://localhost:8000
Swagger:     http://localhost:8000/docs
```

Kiểm tra trạng thái và log:

```powershell
docker compose ps
docker compose logs -f nginx backend backend_2
Invoke-RestMethod http://localhost:8000/api/health
Invoke-RestMethod http://localhost:8000/api/health/ready
```

Dừng hệ thống:

```powershell
docker compose down
```

Không dùng `docker compose down -v` nếu muốn giữ cache model Hugging Face.

## 11. CATEGORY MODE

Backend hỗ trợ:

- `soft_category`: category bonus tăng dần theo confidence. Confidence bằng hoặc
  thấp hơn floor thì fallback về `no_category`; confidence từ floor đến
  threshold nhận bonus từ 0% đến 100%.
- `no_category`: chỉ dùng độ tương đồng CLIP.
- `hard_category`: giới hạn ứng viên theo category, chủ yếu dành cho thử nghiệm.

Frontend không hiển thị nút chọn category mode và luôn gửi `soft_category`.
Backend tự tính category strength từ confidence, không còn chỉ bật/tắt
category tại một ngưỡng duy nhất.
Hai mode `no_category` và `hard_category` vẫn được giữ trong API để kiểm thử và
thực hiện thí nghiệm, không dành cho thao tác thông thường trên giao diện.

## 12. TÌM KIẾM KẾT HỢP ẢNH VÀ MÔ TẢ

Frontend sử dụng một form duy nhất và tự chọn luồng xử lý theo input: chỉ có mô
tả thì gọi text search, chỉ có ảnh thì gọi image search, còn có đủ cả hai thì
gọi `POST /api/recommendations/combined`. Khi dùng cả ảnh và mô tả, người dùng
có thể chọn tỷ trọng của hai nguồn thông tin và backend tính:

```text
final_score = image_weight * visual_score
            + (1 - image_weight) * text_score
```

`visual_score` là cosine similarity giữa ảnh truy vấn và ảnh catalog.
`text_score` kết hợp cosine similarity text-to-image với text-to-text khi đã
có `text_embeddings.npy`. Website kết hợp trực tiếp các raw cosine score sau
khi nhân trọng số.

Khi văn bản nêu category rõ ràng, hệ thống dùng category đó làm candidate pool.
Nếu category trong ảnh và văn bản tương thích, hệ thống giữ subtype từ ảnh, ví
dụ `Shirts` trong text và `Tshirts` trong ảnh sẽ dùng `Tshirts`. Các thuộc tính
màu sắc và họa tiết trong text được xem là yêu cầu ghi đè. Vì vậy ảnh mũ xanh
kết hợp `red cap` sẽ ưu tiên mũ đỏ thuộc `Caps`, còn ảnh áo trơn kết hợp
`patterned T-shirt` sẽ ưu tiên áo có printed/graphic/striped metadata.

## 13. KIỂM THỬ

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Build frontend riêng:

```powershell
Set-Location web/frontend
npm ci
npm run build
Set-Location ../..
```

## 14. LỖI THƯỜNG GẶP

- `unrecognized arguments`: đang chạy source cũ hoặc chạy nhầm bản project.
- Dataset changed: `products.csv` không còn khớp với metadata của run cũ.
- Embedding shape/checksum mismatch: tạo lại cả image và text embeddings từ
  cùng một `products.csv` và checkpoint.
- HTTP 502: kiểm tra readiness và log của cả hai backend.
- Không tìm thấy Docker daemon: mở Docker Desktop và chờ Linux engine sẵn sàng.
- Cổng 8000 hoặc 5173 đang được sử dụng: dừng tiến trình/container cũ trước khi
  khởi động lại.
