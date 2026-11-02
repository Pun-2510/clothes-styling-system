# HƯỚNG DẪN CHẠY HỆ THỐNG HIỆN TẠI

Tài liệu này mô tả bộ artifact diverse seed 42 đang được website sử dụng.
Trừ mục thí nghiệm được ghi rõ chạy từ thư mục gốc `dacntt/`, mọi lệnh còn lại
được chạy từ `dacntt/project/`.

## 1. CẤU HÌNH ĐANG TRIỂN KHAI

```text
Catalog:     data/processed_diverse/products.csv
Images:      data/data/
Checkpoint:  runs/clip_finetune_diverse_seed_42/best/
Embeddings:  embeddings_diverse_seed_42/
Weights:     ../experiments/model_comparison/results/diverse_seed_42/retrieval_weights.json
```

Các thành phần này được tạo từ cùng catalog seed 42. Không thay riêng một file
`products.csv`, checkpoint, embedding hoặc weight JSON bằng artifact của seed
khác vì backend kiểm tra model/catalog checksum khi khởi động.

Workflow:

```text
Dataset gốc
   ↓
Loại ảnh lỗi/trùng + diverse sampling + category audit
   ↓
Catalog 5.000 sản phẩm
   ↓
Fine-tune CLIP và chọn best checkpoint trên validation
   ↓
Sinh image/text embeddings
   ↓
Chọn retrieval weights trên validation
   ↓
Đánh giá test trên seed 42, 123 và 2026
   ↓
Docker Compose: React + Nginx + 2 FastAPI backends
```

## 2. ĐIỀU KIỆN TIÊN QUYẾT

- Python 3.12 hoặc 3.13.
- Docker Desktop và Docker Compose.
- Internet trong lần đầu tải thư viện, image và Hugging Face models.
- Mini Fashion Product Images and Text Dataset:
  https://www.kaggle.com/datasets/nirmalsankalana/mini-product-image-and-text-dataset

Đặt dữ liệu như sau:

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

## 3. TẠO MÔI TRƯỜNG PYTHON

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Không bắt buộc chạy `Activate.ps1` vì các lệnh gọi trực tiếp Python trong venv.

## 4. CHẠY NHANH BẰNG ARTIFACT CÓ SẴN

Liên kết artifact hiện có:

```text
https://drive.google.com/file/d/1_xiQjMBlf5EMZVRflDwC2G5kbgEkwy00/view?usp=drive_link
```

Một bộ artifact đồng bộ cho cấu hình hiện tại cần có:

```text
data/processed_diverse/
├── products.csv
├── products.preparation.json
├── products.category_audit.csv
└── products.category_audit.json

embeddings_diverse_seed_42/
├── image_embeddings.npy
├── image_embeddings.npy.json
├── text_embeddings.npy
└── text_embeddings.npy.json

runs/clip_finetune_diverse_seed_42/
├── dataset.csv
├── experiment.json
├── training.json
└── best/
    ├── config.json
    ├── model.safetensors
    ├── processor_config.json
    ├── tokenizer.json
    └── tokenizer_config.json

../experiments/model_comparison/results/diverse_seed_42/
└── retrieval_weights.json
```

Ảnh catalog vẫn phải nằm trong `data/data/`. Nếu gói tải về sử dụng tên thư mục
cũ, không tự trộn với bộ diverse; hãy sửa toàn bộ năm đường dẫn trong `.env`
theo đúng bộ artifact đó.

Nếu đã có đủ artifact, chuyển thẳng đến mục 9.

## 5. TỰ PREPARE CATALOG DIVERSE SEED 42

```powershell
.\.venv\Scripts\python.exe -m src.prepare_dataset `
  --csv data/data.csv `
  --image-dir data/data `
  --num-products 5000 `
  --min-per-category 25 `
  --sampling-strategy diverse `
  --seed 42 `
  --output data/processed_diverse/products.csv
```

Prepare thực hiện:

- Chuẩn hóa metadata và kiểm tra ảnh.
- Loại ảnh trùng lặp.
- Cố gắng phân bổ tối thiểu 25 sản phẩm cho mỗi category.
- Chọn đại diện khác nhau trong từng category bằng descriptor ảnh nhẹ và
  metadata; đây không phải CLIP embedding.
- Tự sinh category audit.

Kết quả:

```text
data/processed_diverse/products.csv
data/processed_diverse/products.preparation.json
data/processed_diverse/products.category_audit.csv
data/processed_diverse/products.category_audit.json
```

## 6. FINE-TUNE CLIP SEED 42

Mỗi `run-dir` là bất biến và không được ghi đè. Nếu muốn chạy lại, dùng một tên
run mới thay vì xóa kết quả cũ.

```powershell
.\.venv\Scripts\python.exe -m src.finetune_clip `
  --csv data/processed_diverse/products.csv `
  --run-dir runs/clip_finetune_diverse_seed_42 `
  --trainable full `
  --epochs 3 `
  --batch-size 16 `
  --learning-rate 1e-6 `
  --weight-decay 0.01 `
  --category-balance-power 0.5 `
  --max-category-sample-weight 5 `
  --seed 42
```

Validation chọn checkpoint tốt nhất. Test split không được dùng để cập nhật
trọng số hoặc chọn epoch. Checkpoint triển khai nằm tại:

```text
runs/clip_finetune_diverse_seed_42/best/
```

## 7. SINH IMAGE VÀ TEXT EMBEDDINGS

```powershell
.\.venv\Scripts\python.exe -m src.generate_clip_embeddings `
  --csv data/processed_diverse/products.csv `
  --model runs/clip_finetune_diverse_seed_42/best `
  --output-dir embeddings_diverse_seed_42 `
  --batch-size 16
```

Không xóa các file `.npy.json`; backend dùng chúng để kiểm tra tính đồng bộ của
catalog, model và embedding.

## 8. CHỌN RETRIEVAL WEIGHTS TRÊN VALIDATION

Lệnh này chạy từ thư mục gốc repository:

```powershell
Set-Location ..
```

`--weights-output` được chỉ định rõ để không ghi đè file mặc định cũ trong
`project/`:

```powershell
.\project\.venv\Scripts\python.exe -m experiments.model_comparison.tune_serving_weights `
  --run-dir .\project\runs\clip_finetune_diverse_seed_42 `
  --catalog-csv .\project\data\processed_diverse\products.csv `
  --image-embeddings .\project\embeddings_diverse_seed_42\image_embeddings.npy `
  --text-embeddings .\project\embeddings_diverse_seed_42\text_embeddings.npy `
  --expected-gallery-size 5000 `
  --device cpu `
  --batch-size 16 `
  --objective-k 10 `
  --min-combined-image-weight 0.20 `
  --max-combined-image-weight 0.80 `
  --weights-output .\experiments\model_comparison\results\diverse_seed_42\retrieval_weights.json `
  --output-dir .\experiments\model_comparison\results\diverse_seed_42\weight_tuning_rebuild_v1
```

Mỗi lần chạy lại phải dùng một `--output-dir` mới hoặc rỗng. File weight đang
được website sử dụng chứa các giá trị chọn từ validation seed 42:

```text
TEXT_RETRIEVAL_IMAGE_WEIGHT = 0.45
COMBINED_IMAGE_WEIGHT       = 0.80
COMBINED_ATTRIBUTE_WEIGHT   = 0.10
COMBINED_CATEGORY_WEIGHT    = 0.25
CATEGORY_BONUS              = 0.20
```

Các giá trị này là kết quả của seed 42, không phải hằng số áp dụng cho mọi seed.

Quay lại project:

```powershell
Set-Location .\project
```

## 9. CẤU HÌNH `.env`

Repository theo dõi `.env.example`; `.env` là cấu hình cục bộ và bị Git bỏ qua.
Nếu chưa có `.env`:

```powershell
Copy-Item .env.example .env
```

Cấu hình hiện tại:

```env
FASHION_CATALOG_DIR=./data/processed_diverse
FASHION_IMAGE_DIR=./data/data
FASHION_EMBEDDINGS_DIR=./embeddings_diverse_seed_42
FASHION_CHECKPOINT_DIR=./runs/clip_finetune_diverse_seed_42/best
FASHION_RETRIEVAL_WEIGHTS_FILE=../experiments/model_comparison/results/diverse_seed_42/retrieval_weights.json

CATEGORY_CONFIDENCE_FLOOR=0.50
CATEGORY_CONFIDENCE_THRESHOLD=0.60
CATEGORY_TOP_PER_CLASS=3
```

Category confidence hoạt động như sau:

- `confidence <= 0.50`: không cộng category bonus.
- `0.50 < confidence < 0.60`: bonus tăng tuyến tính.
- `confidence >= 0.60`: dùng toàn bộ category bonus trong weight JSON.
- `CATEGORY_TOP_PER_CLASS=3`: mỗi category đóng góp tối đa ba láng giềng mạnh
  khi backend ước lượng category từ ảnh.

## 10. KIỂM TRA ARTIFACT VÀ CHẠY DOCKER

```powershell
Test-Path .\data\processed_diverse\products.csv
Test-Path .\data\data
Test-Path .\embeddings_diverse_seed_42\image_embeddings.npy
Test-Path .\embeddings_diverse_seed_42\text_embeddings.npy
Test-Path .\runs\clip_finetune_diverse_seed_42\best\model.safetensors
Test-Path ..\experiments\model_comparison\results\diverse_seed_42\retrieval_weights.json
docker compose config --quiet
```

Tất cả `Test-Path` phải trả về `True`.

```powershell
docker compose up -d --build
```

Địa chỉ:

```text
Website:       http://localhost:5173
API qua Nginx: http://localhost:8000
Swagger:       http://localhost:8000/docs
```

Kiểm tra:

```powershell
docker compose ps
docker compose logs -f nginx backend backend_2
Invoke-RestMethod http://localhost:8000/api/health
Invoke-RestMethod http://localhost:8000/api/health/ready
```

Sau khi đổi `.env`, checkpoint hoặc weight file:

```powershell
docker compose up -d --force-recreate
```

Dừng hệ thống:

```powershell
docker compose down
```

Không dùng `docker compose down -v` nếu muốn giữ Hugging Face cache.

## 11. LOGIC TRUY XUẤT

Frontend dùng một form duy nhất:

- Chỉ văn bản: text-only retrieval.
- Chỉ ảnh: image-only retrieval và confidence-weighted category bonus.
- Ảnh + văn bản: combined retrieval; văn bản bổ sung category/attribute.

Các cosine component:

```text
S_II: image query  → catalog image
S_TI: text query   → catalog image
S_TT: text query   → catalog text
```

Text score và combined base score:

```text
S_T    = w_t * S_TI + (1 - w_t) * S_TT
S_base = w_i * S_II + (1 - w_i) * S_T
```

Hệ thống dùng raw cosine fusion, không dùng percentile. Sau fusion, attribute
signal và category policy được áp dụng theo weight JSON. Image-only sử dụng
soft category bonus; hard category filter chỉ là baseline thí nghiệm. Với text,
backend chỉ giới hạn candidate pool khi truy vấn nêu đúng một category rõ ràng.
Nếu không có category hoặc có nhiều category độc lập, hệ thống giữ toàn catalog
và giao diện yêu cầu người dùng chọn một loại sản phẩm cụ thể.

## 12. KẾT QUẢ THÍ NGHIỆM HIỆN TẠI

Ba seed diverse đã hoàn tất:

```text
Seed 42:   catalog 5.000, test queries 460
Seed 123:  catalog 5.000, test queries 464
Seed 2026: catalog 5.000, test queries 462
```

Mỗi seed có ba nhóm kết quả:

```text
experiments/model_comparison/results/diverse_seed_<seed>/
├── core_pretrained_multimodal_v1/
├── category_aware_v1/
├── system_pipeline_v1/
└── retrieval_weights.json
```

Kết quả mean ± sample standard deviation:

```text
experiments/model_comparison/results/aggregate_diverse_3_seeds_v1/
├── core_pretrained_multimodal/mean_std_summary.csv
├── category_aware/mean_std_summary.csv
└── system_pipeline/mean_std_summary.csv
```

Thí nghiệm báo cáo Precision, Recall, F1 và NDCG tại `K = 1, 5, 10`.
Lệnh chạy riêng từng seed và lệnh aggregate nằm trong:

```text
../experiments/model_comparison/README.md
```

Không gộp kết quả random sampling cũ với diverse sampling thành sáu seed vì
hai bộ dùng category quota, catalog và test queries khác nhau.

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

- `Output directory is not empty`: dùng một output directory mới cho thí nghiệm.
- `Experiment dataset changed`: catalog không khớp với run; không sửa dataset
  trong một run đã tồn tại.
- Embedding/checksum mismatch: sinh lại cả image và text embeddings bằng cùng
  catalog và checkpoint.
- HTTP 502: kiểm tra readiness và log hai backend.
- HTTP 429: chờ rate-limit window hoặc giảm số request lặp lại.
- Docker không thấy file ngoài `project/`: kiểm tra đường dẫn
  `FASHION_RETRIEVAL_WEIGHTS_FILE` và chạy `docker compose config`.
- Đổi `.env` nhưng container giữ cấu hình cũ: chạy
  `docker compose up -d --force-recreate`.
