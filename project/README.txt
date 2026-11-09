# HƯỚNG DẪN CHẠY HỆ THỐNG TRONG `project/`

Tài liệu này mô tả pipeline vận hành độc lập của website. Sau khi người dùng
clone repository, toàn bộ catalog đã xử lý, checkpoint fine-tuned CLIP,
embeddings, cache validation và retrieval weights đều được tạo bên trong
`project/`. Website không đọc file từ `experiments/`.

## 1. PHÂN TÁCH HỆ THỐNG VÀ THÍ NGHIỆM

```text
<repository>/
├── project/                         Hệ thống chạy thật
│   ├── data/processed/              Catalog đã chuẩn bị và category audit
│   ├── runs/clip_finetune/          Split, log huấn luyện và best checkpoint
│   ├── embeddings/                  Image/text embeddings của catalog
│   ├── validation/                  Cache và bảng thử trọng số validation
│   ├── retrieval_weights.json       Trọng số website sử dụng
│   ├── src/                         Tiền xử lý, training, validation, retrieval
│   ├── web/                         React, FastAPI và Nginx
│   ├── .env                         Cấu hình cục bộ, Git bỏ qua
│   └── docker-compose.yml
└── experiments/                     Đánh giá nghiên cứu, không phục vụ website
```

`seed = 42` trong hướng dẫn chỉ giúp tái lập catalog và split. Nó không có nghĩa
website phải đọc thư mục `experiments/.../seed_42`. Có thể đổi seed, nhưng mọi
artifact của một lần chạy phải được tạo từ cùng catalog và checkpoint.

## 2. ĐIỀU KIỆN TIÊN QUYẾT

- Git.
- Python 3.12 hoặc 3.13.
- Docker Desktop có Docker Compose v2.
- Ít nhất khoảng 15–20 GB dung lượng trống cho dataset, model, image Docker và
  Hugging Face cache.
- Internet ở lần đầu để tải package, Docker image, CLIP và model dịch VI → EN.
- GPU NVIDIA là tùy chọn. CPU vẫn chạy được nhưng fine-tuning và validation lâu hơn.
- Node.js chỉ cần khi chạy frontend ngoài Docker; không cần cho workflow Docker.

Dataset mặc định:

```text
Mini Fashion Product Images and Text Dataset
https://www.kaggle.com/datasets/nirmalsankalana/mini-product-image-and-text-dataset
```

Đặt dữ liệu như sau:

```text
project/
└── data/
    ├── data.csv
    └── data/
        ├── 3238.jpg
        └── ...
```

## 3. CLONE VÀ CHUYỂN VÀO PROJECT

Tên thư mục sau khi clone do người dùng quyết định, vì vậy hướng dẫn không giả
định repository luôn có tên `dacntt`.

```powershell
git clone https://github.com/Pun-2510/clothes-styling-system.git
Set-Location .\clothes-styling-system\project
```

Nếu clone bằng tên khác, thay `clothes-styling-system` bằng tên đó. Từ mục này
trở đi, tất cả lệnh vận hành đều chạy tại thư mục `project/`.

## 4. TẠO MÔI TRƯỜNG PYTHON

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Không bắt buộc chạy `Activate.ps1`; các lệnh dưới đây gọi trực tiếp Python trong
`.venv` để tránh nhầm môi trường.

## 5. OFFLINE PIPELINE TỪ ĐẦU

### 5.1. Chuẩn bị catalog và category audit

```powershell
.\.venv\Scripts\python.exe -m src.prepare_dataset `
  --csv data/data.csv `
  --image-dir data/data `
  --num-products 5000 `
  --min-per-category 25 `
  --sampling-strategy diverse `
  --seed 42 `
  --output data/processed/products.csv
```

Các tham số có thể thay đổi:

- `--num-products 5000`: số sản phẩm muốn đưa vào catalog. Có thể dùng 4000,
  6000 hoặc giá trị phù hợp máy; không được lớn hơn số dòng hợp lệ của dataset.
- `--all`: dùng toàn bộ sản phẩm hợp lệ; dùng thay cho `--num-products`.
- `--min-per-category 25`: số lượng tối thiểu mong muốn cho mỗi category. Nếu
  dữ liệu nguồn không đủ, audit ghi cảnh báo chứ không tạo dữ liệu giả.
- `--sampling-strategy diverse`: ưu tiên mẫu khác nhau trong từng category.
  Có thể dùng `random` cho baseline đơn giản.
- `--seed 42`: cố định lựa chọn ngẫu nhiên và split để có thể tái lập.
- `--output`: nơi ghi catalog vận hành. Giữ đường dẫn mặc định giúp `.env` và
  Docker không cần sửa.

Kết quả đều nằm trong `project/data/processed/`:

```text
products.csv
products.preparation.json
products.category_audit.csv
products.category_audit.json
```

### 5.2. Fine-tune CLIP

```powershell
.\.venv\Scripts\python.exe -m src.finetune_clip `
  --csv data/processed/products.csv `
  --run-dir runs/clip_finetune `
  --trainable full `
  --epochs 3 `
  --batch-size 16 `
  --learning-rate 1e-6 `
  --weight-decay 0.01 `
  --category-balance-power 0.5 `
  --max-category-sample-weight 5 `
  --seed 42
```

Giải thích tham số chính:

- `--epochs 3`: số lần model đi qua tập train. Tăng có thể cải thiện hoặc gây
  overfitting và luôn làm thời gian chạy dài hơn.
- `--batch-size 16`: số cặp ảnh–text xử lý trong một bước. Giảm xuống 8 hoặc 4
  nếu thiếu VRAM/RAM; tăng nếu phần cứng cho phép.
- `--learning-rate 1e-6`: mức cập nhật trọng số mỗi bước.
- `--trainable full`: cập nhật toàn bộ CLIP. Dùng `projections` để chỉ cập nhật
  projection layers và `logit_scale`, nhanh và nhẹ hơn.
- Hai tham số category balance giảm ảnh hưởng của category quá lớn khi train.
- `--seed` phải khớp quy ước của lần chạy nếu muốn tái lập split.

`run-dir` được xem là bất biến và không ghi đè. Nếu thư mục đã tồn tại, dùng tên
mới, ví dụ `runs/clip_finetune_v2`, rồi thay đúng tên đó trong các bước sau và
trong `.env`. Checkpoint triển khai mặc định:

```text
runs/clip_finetune/best/
```

Validation chọn epoch tốt nhất; test split không được dùng để cập nhật model.

### 5.3. Sinh image và text embeddings

```powershell
.\.venv\Scripts\python.exe -m src.generate_clip_embeddings `
  --csv data/processed/products.csv `
  --model runs/clip_finetune/best `
  --output-dir embeddings `
  --modality both `
  --batch-size 16
```

- `--modality both` tạo cả hai file mà website cần.
- `--batch-size` chỉ ảnh hưởng tài nguyên và tốc độ encode, không đổi công thức
  cosine similarity. Giảm khi thiếu bộ nhớ.
- Có thể thêm `--device cpu` hoặc `--device cuda`.

Không xóa các file `.npy.json`; backend dùng chúng để xác minh catalog,
checkpoint và embedding không bị trộn từ các lần chạy khác nhau.

### 5.4. Chọn retrieval weights trên validation

Đây là lệnh vận hành trong `project/`, không gọi module trong `experiments/`:

```powershell
.\.venv\Scripts\python.exe -m src.tune_retrieval_weights `
  --run-dir runs/clip_finetune `
  --catalog-csv data/processed/products.csv `
  --image-embeddings embeddings/image_embeddings.npy `
  --text-embeddings embeddings/text_embeddings.npy `
  --output-dir validation/weight_tuning `
  --weights-output retrieval_weights.json `
  --objective-k 10 `
  --batch-size 16
```

Lệnh chỉ dùng validation split và chọn:

- Tỷ lệ text→image so với text→text.
- Trọng số ảnh, attribute và category cho combined query.
- Category bonus cho image-only.

`--objective-k 10` nghĩa là chọn theo danh sách Top-10; có thể đổi nếu giao diện
chủ yếu trả Top-5. `--batch-size` điều khiển encode các query validation. Không
cần truyền `--expected-gallery-size`; chương trình tự đọc số dòng catalog. Nếu
muốn kiểm tra nghiêm ngặt, thêm đúng giá trị đã prepare, ví dụ:

```powershell
  --expected-gallery-size 5000
```

Các bảng thử được lưu trong `validation/weight_tuning/`; file website đọc là
`project/retrieval_weights.json`. Mỗi lần tune lại nên dùng output mới, ví dụ
`validation/weight_tuning_v2`, vì chương trình không ghi đè thư mục kết quả cũ.

## 6. CẤU HÌNH CỤC BỘ

Tạo `.env` một lần:

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Cấu hình mặc định hoàn toàn nằm trong `project/`:

```env
FASHION_CATALOG_DIR=./data/processed
FASHION_IMAGE_DIR=./data/data
FASHION_EMBEDDINGS_DIR=./embeddings
FASHION_CHECKPOINT_DIR=./runs/clip_finetune/best
FASHION_RETRIEVAL_WEIGHTS_FILE=./retrieval_weights.json

CATEGORY_CONFIDENCE_FLOOR=0.50
CATEGORY_CONFIDENCE_THRESHOLD=0.60
CATEGORY_TOP_PER_CLASS=3
CATEGORY_PROTOTYPE_WEIGHT=1.0
```

Đường dẫn tương đối được Docker Compose hiểu từ `project/`, không phụ thuộc tên
thư mục repository. Nếu dùng run mới, sửa đồng bộ checkpoint, embeddings,
catalog và weight file; không trộn artifact từ các lần chạy khác nhau.

## 7. KIỂM TRA ARTIFACT VÀ CHẠY ONLINE

Kiến trúc Docker chỉ public một cổng:

```text
Browser :80
  → Web Reverse Proxy
      ├── website → frontend:5173
      └── API/docs/images → API Gateway
                              → least_conn → backend / backend_2
```

Frontend, API Gateway và backend không publish cổng trực tiếp ra host.

```powershell
Test-Path .\data\processed\products.csv
Test-Path .\data\data
Test-Path .\runs\clip_finetune\best\model.safetensors
Test-Path .\embeddings\image_embeddings.npy
Test-Path .\embeddings\text_embeddings.npy
Test-Path .\retrieval_weights.json
docker compose config --quiet
```

Tất cả `Test-Path` phải trả về `True`. Sau đó chạy website:

```powershell
docker compose up -d --build
```

Địa chỉ:

- Website qua Web Reverse Proxy: `http://localhost`
- API qua cùng public origin: `http://localhost/api`
- Swagger: `http://localhost/docs`
- Developer Dashboard: `http://localhost/developer`

Kiểm tra trạng thái:

```powershell
docker compose ps
Invoke-RestMethod http://localhost/api/health
Invoke-RestMethod http://localhost/api/health/ready
docker compose exec reverse-proxy nginx -t
docker compose exec api-gateway nginx -t
docker compose logs -f reverse-proxy api-gateway backend backend_2
```

Sau khi đổi `.env`, checkpoint, embeddings hoặc weights:

```powershell
docker compose up -d --build --force-recreate
```

Dừng hệ thống:

```powershell
docker compose down
```

Không dùng `docker compose down -v` nếu muốn giữ Hugging Face và Node cache.

## 8. CHẠY NHANH BẰNG ARTIFACT CÓ SẴN (TÙY CHỌN)

Nếu không muốn tự fine-tune, có thể tải gói artifact đồng bộ:

```text
https://drive.google.com/file/d/1_xiQjMBlf5EMZVRflDwC2G5kbgEkwy00/view?usp=drive_link
```

Sau khi giải nén, sắp xếp hoặc đổi tên thành đúng cấu trúc project-local:

```text
project/data/processed/products.csv
project/data/data/<images>
project/runs/clip_finetune/best/<checkpoint files>
project/embeddings/image_embeddings.npy
project/embeddings/image_embeddings.npy.json
project/embeddings/text_embeddings.npy
project/embeddings/text_embeddings.npy.json
project/retrieval_weights.json
```

Nếu gói cũ mang tên seed, có thể giữ tên thư mục đó nhưng phải cập nhật `.env`.
Website vẫn không được trỏ tới `experiments/`.

## 9. HỆ THỐNG ONLINE LÀM GÌ

- Text-only: kết hợp text→image và text→text; chỉ hard-filter khi câu truy vấn
  nêu rõ đúng một catalog category.
- Image-only: image→image trên toàn catalog; category prediction chỉ thêm soft
  bonus theo confidence và quan hệ Top-2, không lọc cứng.
- Image + text: ảnh giữ ngữ cảnh sản phẩm và text bổ sung category/attribute.
  Một category được ghi rõ trong text tạo hard candidate pool. Nếu text không
  nêu category, hệ thống dùng confidence và quan hệ Top-2 của ảnh để cộng soft
  bonus động giống Image-only, không loại sản phẩm ngoài category.
- Tất cả nhánh dùng raw cosine similarity; percentile không được dùng.

## 10. THÍ NGHIỆM NGHIÊN CỨU RIÊNG

Các so sánh pretrained/fine-tuned CLIP, TF-IDF, BERT, SBERT, ResNet18,
EfficientNet, ViT, category ablation, system pipeline và tổng hợp nhiều seed nằm ở:

```text
../experiments/model_comparison/README.md
```

Thí nghiệm có thể đọc artifact của từng seed để đánh giá, nhưng Docker và code
trong `project/` không đọc kết quả `experiments/`. Không cần chạy thí nghiệm để
khởi động website.

## 11. KIỂM THỬ CODE

```powershell
.\.venv\Scripts\python.exe -m src.tune_retrieval_weights --self-test
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Build frontend ngoài Docker (tùy chọn):

```powershell
Set-Location web/frontend
npm ci
npm run build
Set-Location ../..
```

## 12. LỖI THƯỜNG GẶP

- `Output directory is not empty`: chọn tên mới cho `--output-dir` validation.
- `Experiment dataset changed`: không chỉnh `dataset.csv` bên trong một run cũ;
  tạo run mới từ catalog mới.
- Embedding/checksum mismatch: sinh lại cả image và text embeddings bằng đúng
  catalog và checkpoint đang cấu hình.
- Docker báo thiếu mount: kiểm tra năm biến `FASHION_*` trong `.env`; các đường
  dẫn phải tồn tại trước khi chạy Compose.
- HTTP 502: xem `docker compose ps` và log hai backend.
- HTTP 429: chờ rate-limit window hoặc giảm request lặp lại.
- CPU quá chậm: giảm catalog/batch size để thử flow trước, hoặc dùng CUDA.
- Git clone vào tên khác: không sửa code; chỉ chuyển đúng vào `<tên-clone>/project`.
