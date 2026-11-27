# HƯỚNG DẪN CHẠY HỆ THỐNG TỪ GÓI NỘP

Tài liệu này dành cho người nhận file ZIP của dự án. Giải nén gói nộp, mở thư mục
`source_code/` trong VS Code hoặc File Explorer, rồi mở PowerShell **ngay tại thư
mục này**. Mọi lệnh bên dưới dùng đường dẫn tương đối từ `source_code/`; không
cần Git, không cần đổi tên thư mục và không cần gõ `Set-Location`.

## 1. THÀNH PHẦN SAU KHI GIẢI NÉN

```text
source_code/
├── src/                              Tiền xử lý, huấn luyện, validation, retrieval
├── web/                              React, FastAPI và Nginx
├── tests/                            Kiểm tra code
├── requirements.txt
├── .env.example                      Mẫu cấu hình
├── docker-compose.yml
└── readme.txt                        Tài liệu này
```

Các thư mục `data/`, `runs/`, `embeddings/`, `metadata_index/`, `validation/`
và các file `.env`, `retrieval_weights.json` được tạo hoặc bổ sung tại máy chạy.
Website dùng những artifact trong chính `source_code/`, không đọc kết quả thí
nghiệm trong thư mục khác. Catalog, checkpoint, embeddings, TF-IDF index và
retrieval weights phải thuộc cùng cấu hình dữ liệu.

Nếu hồ sơ nộp chứa `data.zip` riêng, giải nén dữ liệu vào đúng vị trí sau:

```text
source_code/
└── data/
    ├── data.csv                      Metadata nguồn
    └── data/                         Ảnh mà metadata tham chiếu
        ├── 3238.jpg
        └── ...
```

Nếu chỉ có `data/processed/products.csv` và những ảnh được catalog tham chiếu,
có thể bắt đầu từ bước fine-tune 5.2. Muốn chạy lại cả bước chuẩn bị catalog 5.1
thì cần metadata nguồn `data/data.csv` và ảnh trong `data/data/`.

## 2. ĐIỀU KIỆN TIÊN QUYẾT

- Windows PowerShell, Python 3.12 hoặc 3.13 và Docker Desktop có Compose v2.
- Ít nhất khoảng 15–20 GB dung lượng trống cho dataset, model, Docker image
  và Hugging Face cache.
- Internet ở lần đầu để tải package, Docker image, CLIP và model dịch VI → EN.
- GPU NVIDIA là tùy chọn. CPU vẫn chạy được nhưng fine-tuning lâu hơn.
- Node.js chỉ cần nếu tự chạy frontend ngoài Docker.

Dữ liệu gốc được dùng trong dự án là Mini Fashion Product Images and Text
Dataset: https://www.kaggle.com/datasets/nirmalsankalana/mini-product-image-and-text-dataset
Nếu gói nộp không kèm dữ liệu, tải và đặt theo cấu trúc ở mục 1.

## 3. KIỂM TRA THƯ MỤC LÀM VIỆC

Trong PowerShell đang mở tại `source_code/`, chạy:

```powershell
(Get-Location).Path
Test-Path .\docker-compose.yml
Test-Path .\src
Test-Path .\readme.txt
```

Ba lệnh `Test-Path` phải trả về `True`. Nếu có `False`, mở lại terminal tại
`source_code/` trước khi chạy các bước tiếp theo. Không chạy lệnh ở thư mục
chứa file ZIP hoặc ở cấp cha của `source_code/`.

## 4. TẠO MÔI TRƯỜNG PYTHON

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Không bắt buộc chạy `Activate.ps1`; các lệnh dưới đây gọi trực tiếp Python trong
`.venv` để tránh nhầm môi trường.

## 5. OFFLINE PIPELINE TỪ ĐẦU

### 5.1. Chuẩn bị catalog, category audit và audience audit

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

Kết quả đều nằm trong `data/processed/` của thư mục đang chạy:

```text
products.csv
products.preparation.json
products.category_audit.csv
products.category_audit.json
products.audience_audit.csv
products.audience_audit.json
```

`audience_audit` tách riêng `age_group` và `gender_group`, đồng thời ghi nguồn,
độ tin cậy và bằng chứng của từng nhãn. Hệ thống ưu tiên tên sản phẩm, sau đó
đến category; mô tả dài chỉ được dùng khi có cụm từ chỉ đối tượng rõ ràng.
Những trường hợp thiếu hoặc mâu thuẫn bằng chứng được giữ ở `unknown` để tránh
đưa nhầm sản phẩm người lớn vào trang Trẻ em. Audit này không thay đổi
`products.csv`, checkpoint CLIP hoặc embeddings.

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
  nếu thiếu VRAM/RAM; tăng khi phần cứng cho phép để xử lý nhiều mẫu mỗi bước.
- `--learning-rate 1e-6`: mức cập nhật trọng số sau mỗi bước tối ưu.
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

### 5.4. Sinh TF-IDF metadata index

TF-IDF được xây trực tiếp từ `product_name` và `description` của catalog đã xử
lý. Nó bổ sung mức khớp từ khóa cho text-only và combined query; image-only
không dùng TF-IDF. Text-only tính điểm TF-IDF rồi, nếu truy vấn nêu rõ đúng một
category, chỉ xếp hạng ứng viên trong category đó. Combined chọn category từ
văn bản nêu rõ hoặc từ ảnh; TF-IDF chỉ tăng điểm cho sản phẩm thuộc category
đã chọn, sau đó Top-K cũng chỉ lấy từ category này. Vì vậy một từ chung như
`black` không kéo sản phẩm khác loại vào kết quả combined. Trường category
không được đưa vào tài liệu TF-IDF để tránh tính lặp cùng một bằng chứng.

```powershell
.\.venv\Scripts\python.exe -m src.build_metadata_index `
  --catalog-csv data/processed/products.csv `
  --output-dir metadata_index `
  --min-df 2 `
  --max-features 50000
```

- `--min-df 2`: bỏ từ chỉ xuất hiện trong một sản phẩm, giúp giảm nhiễu.
- `--max-features 50000`: giới hạn kích thước từ vựng và bộ nhớ; có thể giảm
  nếu máy ít RAM.
- Phải chạy lại bước này khi thay `products.csv`; không cần fine-tune CLIP hay
  sinh lại CLIP embeddings chỉ vì thay TF-IDF index.

Kết quả nằm trong `metadata_index/` và có hash catalog để backend phát
hiện trường hợp index không cùng catalog hoặc sai thứ tự dòng.

### 5.5. Chọn retrieval weights trên validation

Đây là lệnh vận hành trong thư mục hệ thống, không gọi module trong `experiments/`:

```powershell
.\.venv\Scripts\python.exe -m src.tune_retrieval_weights `
  --run-dir runs/clip_finetune `
  --catalog-csv data/processed/products.csv `
  --image-embeddings embeddings/image_embeddings.npy `
  --text-embeddings embeddings/text_embeddings.npy `
  --metadata-index-dir metadata_index `
  --output-dir validation/weight_tuning_category_gated_v1 `
  --weights-output retrieval_weights.json `
  --objective-k 10 `
  --batch-size 16
```

Lệnh chỉ dùng validation split và chọn:

- Tỷ lệ text→image so với text→text và trọng số TF-IDF cho text-only.
- Trọng số ảnh và TF-IDF metadata cho combined query. Website chỉ xếp hạng
  trong một danh mục: danh mục nêu rõ trong text hoặc danh mục ảnh dự đoán.
  Bộ chọn giữ ảnh là nguồn chính (`image_weight >= 0.60`), giới hạn TF-IDF ở
  mức tối đa `0.30` và chỉ chấp nhận mức giảm category macro-F1 rất nhỏ so với
  cấu hình tốt nhất.
- Website hiện dùng Hard Category cho image-only và combined; các tham số
  category bonus của phương pháp soft/dynamic chỉ phục vụ đối chiếu lịch sử,
  không được cộng vào điểm của hai luồng đang triển khai.

`--objective-k 10` nghĩa là chọn theo danh sách Top-10; có thể đổi nếu giao diện
chủ yếu trả Top-5. `--batch-size` điều khiển encode các query validation. Không
cần truyền `--expected-gallery-size`; chương trình tự đọc số dòng catalog. Nếu
muốn kiểm tra nghiêm ngặt, thêm đúng giá trị đã prepare, ví dụ:

```powershell
  --expected-gallery-size 5000
```

Các bảng thử được lưu trong `validation/weight_tuning_category_gated_v1/`; file website đọc là
`retrieval_weights.json` ở ngay thư mục hệ thống. Mỗi lần tune lại nên dùng output mới, ví dụ
`validation/weight_tuning_v2`, vì chương trình không ghi đè thư mục kết quả cũ.

## 6. CẤU HÌNH CỤC BỘ

Tạo `.env` một lần:

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Cấu hình mặc định hoàn toàn nằm trong thư mục hệ thống:

```env
FASHION_CATALOG_DIR=./data/processed
FASHION_IMAGE_DIR=./data/data
FASHION_EMBEDDINGS_DIR=./embeddings
FASHION_METADATA_INDEX_DIR=./metadata_index
FASHION_CHECKPOINT_DIR=./runs/clip_finetune/best
FASHION_RETRIEVAL_WEIGHTS_FILE=./retrieval_weights.json

CATEGORY_CONFIDENCE_FLOOR=0.50
CATEGORY_CONFIDENCE_THRESHOLD=0.60
CATEGORY_TOP_PER_CLASS=3
CATEGORY_PROTOTYPE_WEIGHT=1.0
```

`CATEGORY_TOP_PER_CLASS` và `CATEGORY_PROTOTYPE_WEIGHT` tham gia dự đoán
category từ ảnh. Confidence vẫn được ghi để xem và phân tích lỗi, nhưng giá
trị floor/threshold không bật hoặc tắt hard filter của website hiện tại.

Đường dẫn tương đối được Docker Compose hiểu từ thư mục chứa `docker-compose.yml`, không phụ thuộc tên
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
Test-Path .\metadata_index\catalog_tfidf.npz
Test-Path .\metadata_index\tfidf_vectorizer.joblib
Test-Path .\metadata_index\metadata.json
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

## 8. HỆ THỐNG ONLINE LÀM GÌ

- Text-only: kết hợp text→image và text→text; chỉ hard-filter khi câu truy vấn
  nêu rõ đúng một catalog category.
- Image-only: hệ thống mã hóa ảnh bằng CLIP, dự đoán một danh mục từ ảnh,
  sau đó chỉ xếp hạng sản phẩm thuộc danh mục đó theo cosine similarity.
- Image + text: ảnh và văn bản cùng đóng góp vào điểm CLIP. Nếu văn bản nêu rõ
  đúng một danh mục, hệ thống chỉ tìm trong danh mục đó; nếu không, hệ thống
  dùng danh mục dự đoán từ ảnh. TF-IDF tinh chỉnh điểm bằng tên và mô tả
  sản phẩm trong danh mục đã chọn. Nếu văn bản nêu nhiều danh mục, giao diện
  gợi ý chọn một danh mục để tìm lại.
- Tất cả nhánh bắt đầu từ raw cosine similarity của CLIP; text-only và
  combined còn dùng TF-IDF để tinh chỉnh thứ hạng. Percentile không được dùng.
- Website có duyệt category, chi tiết sản phẩm, Favorites và lịch sử tìm kiếm
  lưu trên thiết bị, không yêu cầu đăng nhập.
- Catalog browsing có thêm bộ lọc metadata `Women / Men / Kids / Unisex`.
  Nhãn này được suy ra bảo thủ từ tên sản phẩm và category; mô tả chỉ là nguồn
  dự phòng khi có cụm từ rõ ràng. Hệ thống không đoán
  giới tính từ ảnh. Mặc định nó không thay đổi kết quả truy xuất. Nếu người dùng
  chủ động chọn một nhóm trong `Tùy chọn tìm kiếm`, hệ thống giới hạn tập ứng
  viên theo nhóm đó nhưng vẫn giữ nguyên embedding, cosine similarity và công
  thức category-aware ranking.

## 9. THÍ NGHIỆM NGHIÊN CỨU RIÊNG

Các bảng so sánh mô hình và category ranking nằm trong báo cáo nộp kèm.
Chúng là bằng chứng đánh giá, không phải đầu vào của website. Người nhận không
cần chạy thí nghiệm để khởi động hệ thống trong `source_code/`.

## 10. KIỂM THỬ CODE

```powershell
.\.venv\Scripts\python.exe -m src.tune_retrieval_weights --self-test
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## 11. LỖI THƯỜNG GẶP

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
- Lệnh chạy sai vị trí: kiểm tra PowerShell đang mở tại `source_code/` bằng
  `Test-Path .\docker-compose.yml`.
