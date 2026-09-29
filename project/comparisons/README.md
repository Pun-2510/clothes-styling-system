# Thí nghiệm so sánh CLIP

Chạy các lệnh dưới đây từ thư mục `project/`, nơi chứa `docker-compose.yml`.
Đã có Python environment, dependencies, `data/data.csv` và ảnh trong `data/data/`.
Pipeline không tự khởi động Docker.

## 1. Dùng checkpoint đã fine-tune — không huấn luyện lại

```powershell
.\.venv\Scripts\python.exe -m comparisons.run_pipeline `
  --num-products 4000 `
  --reuse-run runs/clip_finetune_3epochs_local `
  --output-dir comparisons/results/clip_4000
```

Các bước tự chạy theo thứ tự:

1. Chọn đúng 4.000 sản phẩm hợp lệ, loại ảnh thiếu/hỏng/trùng nội dung.
2. Kiểm tra run cũ, checkpoint và các sản phẩm test có trong catalog mới.
3. Sinh image/text embeddings cho catalog bằng pretrained CLIP.
4. Sinh image/text embeddings cho cùng catalog bằng fine-tuned CLIP.
5. Dùng embeddings vừa sinh để tính Precision@K, Recall@K và F1@K cho hai model.

Run cũ cần đầy đủ `dataset.csv`, `experiment.json`, `training.json`, `best/`
và ảnh của dataset gốc. Training phải có trạng thái `complete` và vượt qua
kiểm tra tính toàn vẹn. Không sửa hash trong metadata để bỏ qua lỗi dataset.

**4.000 là số sản phẩm của catalog, không phải số mẫu test.** Khi dùng lại run,
chỉ đánh giá phần giao giữa catalog mới và **tập test gốc của run đó**. Cả query
và gallery đều lấy từ phần giao này; không đưa sản phẩm train/validation vào test.
Không tự chia lại test cho checkpoint cũ. Số mẫu được đánh giá có thể nhỏ hơn nhiều
so với 4.000. Nếu còn dưới hai mẫu test, pipeline dừng trước bước sinh embedding:
tăng số lượng, nới điều kiện lọc hoặc chạy thí nghiệm huấn luyện mới.

Product ID hiện được tạo từ vị trí dòng của CSV nguồn. Khi dùng lại run, nên dùng
cùng CSV nguồn, không đổi thứ tự dòng; ảnh, tên/mô tả và category phải khớp.

## 2. Huấn luyện và so sánh trên một catalog mới

Bỏ `--reuse-run` để fine-tune một model mới trên catalog được chọn:

```powershell
.\.venv\Scripts\python.exe -m comparisons.run_pipeline `
  --num-products 4000 `
  --epochs 3 `
  --output-dir comparisons/results/clip_4000_new
```

Quy trình: chuẩn bị catalog → chia train/validation/test → fine-tune → sinh
embeddings cho hai model → compare trên test. Checkpoint tốt nhất được chọn bằng
validation, không dùng test để chọn epoch. Mặc định validation/test mỗi tập khoảng
10%; số lượng thực tế phụ thuộc việc nhóm các bản ghi liên quan để tránh rò rỉ dữ liệu.

Lệnh này **có huấn luyện**, có thể mất nhiều thời gian. Có thể thêm
`--device cuda`, `--batch-size 16`, `--learning-rate 1e-6`,
`--trainable projections` hoặc `--model openai/clip-vit-base-patch32`.
Các tham số training chỉ có tác dụng khi không dùng `--reuse-run`.

## 3. Số lượng và điều kiện linh hoạt

```powershell
.\.venv\Scripts\python.exe -m comparisons.run_pipeline `
  --num-products 1000 `
  --categories "Backpacks" "Handbags" `
  --seed 42 `
  --epochs 3 `
  --ks 1 5 10 `
  --output-dir comparisons/results/bags_1000
```

Tên category phải khớp giá trị trong CSV nguồn; ví dụ trên chỉ dùng được nếu dữ liệu
có các category đó. Có thể chỉ định `--csv` và `--image-dir` để đổi nguồn.
Số lượng được lấy chính xác sau kiểm tra dữ liệu, có phân bổ theo category;
không còn cố định 5.015 sản phẩm. Nếu số yêu cầu vượt số hợp lệ sau lọc, lệnh báo lỗi,
không âm thầm lấy ít hơn. Bước kiểm tra ảnh có thể cần quét nhiều hơn N ảnh.

Thêm `--dry-run` vào bất kỳ lệnh pipeline nào để xem các lệnh sẽ chạy;
không tạo dữ liệu, tải model hay huấn luyện. Với `--reuse-run`, bước xem trước vẫn
đọc `training.json` để xác định model gốc. Xem mọi tùy chọn bằng `--help`.

Chỉ tiền xử lý, không training/embedding/compare:

```powershell
.\.venv\Scripts\python.exe -m src.prepare_dataset `
  --num-products 4000 --output data/processed/products.csv
```

Lệnh tiền xử lý riêng có thể ghi đè CSV đầu ra đã có; sau khi thay catalog cần tạo
lại embeddings tương ứng. Dùng `--all` thay cho `--num-products` để chọn toàn bộ
sản phẩm hợp lệ ở lệnh tiền xử lý riêng.

## 4. Cấu trúc kết quả

```text
comparisons/
├── run_pipeline.py        # Điều phối các bước, nhận số lượng và bộ lọc
├── compare_clip.py        # Tính metric, xuất bảng so sánh
├── README.md
└── results/
    └── clip_4000/
        ├── data/
        │   ├── products.csv
        │   └── products.preparation.json
        ├── embeddings/
        │   ├── pretrained/   # Image/text .npy và metadata
        │   └── finetuned/    # Image/text .npy và metadata
        ├── training/         # Chỉ xuất hiện khi huấn luyện mới
        │   ├── dataset.csv
        │   ├── experiment.json
        │   ├── training.json
        │   └── best/
        ├── logs/             # Một log cho mỗi bước
        ├── pipeline.json     # Tham số, lệnh đã chạy, trạng thái từng bước
        ├── comparison.csv
        ├── comparison.json
        └── website.env       # Sinh sau khi pipeline hoàn thành
```

Mỗi lần chạy phải dùng `--output-dir` mới. Pipeline từ chối ghi đè thư mục đã có,
không sửa run cũ hoặc catalog đang phục vụ web. Nếu lỗi, xem `pipeline.json` và
`logs/`; hiện chưa có tự động resume. Có thể kiểm tra lỗi rồi chạy lại với thư mục
mới, hoặc tự chạy bước cần thiết bằng lệnh đã ghi trong manifest.

`comparison.csv` có ba tác vụ: text→image, image→text, image→image theo category;
mỗi tác vụ có Precision, Recall, F1 ở các K được chọn và chênh lệch hai model.
`comparison.json` ghi thêm `catalog_size`, `original_test_size`,
`evaluated_test_size`, `omitted_test_products` và giao thức đánh giá.
Metric được lấy trung bình theo query; category là nhãn tương đồng thay thế,
không phải đánh giá trực tiếp sở thích người dùng.

Không kết luận model tốt hơn chỉ bằng so điểm giữa hai lần chạy có test/gallery
khác nhau: kích thước và thành phần gallery cũng ảnh hưởng độ khó của truy hồi.
So sánh pretrained với fine-tuned trong **cùng một thí nghiệm** mới giữ cùng điều kiện.

## 5. Chạy website bằng catalog của thí nghiệm

Sau khi pipeline hoàn thành:

```powershell
docker compose --env-file comparisons/results/clip_4000/website.env up -d --build
```

Web dùng catalog 4.000 sản phẩm, embeddings fine-tuned và đúng checkpoint tương ứng.
Vẫn chỉ một cấu hình Docker Compose, không tách dev/prod. Muốn chỉ chạy
`docker-compose up -d --build`, đặt bốn đường dẫn artifact vào `project/.env`
một lần (tham khảo `.env.example`); Compose tự đọc file này. Không truyền `--env-file`
thì Compose dùng `.env` nếu có, sau đó mới đến đường dẫn mặc định. Chạy lệnh này khi muốn chuyển
catalog; nó có thể tạo lại container đang phục vụ web.

`website.env` chứa đường dẫn tuyệt đối trên máy hiện tại. Khi chuyển sang máy khác,
cần chỉnh bốn đường dẫn trong file. Các artifact dung lượng lớn trong `results/`
được bỏ qua bởi Git và Docker build context; muốn chia sẻ thì đóng gói riêng.

## 6. Chỉ chạy compare cho run cũ

```powershell
.\.venv\Scripts\python.exe -m comparisons.compare_clip `
  --run-dir runs/clip_finetune_3epochs_local `
  --output-dir comparisons/results/compare_existing `
  --ks 1 5 10
```

Lệnh này đánh giá toàn bộ test gốc, không tạo catalog web mới. Khi không cung cấp
cache, lệnh tự mã hóa test bằng hai model.
Các hàm tính metric dùng chung vẫn nằm trong `src/evaluation.py` vì fine-tuning
cũng cần chúng để đánh giá validation.
