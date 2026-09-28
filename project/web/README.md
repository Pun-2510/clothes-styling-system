# Website: React + FastAPI + Nginx

`web/backend/`, `web/frontend/`, `web/nginx/` chứa code phục vụ website.
Logic ML dùng chung vẫn ở `src/`. Dữ liệu, embeddings, checkpoint và log vẫn
nằm ở thư mục gốc `project/`, không chuyển vào web hoặc kết quả thí nghiệm.

## Chạy một lệnh

Từ `project/`, sau khi bật Docker Desktop và chuẩn bị đủ artifact:

```powershell
docker-compose up -d --build
```

Lệnh tương đương: `docker compose up -d --build`.
Web: http://localhost:5173 — API/Swagger: http://localhost:8000/docs.
Compose tự đọc `.env`, khởi động frontend, Nginx và hai backend. Không tự
tiền xử lý, training, sinh embedding hoặc compare khi khởi động Docker.

## Artifact riêng cho web

```dotenv
FASHION_CATALOG_DIR=./data/processed
FASHION_EMBEDDINGS_DIR=./embeddings_finetuned
FASHION_CHECKPOINT_DIR=./runs/clip_finetune_3epochs_local/best
FASHION_IMAGE_DIR=./data/data
```

Các thư mục được hiểu tương đối với `project/docker-compose.yml`. `.env` không
được lưu trên Git; máy mới dùng giá trị mặc định hoặc sao chép `.env.example`.
Không sửa tên đường dẫn theo số sản phẩm. Số lượng do CSV và embeddings quyết định.

Muốn thay số lượng (ví dụ 4.000), tự chạy từ `project/`:

```powershell
.\.venv\Scripts\python.exe -m src.prepare_dataset --num-products 4000
.\.venv\Scripts\python.exe -m src.generate_clip_embeddings `
  --csv data/processed/products.csv `
  --model runs/clip_finetune_3epochs_local/best `
  --output-dir embeddings_finetuned
docker-compose up -d --build --force-recreate backend backend_2
```

Hai lệnh Python ghi lại catalog/embeddings web; lưu bản sao nếu cần giữ bộ cũ.
Tạm dừng hai backend trước khi ghi đè artifact đang dùng. Sau khi ghi đè nội dung
artifact, `up --build` có thể không tự tạo lại backend nếu image/config không đổi;
dùng `--force-recreate` như trên hoặc restart cả hai để nạp dữ liệu mới.
Khởi động hệ thống bình thường sau khi chuẩn bị xong vẫn chỉ cần một lệnh `up`.

Fine-tune là tùy chọn: nếu cần, chạy `src.finetune_clip` vào một `runs/<tên-mới>`
sau tiền xử lý và trước sinh embeddings. Sau đó dùng đúng checkpoint mới để sinh
embeddings và cập nhật `FASHION_CHECKPOINT_DIR` trong `.env`. Catalog, embeddings
và checkpoint phải khớp provenance; không sửa hash để bỏ qua kiểm tra.

`comparisons/` chỉ dùng khi cần số liệu đánh giá. Website không yêu cầu chạy nó.

## Chạy API hoặc frontend riêng ngoài Docker

API, từ `project/`:

```powershell
.\.venv\Scripts\python.exe -m uvicorn web.backend.main:app --host 127.0.0.1 --port 8000
```

Frontend:

```powershell
Set-Location web/frontend
npm ci
npm run dev
```

Vite mặc định proxy tới localhost:8000; trong Compose proxy tới Nginx.
`.env` dành cho Compose không tự cấu hình lệnh Python chạy trực tiếp; Python dùng
`src/config.py` và các biến `CLIP_MODEL_PATH`, `CLIP_EMBEDDING_DIR` nếu được đặt.

## Tài liệu và kiểm tra

- [Nginx, phân phối request và thử dự phòng](nginx/README.md)
- [Hướng dẫn tổng quan](../README.md)
- [Fine-tuning](../FINETUNING.md)
- [Thí nghiệm độc lập](../comparisons/README.md)

```powershell
docker compose config --quiet
docker compose exec nginx nginx -t
docker compose logs -f nginx backend backend_2
```

Hai backend nạp model vào RAM riêng; tên service và đường API không đổi khi
chuyển thư mục. Backend build context vẫn là `project/` để lấy được `src/`;
frontend build context là `web/frontend/`.
