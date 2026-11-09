# Website: React + Web Reverse Proxy + API Gateway + FastAPI

`web/backend/`, `web/frontend/`, `web/reverse-proxy/` và `web/api-gateway/`
chứa code phục vụ website.
Logic ML dùng chung vẫn ở `src/`. Catalog, checkpoint, embeddings, validation
weights và log đều nằm trong `project/`; website không đọc `experiments/`.

## Chạy một lệnh

Từ `project/`, sau khi bật Docker Desktop và chuẩn bị đủ artifact:

```powershell
docker compose up -d --build
```

Web: http://localhost — API/Swagger: http://localhost/docs.
Compose tự đọc `.env`, khởi động frontend, Web Reverse Proxy, API Gateway và hai backend. Không tự
tiền xử lý, training, sinh embedding hoặc compare khi khởi động Docker.

## Artifact riêng cho web

```dotenv
FASHION_CATALOG_DIR=./data/processed
FASHION_EMBEDDINGS_DIR=./embeddings
FASHION_CHECKPOINT_DIR=./runs/clip_finetune/best
FASHION_IMAGE_DIR=./data/data
FASHION_RETRIEVAL_WEIGHTS_FILE=./retrieval_weights.json
```

Các thư mục được hiểu tương đối với `project/docker-compose.yml`. `.env` không
được lưu trên Git; máy mới dùng giá trị mặc định hoặc sao chép `.env.example`.
Không sửa tên đường dẫn theo số sản phẩm. Số lượng do CSV và embeddings quyết định.

Muốn thay số lượng (ví dụ 4.000), tự chạy từ `project/`:

```powershell
.\.venv\Scripts\python.exe -m src.prepare_dataset --num-products 4000
.\.venv\Scripts\python.exe -m src.generate_clip_embeddings `
  --csv data/processed/products.csv `
  --model runs/clip_finetune/best `
  --output-dir embeddings
docker compose up -d --build --force-recreate backend backend_2
```

Hai lệnh Python ghi lại catalog/embeddings web; lưu bản sao nếu cần giữ bộ cũ.
Tạm dừng hai backend trước khi ghi đè artifact đang dùng. Sau khi ghi đè nội dung
artifact, `up --build` có thể không tự tạo lại backend nếu image/config không đổi;
dùng `--force-recreate` như trên hoặc restart cả hai để nạp dữ liệu mới.
Khởi động hệ thống bình thường sau khi chuẩn bị xong vẫn chỉ cần một lệnh `up`.

Fine-tune là tùy chọn khi đã tải một bộ artifact đồng bộ. Nếu tự xây từ đầu, chạy
`src.finetune_clip`, sinh embeddings, rồi chạy `src.tune_retrieval_weights` để
ghi `project/retrieval_weights.json`. Catalog, embeddings, checkpoint và weights
phải cùng một pipeline; không sửa hash để bỏ qua kiểm tra.

Các thí nghiệm so sánh nằm tại `experiments/model_comparison/` ở cấp repository,
hoàn toàn độc lập và không phải đầu vào của website.

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

Khi chạy ngoài Docker, Vite mặc định proxy API tới `localhost:8000`. Trong
Compose, trình duyệt chỉ gọi cùng origin `localhost:80`; Web Reverse Proxy tự
định tuyến `/api` sang API Gateway.
`.env` dành cho Compose không tự cấu hình lệnh Python chạy trực tiếp; Python dùng
`src/config.py` và các biến `CLIP_MODEL_PATH`, `CLIP_EMBEDDING_DIR` nếu được đặt.

## Tài liệu và kiểm tra

- [Web Reverse Proxy](reverse-proxy/README.md)
- [API Gateway, load balancing và rate limiting](api-gateway/README.md)
- [Workflow và toàn bộ câu lệnh](../README.txt)
- [Thí nghiệm độc lập](../../experiments/model_comparison/README.md)

```powershell
docker compose config --quiet
docker compose exec reverse-proxy nginx -t
docker compose exec api-gateway nginx -t
docker compose logs -f reverse-proxy api-gateway backend backend_2
```

Reverse Proxy là service duy nhất publish port `80`; frontend, API Gateway và
hai backend chỉ dùng Docker network. Hai backend nạp model vào RAM riêng; tên service và đường API không đổi khi
chuyển thư mục. Backend build context vẫn là `project/` để lấy được `src/`;
frontend build context là `web/frontend/`.
