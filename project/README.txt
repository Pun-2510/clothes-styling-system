HƯỚNG DẪN CHẠY HỆ THỐNG GỢI Ý SẢN PHẨM THỜI TRANG — BACKEND
====================================================================

1. ĐIỀU KIỆN TIÊN QUYẾT

- Python 3.12 để đồng nhất với base image python:3.12-slim của Dockerfile.
  Môi trường .venv hiện có có thể dùng để chạy lại nếu đã cài đủ thư viện.
- Docker Desktop cho Windows, bật Linux containers và cấu hình WSL 2 theo
  tài liệu chính thức. Khởi động Docker Desktop, đợi engine sẵn sàng.
  https://docs.docker.com/desktop/setup/install/windows-install/
  Docker Desktop có kèm Docker Compose:
  https://docs.docker.com/compose/install/
- Internet để tải dataset, thư viện, Docker image và model Hugging Face
  trong lần khởi chạy đầu. Tài khoản Kaggle nếu trang tải yêu cầu đăng nhập.
- Dataset: Mini Fashion Product Images and Text Dataset.
  Nguồn do tác giả dự án cung cấp:
  https://www.kaggle.com/datasets/nirmalsankalana/mini-product-image-and-text-dataset
- Dung lượng trống cho ảnh, môi trường Python, model và Docker image.
  Cấu hình Docker hiện tại chạy CPU; chưa cấu hình GPU passthrough.

2. TẢI VÀ ĐẶT DATASET

Mở link Kaggle ở mục 1, chọn Download và giải nén. Đặt CSV và thư mục ảnh
đúng cấu trúc sau (không lồng thêm một tầng thư mục ngoài ý muốn):

  project/
    data/
      data.csv
      data/
        3238.jpg
        ...

CSV của dự án có các cột: image, description, display name, category.
Tên ảnh trong cột image phải khớp tên file trong data/data/.
Nếu đã có gói data.zip của dự án, gói này chứa data.csv và data/*.jpg;
có thể giải nén vào data bằng lệnh dưới đây khi chưa có dữ liệu đích:
  Expand-Archive -LiteralPath .\data.zip -DestinationPath .\data
Không cần tải/giải nén lại nếu dữ liệu đã có đủ.

Kiểm tra:
  Test-Path .\data\data.csv
  Test-Path .\data\data

3. CHUẨN BỊ MÔI TRƯỜNG PYTHON

Chỉ tạo môi trường nếu chưa có .venv:
  py -3.12 -m venv .venv

Cài dependencies của dự án:
  .\.venv\Scripts\python.exe -m pip install -r requirements.txt

Các lệnh dùng trực tiếp Python trong .venv, không cần Activate.ps1.
requirements.txt ở gốc phục vụ toàn bộ pipeline; backend/requirements.txt
là danh sách được Dockerfile dùng cho API. Hai file đã pin phiên bản;
nếu pip không tìm thấy một phiên bản, cần kiểm tra index/Python/platform
và ghi nhận phiên bản đã dùng, không coi môi trường là đã cài thành công.

4. XỬ LÝ DỮ LIỆU

  .\.venv\Scripts\python.exe -m src.prepare_dataset

Script đọc data/data.csv, chuẩn hóa trường dữ liệu, kiểm tra file ảnh,
loại ảnh thiếu và bản ghi trùng ảnh, lấy mẫu theo category, rồi tạo:
  data/processed/products.csv

Các cột đầu ra: product_id, image_reference, product_name, description,
image_path, category. Cấu hình đường dẫn và lấy mẫu ở src/config.py.
MAX_PRODUCTS là mục tiêu lấy mẫu; làm tròn theo category có thể làm số
dòng cuối khác giá trị mục tiêu. Không giả định số hàng trước khi chạy.

5. TẠO IMAGE EMBEDDINGS VÀ TEXT EMBEDDINGS

Chạy lần lượt, cùng dùng một bản products.csv:
  .\.venv\Scripts\python.exe -m src.generate_image_embeddings
  .\.venv\Scripts\python.exe -m src.generate_text_embeddings

Kết quả:
  embeddings/image_embeddings.npy
  embeddings/text_embeddings.npy

Model: openai/clip-vit-base-patch32. Nhánh ảnh đọc image_path; nhánh
văn bản ghép tên, category và mô tả. Cả hai tạo vector CLIP chuẩn hóa L2.
Các hàng embedding phải giữ đúng thứ tự sản phẩm trong products.csv.
Nếu thay dataset hoặc chạy lại bước lấy mẫu, cần tạo lại cả hai file.
Nếu các file hiện tại đã đồng bộ, không cần sinh lại mỗi lần chạy API.

Kiểm tra shape mà không tải model:
  .\.venv\Scripts\python.exe -c "import numpy as np,pandas as pd; p=pd.read_csv('data/processed/products.csv'); a=np.load('embeddings/image_embeddings.npy'); b=np.load('embeddings/text_embeddings.npy'); print('products:',len(p),'image:',a.shape,'text:',b.shape); assert a.shape==b.shape==(len(p),512)"

6. CẤU TRÚC BACKEND

  backend/
    main.py                    Khởi tạo FastAPI, nạp model qua lifespan
    api.py                     Gom router dưới prefix /api
    dependencies.py            Cấp recommender/translator; báo lỗi 503
    serializers.py             Chuyển kết quả DataFrame thành response
    routers/
      health.py                Health và readiness
      recomendations.py        API ảnh và văn bản (tên file hiện tại)
    schemas/
      recommendation.py        Kiểu request/response và ràng buộc dữ liệu
    requirements.txt           Dependencies dùng trong Docker image
    Dockerfile                 Đóng gói API
  src/                         Tiền xử lý, CLIP, similarity, dịch, logging
  docker-compose.yml           Cấu hình service backend và network

Chạy trực tiếp để kiểm tra trước khi dùng Docker:
  .\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

API nạp CLIP và model dịch Helsinki-NLP/opus-mt-vi-en khi khởi động.
Đợi startup hoàn tất; lần đầu có thể mất thời gian tải model.
Mở terminal khác để kiểm tra các URL ở mục 8. Dừng bằng Ctrl+C trước
khi chạy Docker nếu cả hai cùng dùng cổng 8000.

7. BUILD VÀ CHẠY DOCKER BACKEND

Trước build phải có products.csv và thư mục embeddings từ các bước trên.
Dockerfile COPY các dữ liệu này vào image, không tự xử lý dữ liệu hay
tự sinh embeddings khi container khởi động.

Kiểm tra Docker Desktop/engine và cấu hình:
  docker version
  docker compose version
  docker compose config --quiet

Build và khởi động riêng backend:
  docker compose up -d --build backend

Theo dõi:
  docker compose ps backend
  docker compose logs -f backend

API truy cập tại http://localhost:8000. Cấu hình hiện tại dùng Python 3.12,
WORKDIR /app, chạy uvicorn backend.main:app --host 0.0.0.0; cổng mặc định
của Uvicorn là 8000, được Compose ánh xạ ra cổng 8000 trên máy.

Lưu ý đúng với cấu hình cơ bản hiện có:
- backend/ và src/ được bind mount. Dữ liệu processed/embeddings được COPY
  vào image, nên sau khi thay dữ liệu phải build lại backend.
- Chưa có volume model cache. Container tải model vào cache của nó; tạo
  lại container có thể phải tải lại. Cache model ở Windows không tự động
  được chia sẻ với container Linux.
- Chưa mount logs ra máy. Log file có thể đọc trong container bằng:
    docker compose exec backend sh -c "tail -n 50 /app/logs/recommendation.log"
- Không COPY ảnh catalog gốc vào image. API trả image_reference, không
  phục vụ URL ảnh. API ảnh mã hóa file được upload, tìm trên embeddings
  có sẵn, nên không cần đọc ảnh catalog gốc cho bước xếp hạng này.
- products.csv tạo ở Windows chứa image_path kiểu Windows. Đường dẫn đó
  không dùng được trong Linux nếu muốn chạy lại image encoder ở container.
  Quy trình hướng dẫn ở đây sinh embeddings trên máy trước khi build.

Dừng/khởi động lại riêng backend:
  docker compose stop backend
  docker compose start backend
Sau khi sửa code, nếu không tự cập nhật tiến trình:
  docker compose restart backend

8. KIỂM TRA VÀ GỌI API

Swagger UI: http://localhost:8000/docs
OpenAPI:   http://localhost:8000/openapi.json

Health:
  Invoke-RestMethod http://localhost:8000/api/health
Readiness:
  Invoke-RestMethod http://localhost:8000/api/health/ready

/health báo tiến trình API có phản hồi. /health/ready báo trạng thái
recommender, device, products, image_embeddings_ready,
text_embeddings_ready, translator_ready. Readiness có thể trả 200 khi
translator_ready=false; muốn tìm tiếng Việt phải kiểm tra trường này.

Văn bản tiếng Việt (JSON được mã hóa UTF-8 khi gửi):
  $payload = @{ query='váy nữ màu đỏ'; language='vi'; top_k=5; image_weight=0.5 } | ConvertTo-Json
  Invoke-RestMethod -Method Post -Uri http://localhost:8000/api/recommendations/text -ContentType 'application/json; charset=utf-8' -Body ([System.Text.Encoding]::UTF8.GetBytes($payload))

Tiếng Anh: đặt language='en', ví dụ query='a black leather handbag'.
Response có query, query_used, language, top_k, image_weight,
elapsed_seconds và items. Mỗi item có mã/tên/category/image_reference,
text_to_image_score, text_to_text_score và final_score.
elapsed_seconds của nhánh text chưa bao gồm thời gian dịch và toàn bộ HTTP.

Ảnh (thay đường dẫn sample bằng file thật, dung lượng tối đa 5 MiB):
  curl.exe -X POST http://localhost:8000/api/recommendations/image -F "file=@data/data/3238.jpg;type=image/jpeg" -F "top_k=5" -F "category_mode=no_category"

category_mode nhận no_category, hard_category hoặc soft_category.
top_k từ 1 đến 10; ảnh nhận MIME image/jpeg, image/png, image/webp.
Response có request_id, category_mode, predicted_category,
category_confidence, processing_times và items.

Mã lỗi cần kiểm tra:
  400: query trắng, file rỗng hoặc ảnh không đọc được.
  413: ảnh vượt quá giới hạn dung lượng.
  415: MIME của ảnh không được hỗ trợ.
  422: tham số không đúng schema, ngoài miền hoặc thiếu trường bắt buộc.
  503: recommender hoặc bộ dịch chưa sẵn sàng.
  500: lỗi trong quá trình xử lý recommendation.

9. TÁI HIỆN KIỂM CHỨNG CHO BÁO CÁO

Sau khi model đã được tải/cache trên máy và dataset/embeddings đã có:
  .\.venv\Scripts\python.exe scripts\verify_backend_report.py

Script dùng FastAPI TestClient với model thật và ghi response, HTTP status,
expected status, pass/fail, timestamp vào reports/evidence/*-backend-check.json.
Đây là kiểm tra chức năng cục bộ, không thay thế kiểm thử Docker hay tải đồng thời.
Script mặc định HF_HUB_OFFLINE=1; nếu chưa có model cache, chạy API với
Internet trước để tải model. Xem thêm logs/recommendation.log.

10. XỬ LÝ LỖI THƯỜNG GẶP

- Không tìm thấy Docker pipe/daemon: mở Docker Desktop, chờ engine Linux
  chạy rồi kiểm tra lại docker version.
- Thiếu products.csv/embeddings trong build: hoàn tất mục 4–5 tại đúng
  thư mục gốc rồi build lại; không chạy build context từ backend/.
- Lỗi tải model hoặc translator_ready=false: xem log, kiểm tra Internet
  và dependencies trong backend/requirements.txt.
- Không khớp số dòng embedding: tạo lại cả image và text từ cùng CSV.
- Cổng 8000 đang dùng: dừng tiến trình API cũ trước khi khởi động container.

PHẠM VI KIỂM CHỨNG
README mô tả mã nguồn hiện có và cách tái hiện. Tại thời điểm soạn báo cáo,
Compose qua kiểm tra cú pháp nhưng Docker Engine chưa chạy; chưa xác nhận
build/start container thành công. Cần bổ sung log chạy Docker khi kiểm thử.

Để nộp báo cáo theo quy định: cập nhật code, README, script kiểm chứng và
log lên repository đã chia sẻ với giảng viên; lưu thêm ảnh terminal của
lần chạy thực tế. Không đưa dữ liệu lớn/model/cache lên Git ngoài ý muốn.
