# Hệ thống gợi ý sản phẩm thời trang đa phương thức

Dự án xây dựng hệ thống tìm kiếm và gợi ý sản phẩm bằng ảnh hoặc văn bản. CLIP
được fine-tune trên dữ liệu thời trang, sau đó dùng chung không gian embedding để
xếp hạng các sản phẩm gần với truy vấn của người dùng.

## Workflow của hệ thống

Quy trình chuẩn bị model và dữ liệu:

```text
Dataset ảnh + mô tả
        ↓
Prepare dataset và audit category
        ↓
Chia train / validation / test
        ↓
Fine-tune CLIP và chọn checkpoint tốt nhất
        ↓
So sánh pretrained / fine-tuned CLIP (tùy chọn)
        ↓
Sinh image embeddings và text embeddings
        ↓
Cấu hình .env
        ↓
Chạy website bằng Docker Compose
```

Luồng xử lý khi website hoạt động:

```text
Ảnh hoặc văn bản của người dùng
        ↓
React frontend
        ↓
Nginx API Gateway
        ↓
Hai FastAPI backend
        ↓
CLIP + similarity search + category policy
        ↓
Danh sách sản phẩm được xếp hạng
```

## Bắt đầu

Sau khi clone repository, chuyển vào thư mục `project/`:

```powershell
Set-Location .\project
```

Toàn bộ điều kiện, cấu trúc dataset và câu lệnh cho hai cách chạy — tải artifact
có sẵn hoặc tự prepare/fine-tune — nằm trong
[project/README.txt](project/README.txt).

Có thể đọc trực tiếp trong PowerShell:

```powershell
Get-Content .\README.txt
```

Không chạy các lệnh ML hoặc Docker từ thư mục ngoài `project/`, vì các đường dẫn
tương đối trong cấu hình được thiết kế dựa trên thư mục này.

## Cấu trúc chính

```text
project/
├── src/                    Tiền xử lý, fine-tuning và recommendation
├── comparisons/            Thí nghiệm so sánh model, độc lập với website
├── web/
│   ├── frontend/           ReactJS
│   ├── backend/            FastAPI
│   └── api-gateway/        Nginx và load balancing
├── data/                   Dataset và catalog đã xử lý
├── runs/                   Checkpoint và metadata huấn luyện
├── embeddings/             Embeddings dùng cho website
├── docker-compose.yml
└── README.txt              Hướng dẫn câu lệnh đầy đủ
```

