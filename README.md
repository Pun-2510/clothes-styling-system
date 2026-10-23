# Category-Aware Multimodal Fashion Retrieval System

Dự án xây dựng hệ thống tìm kiếm sản phẩm thời trang bằng ảnh, văn bản hoặc kết
hợp cả hai. CLIP được fine-tune trên dữ liệu thời trang và dùng chung không gian
embedding để xếp hạng sản phẩm. Hệ thống còn sử dụng tín hiệu category, màu sắc
và họa tiết để điều chỉnh kết quả truy xuất.

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
Sinh image embeddings và text embeddings
        ↓
Chọn trọng số retrieval trên validation set
        ↓
Đánh giá cuối trên test set (tùy chọn)
        ↓
Cấu hình .env
        ↓
Chạy website bằng Docker Compose
```

Luồng xử lý khi website hoạt động:

```text
Ảnh và/hoặc văn bản của người dùng
        ↓
React frontend với một form tìm kiếm thống nhất
        ↓
Nginx reverse proxy and load balancer
        ↓
Hai FastAPI backend
        ↓
Fine-tuned CLIP + raw cosine fusion
        ↓
Attribute-aware và category-aware ranking
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

Các lệnh chuẩn bị dữ liệu, fine-tuning, sinh embeddings và Docker được chạy trong
`project/`. Riêng pipeline thí nghiệm trong `experiments/` được chạy từ thư mục
gốc repository theo hướng dẫn tương ứng.

## Cấu trúc chính

```text
project/
├── src/                    Tiền xử lý, fine-tuning và recommendation
├── web/
│   ├── frontend/           ReactJS
│   ├── backend/            FastAPI
│   └── nginx/              Reverse proxy, load balancing và rate limiting
├── data/                   Dataset và catalog đã xử lý
├── runs/                   Checkpoint và metadata huấn luyện
├── embeddings/             Embeddings dùng cho website
├── retrieval_weights.json  Trọng số được chọn trên validation set
├── docker-compose.yml
└── README.txt              Hướng dẫn câu lệnh đầy đủ

experiments/model_comparison/
├── compare_full_gallery.py       So sánh model và modality
├── tune_serving_weights.py       Chọn trọng số trên validation set
└── compare_system_pipeline.py    Đánh giá pipeline cuối trên test set
```
