# Category-Aware Multimodal Fashion Retrieval System using CLIP

Hệ thống truy xuất sản phẩm thời trang bằng ảnh, văn bản hoặc kết hợp cả hai.
Fine-tuned CLIP tạo image/text embeddings trong cùng không gian; raw cosine
similarity, thuộc tính văn bản và category confidence được dùng để xếp hạng
Top-K sản phẩm. Frontend tự nhận biết loại truy vấn từ input, không yêu cầu
người dùng chọn image-only, text-only hay combined.

## Pipeline hiện tại

```text
43.662 sản phẩm hợp lệ
        ↓
Category-balanced diverse sampling → catalog 5.000 sản phẩm
        ↓
Category audit + train/validation/test split
        ↓
Fine-tune CLIP và chọn best checkpoint trên validation
        ↓
Sinh image_embeddings.npy + text_embeddings.npy
        ↓
Chọn retrieval weights trên validation
        ↓
Đánh giá test trên 3 seed: 42, 123, 2026
        ↓
React → Nginx → 2 FastAPI backends → Fine-tuned CLIP
```

Website hỗ trợ:

- Một form thống nhất cho ảnh, văn bản và truy vấn kết hợp.
- Crop thủ công vùng trang phục trước khi tìm bằng ảnh.
- Gợi ý bổ sung mô tả khi kết quả image-only chưa đúng.
- Category-aware ranking với confidence-weighted soft bonus.
- Duyệt catalog, xem chi tiết sản phẩm và lịch sử tìm kiếm trên thiết bị.
- Developer dashboard hiển thị health, model artifacts và ranking formulas.

## Artifact đang được website sử dụng

Cấu hình triển khai hiện tại nằm trong `project/.env`:

```text
Catalog:     project/data/processed_diverse/products.csv
Images:      project/data/data/
Embeddings:  project/embeddings_diverse_seed_42/
Checkpoint:  project/runs/clip_finetune_diverse_seed_42/best/
Weights:     experiments/model_comparison/results/diverse_seed_42/retrieval_weights.json
```

Các artifact trên phải thuộc cùng catalog và checkpoint. Không trộn
`products.csv`, embeddings, checkpoint hoặc retrieval weights của seed khác.

## Chạy website

```powershell
Set-Location .\project
docker compose config --quiet
docker compose up -d --build
```

Địa chỉ:

- Website: `http://localhost:5173`
- API qua Nginx: `http://localhost:8000`
- Swagger: `http://localhost:8000/docs`

Kiểm tra:

```powershell
docker compose ps
Invoke-RestMethod http://localhost:8000/api/health/ready
```

## Hướng dẫn đầy đủ

Sau khi clone repository, chuyển vào `project/` và đọc
[project/README.txt](project/README.txt). Tài liệu này chứa:

- Cấu trúc dataset và môi trường Python.
- Lệnh diverse sampling, fine-tuning và sinh embeddings.
- Lệnh chọn retrieval weights trên validation.
- Cấu hình `.env` và Docker Compose.
- Vị trí kết quả thí nghiệm ba seed.

```powershell
Set-Location .\project
Get-Content .\README.txt -Encoding utf8
```

Thí nghiệm được chạy từ thư mục gốc repository. Kết quả tổng hợp chính:

```text
experiments/model_comparison/results/aggregate_diverse_3_seeds_v1/
├── core_pretrained_multimodal/
├── category_aware/
└── system_pipeline/
```

## Cấu trúc chính

```text
dacntt/
├── project/
│   ├── src/                         Data, training và retrieval logic
│   ├── web/
│   │   ├── frontend/                React
│   │   ├── backend/                 FastAPI
│   │   └── nginx/                   Reverse proxy, load balancer, rate limit
│   ├── data/processed_diverse/      Catalog seed 42 đang triển khai
│   ├── runs/clip_finetune_diverse_seed_42/
│   ├── embeddings_diverse_seed_42/
│   ├── .env
│   ├── docker-compose.yml
│   └── README.txt
└── experiments/model_comparison/
    ├── compare_full_gallery.py
    ├── compare_category_aware.py
    ├── compare_system_pipeline.py
    ├── tune_serving_weights.py
    └── aggregate_seeds.py
```

## Ghi chú đánh giá

- Test gallery có đúng 5.000 sản phẩm.
- Mỗi seed dùng catalog, split, checkpoint, embeddings và validation weights
  tương ứng.
- Báo cáo Precision@K, Recall@K, F1@K và NDCG@K tại `K = 1, 5, 10`.
- Bộ random sampling cũ và diverse sampling mới là hai protocol khác nhau;
  không gộp chúng thành sáu seed để tính một giá trị trung bình.
