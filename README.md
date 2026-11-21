# Category-Aware Multimodal Fashion Retrieval System using CLIP

Hệ thống truy xuất sản phẩm thời trang bằng ảnh, văn bản hoặc kết hợp cả hai.
Fine-tuned CLIP tạo image/text embeddings trong cùng không gian; raw cosine
similarity, TF-IDF metadata matching và category evidence được dùng để xếp
hạng Top-K.

## Hai phạm vi độc lập

```text
<repository>/
├── project/       Pipeline vận hành và website độc lập
└── experiments/   So sánh mô hình và đánh giá nghiên cứu
```

`project/` không đọc checkpoint, embeddings hoặc retrieval weights từ
`experiments/`. Sau khi clone, pipeline vận hành tạo toàn bộ artifact tại:

```text
project/data/processed/products.csv
project/runs/clip_finetune/best/
project/embeddings/
project/validation/
project/retrieval_weights.json
```

Thư mục repository có thể mang bất kỳ tên nào. Chỉ cần chuyển vào `project/`
trước khi chạy các lệnh hệ thống.

## Workflow hệ thống

```text
Raw metadata + images
        ↓
Prepare catalog + category audit + audience audit
        ↓
Fine-tune CLIP; select best epoch on validation
        ↓
Generate catalog image/text embeddings
        ↓
Build TF-IDF metadata index from the processed catalog
        ↓
Tune retrieval weights on validation
        ↓
Browser :80 → Web Reverse Proxy → frontend / API Gateway → 2 FastAPI backends → Fine-tuned CLIP
```

Website hỗ trợ:

- Một form cho image-only, text-only và combined query.
- Crop thủ công vùng trang phục.
- Category-aware soft ranking cho ảnh và explicit category routing cho text.
- Duyệt catalog theo category hoặc nhóm metadata `Women / Men / Kids / Unisex`,
  trang chi tiết, lịch sử cục bộ và Developer Dashboard.
- Nhóm người dùng được suy ra bảo thủ từ tên sản phẩm và category; mô tả chỉ
  được dùng khi có bằng chứng rõ ràng. `age_group` và `gender_group` được audit
  riêng, còn trường hợp không chắc chắn được giữ ở `unknown`. Hệ thống không đoán
  giới tính từ ảnh. Mặc định nhãn này không tác động đến truy xuất. Khi người
  dùng chủ động chọn `Nữ / Nam / Trẻ em / Unisex` trong tùy chọn tìm kiếm, hệ
  thống chỉ xếp hạng các ứng viên thuộc nhóm đã chọn; cosine similarity và
  category-aware scoring vẫn được giữ nguyên.
- Nginx reverse proxy, least-connection load balancing và rate limiting.

## Bắt đầu

```powershell
git clone https://github.com/Pun-2510/clothes-styling-system.git
Set-Location .\clothes-styling-system\project
Get-Content .\README.txt -Encoding utf8
```

Tài liệu [project/README.txt](project/README.txt) hướng dẫn đầy đủ:

- Điều kiện tiên quyết và cấu trúc dataset.
- Prepare catalog với số lượng sản phẩm tùy chỉnh.
- Fine-tuning, sinh embeddings và chọn weights trên validation.
- Ý nghĩa các tham số như seed, batch size, epoch và category balance.
- Cấu hình `.env`, kiểm tra artifact và Docker Compose.
- Gói artifact Google Drive để chạy nhanh.

Sau khi đã chuẩn bị đủ artifact:

```powershell
docker compose config --quiet
docker compose up -d --build
```

- Website: `http://localhost`
- API/Swagger: `http://localhost/docs`
- Developer Dashboard: `http://localhost/developer`

## Thí nghiệm nghiên cứu

Các so sánh chính và baseline mở rộng (TF-IDF, BERT, SBERT, ResNet18,
EfficientNet, ViT, pretrained/fine-tuned CLIP), Category-Aware ranking và
system pipeline nằm riêng tại
[experiments/model_comparison/README.md](experiments/model_comparison/README.md).
Kết quả thí nghiệm không phải đầu vào để Docker chạy website.
