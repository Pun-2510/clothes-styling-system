# Fine-tuning CLIP và so sánh Recall

Trước thay đổi này, `src/` chỉ dùng `openai/clip-vit-base-patch32` để inference.
`recall_at_k()` đã tồn tại nhưng chưa được nối vào một quy trình đánh giá.
Pipeline mới cập nhật trọng số CLIP bằng cặp ảnh–mô tả từ `data/processed/products.csv`.

## 1. Huấn luyện

Chạy tại thư mục gốc dự án, dùng các thư viện trong môi trường `.venv` hiện có:

```powershell
# Tùy chọn: chỉ kiểm tra ảnh và tạo tập train/validation/test, chưa tải model.
.\.venv\Scripts\python.exe -m src.finetune_clip --prepare-only

# Fine-tune cả image encoder, text encoder và projection layers.
.\.venv\Scripts\python.exe -m src.finetune_clip --epochs 5 --batch-size 16 --learning-rate 1e-6
```

Mặc định chọn CUDA nếu PyTorch nhận GPU, ngược lại dùng CPU. Môi trường được
kiểm tra khi bổ sung tính năng là `torch 2.13.0+cpu`, không nhận CUDA.
Full fine-tuning trên CPU có thể mất nhiều thời gian. Nếu hết bộ nhớ, giảm
`--batch-size` xuống 4 hoặc 2. Batch lớn hơn cung cấp nhiều negative hơn.

Có thể thử cách nhẹ hơn, chỉ cập nhật hai projection layers và `logit_scale`:

```powershell
.\.venv\Scripts\python.exe -m src.finetune_clip --run-dir runs/clip_projection --trainable projections --epochs 5 --batch-size 16 --learning-rate 1e-5
```

Chế độ `projections` vẫn học trọng số, nhưng giữ nguyên hai encoder; không tương
đương full fine-tuning. Mỗi thí nghiệm cần một `--run-dir` riêng. Một thư mục đã
có huấn luyện sẽ không bị ghi đè; chưa hỗ trợ resume optimizer giữa chừng.
Thư mục mới tạo bằng `--prepare-only` có thể dùng tiếp với cùng CSV/seed/tỷ lệ chia.

Quy trình:

- Kiểm tra ảnh, loại ảnh có nội dung file trùng hệt nhau trong bản dữ liệu thí nghiệm.
  Không sửa catalog gốc. Ảnh hỏng/thiếu làm lệnh dừng với lỗi.
- Chia khoảng 80% train, 10% validation, 10% test theo category với seed 42.
  Cùng `product_id` hoặc cùng prompt được giữ chung một tập. Category có ít hơn
  ba nhóm được giữ trong train, nên tỷ lệ thực tế có thể khác 80/10/10.
  Chưa dò ảnh gần giống nhưng khác nội dung file; nên rà soát các biến thể sản phẩm
  và cung cấp cùng product_id khi chúng thuộc cùng một sản phẩm.
- Prompt dùng `product_name`, `category`, `description`, giống lúc tạo text embedding.
  Text được cắt theo giới hạn token của model. Dataset hiện tại dùng tiếng Anh;
  pipeline không tự dịch dataset tiếng Việt.
- Học bằng symmetric contrastive loss và AdamW, có gradient clipping.
  Cặp cùng sản phẩm/cùng prompt trong một batch được xem là nhiều positive.
- Chọn checkpoint tốt nhất theo trung bình Recall@1 text→image và image→text
  trên validation. Test không tham gia cập nhật trọng số hoặc chọn epoch.

File được tạo trong `runs/clip_finetune/`:

| File | Nội dung |
| --- | --- |
| `dataset.csv` | Danh sách sản phẩm, fingerprint ảnh và tập train/validation/test |
| `experiment.json` | Seed, tỷ lệ, số lượng, fingerprint CSV |
| `training.json` | Loss từng epoch, Recall validation, cấu hình và epoch tốt nhất |
| `best/` | Model và processor dùng được với `from_pretrained()` |

Lần kiểm tra dữ liệu hiện tại: 5.015 dòng đầu vào, loại 11 ảnh trùng, còn
4.036 train / 484 validation / 484 test. Đây là số lượng dữ liệu, chưa phải kết quả học.

## 2. So sánh pretrained và fine-tuned

```powershell
.\.venv\Scripts\python.exe -m src.compare_clip --run-dir runs/clip_finetune --ks 1 5 10
```

Lệnh nạp lần lượt model gốc dùng để khởi tạo huấn luyện và checkpoint `best/`,
tính lại embedding trên **cùng tập test, cùng query, cùng gallery, cùng K**.
Gallery chỉ gồm sản phẩm test. Không cộng category bonus hay lọc theo category
dự đoán trong phép so sánh này; mục tiêu là đo thay đổi của embedding CLIP.

Xuất bảng trên terminal và lưu `comparison.csv`, `comparison.json`:

| task | Ý nghĩa positive |
| --- | --- |
| `text_to_image` | Ảnh có cùng product_id với mô tả truy vấn |
| `image_to_text` | Mô tả có cùng product_id với ảnh truy vấn |
| `image_to_image_category` | Ảnh cùng category thực tế, loại tất cả ảnh của chính sản phẩm truy vấn |

`Recall@K = số positive trong Top-K / tổng positive có trong gallery`.
Lấy trung bình Recall trên các query hợp lệ, báo cả số query bị bỏ qua vì không
có positive. Nếu không có query hợp lệ, Recall là `null`, không giả định bằng 0.
Nếu K lớn hơn số ứng viên, lấy toàn bộ ứng viên có thể trả về.

Với một ảnh/mô tả cho mỗi sản phẩm, Recall text↔image là tỷ lệ query tìm đúng
sản phẩm trong Top-K. Với image→image category, một query có thể có nhiều positive:
ví dụ tìm được 4 trong tổng 100 ảnh cùng category thì Recall@5 = 4%, dù 4/5 kết quả
đúng category. Vì vậy Recall này khác Precision@K hoặc tỷ lệ query có ít nhất một hit.

Category là nhãn thay thế cho mức độ liên quan, chưa phải đánh giá của người dùng.
Điểm text↔image dùng mô tả catalog tiếng Anh, chưa đo chất lượng truy vấn tiếng Việt
tự do, chất lượng dịch hoặc toàn bộ pipeline reranking. Không suy ra Recall từ
category do chính model dự đoán cho ảnh người dùng tải lên khi chưa có ground truth.

`pretrained` và `finetuned` nằm trong [0, 1]. `delta_percentage_points` là
`100 × (finetuned − pretrained)`, dương là cải thiện, âm là giảm chất lượng.
Fine-tuning không đảm bảo tăng Recall; so sánh test chỉ sau khi chốt cấu hình bằng
validation. Không dùng điểm test để chọn lại learning rate/epoch.

## 3. Dùng checkpoint fine-tuned trong ứng dụng

Tạo lại **cả image và text embedding** cho toàn bộ catalog bằng cùng checkpoint:

```powershell
.\.venv\Scripts\python.exe -m src.generate_clip_embeddings --model runs/clip_finetune/best --output-dir embeddings/finetuned
```

Hai lệnh cũ `src.generate_image_embeddings` và `src.generate_text_embeddings`
vẫn dùng được, nay cũng nhận `--model`, `--output-dir`, `--csv`, `--batch-size`.
Mỗi `.npy` mới có file `.npy.json` ghi nhận model và thứ tự catalog để phát hiện
việc trộn embedding/model hoặc đổi CSV. Ảnh hỏng làm lệnh dừng, không ghi zero vector.
Embedding pretrained cũ chưa có metadata vẫn được hỗ trợ khi dùng model mặc định.

Chọn model và thư mục embedding trước khi khởi động lại backend hoặc Streamlit:

```powershell
$env:CLIP_MODEL_PATH = 'D:\project\runs\clip_finetune\best'
$env:CLIP_EMBEDDING_DIR = 'D:\project\embeddings\finetuned'
.\.venv\Scripts\python.exe -m streamlit run app.py
# Hoặc chạy lệnh khởi động backend hiện tại trong cùng terminal.
```

Cũng có thể gọi trực tiếp trong Python:

```python
FashionRecommender(
    model_source="runs/clip_finetune/best",
    embedding_dir="embeddings/finetuned",
)
```

Muốn quay lại pretrained: bỏ hai biến môi trường rồi khởi động lại ứng dụng
(cần có catalog embedding pretrained ở `embeddings/`). Docker Compose hiện bind
mount read-only checkpoint `runs/clip_finetune_3epochs_local/best`, thư mục
`embeddings_finetuned` và `data/processed`; sau khi tạo lại các artifact này chỉ
cần `docker compose restart backend`. Nếu dùng run hoặc thư mục embedding khác,
cập nhật đồng thời source mount, `CLIP_MODEL_PATH` và `CLIP_EMBEDDING_DIR` trong
`docker-compose.yml`; biến PowerShell không tự được truyền vào Docker Compose.

## 4. Kiểm thử

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Kiểm thử offline dùng CLIP rất nhỏ khởi tạo ngẫu nhiên và ảnh tổng hợp để kiểm tra
gradient cập nhật trọng số, lưu/nạp checkpoint, so sánh và tạo catalog embedding.
Các test khác kiểm tra Recall với thứ hạng biết trước, tránh self-match, tập chia
không trùng nhóm và từ chối model/catalog không khớp. Điểm trong test tổng hợp
không phải kết quả pretrained/fine-tuned trên dataset thật.

Tham khảo API: [Hugging Face CLIP](https://huggingface.co/docs/transformers/model_doc/clip).
