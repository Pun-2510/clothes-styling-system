# SBERT Fashion Embedding

## Dataset

Model dùng chung `../../project/data/processed/products.csv` với project. Các
cột huấn luyện là `product_name` và `category`; nếu có cột `split`, script giữ
nguyên hai tập `train` và `validation` của project.

## Cài đặt

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Chạy

### Fine-tune SBERT retrieval dùng cho tìm kiếm

`train_sbert_retrieval.py` fine-tune đúng SentenceTransformer dùng cho truy
hồi. Mặc định script train 3 epoch trên sáu lớp thời trang cân bằng, đánh giá
trước/sau train và ghi lịch sử từng epoch vào `logs/sbert_finetune_<timestamp>`.

```powershell
Set-Location -LiteralPath 'C:\DACNTT\model\SBERT_Model'
& 'C:\DACNTT\.venv\Scripts\python.exe' '.\train_sbert_retrieval.py' 2>&1 |
    Tee-Object -FilePath '.\logs\sbert_finetune_terminal.log'
```

Model đầu ra nằm tại `models/sbert_fashion_finetuned`. Để Web_Test dùng model
này, đặt `WEB_SBERT_MODEL` tới thư mục trên trước khi khởi động web.

### BERT classifier thử nghiệm

Huấn luyện model:

```bash
python train.py
```

Tạo embedding:

```bash
python embedding.py
```

Tìm kiếm, dự đoán:

```bash
python predict.py
```

Embedding sau khi tạo được dùng cho tìm kiếm và so sánh sản phẩm.

Thoát chương trình:

nhập 'exit'
