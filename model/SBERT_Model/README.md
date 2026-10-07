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
