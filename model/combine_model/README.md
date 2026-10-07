Combine Model

Dataset mặc định là `../../project/data/processed/products.csv`, giống pipeline
chính. Có thể đổi đường dẫn bằng biến môi trường `FASHION_DATASET_PATH`. Khi có
cột `split`, phần đánh giá combine ưu tiên tập `test`.

Mô hình kết hợp tìm kiếm sản phẩm thời trang bằng:

ResNet18: Image Search
SBERT: Text Search
RRF: Multimodal Search

Cài đặt

Tạo và kích hoạt môi trường ảo:

python -m venv .venv

Windows PowerShell:

.venv\Scripts\activate

Cài đặt thư viện:

pip install -r requirements.txt

Chạy Combine Model

python main.py

Kiểm thử Model

python test.py
