# Standalone outfit recommendation prototype

Module này độc lập với luồng Similar Products trong `src/recommend.py`, API và
React web. Nó dùng split `disjoint` của Polyvore để trả về một outfit hoàn chỉnh
đã được phối trong dataset.

Đây là retrieval baseline: CLIP tìm món neo phù hợp với ảnh/văn bản đầu vào,
sau đó hệ thống xếp hạng các outfit có chứa món neo. Đây chưa phải compatibility
model được huấn luyện bằng positive và negative outfits.

## 1. Chuẩn bị index thử nghiệm

Chạy từ thư mục `project/`:

```powershell
.\.venv\Scripts\python.exe -m outfit_recommendation.prepare_index `
  --dataset-dir .\datasets\polyvore-outfits `
  --max-outfits 1000 `
  --batch-size 16
```

Lệnh đọc ảnh trực tiếp từ `data/disjoint/train.parquet`, chọn ngẫu nhiên có seed
và sinh artifact riêng tại `outfit_recommendation/artifacts/`. Lần đầu có thể
phải tải `openai/clip-vit-base-patch32`. Dùng `--device cuda` nếu môi trường
PyTorch đã hỗ trợ GPU. Dùng `--max-outfits 0` nếu muốn index toàn bộ split.

Nếu chạy lại trên cùng output, thêm `--overwrite`, hoặc đặt một `--output-dir`
mới để giữ lại thí nghiệm cũ.

## 2. Chạy Streamlit

```powershell
.\.venv\Scripts\python.exe -m streamlit run outfit_recommendation\app.py --server.port 8502
```

Mở `http://localhost:8502`. Port 8502 giúp tránh trùng với `app.py` cũ nếu app
Similar Products đang chạy ở port 8501.

Có thể cấu hình trong `.env`:

```env
POLYVORE_DATA_DIR=./datasets/polyvore-outfits
OUTFIT_ARTIFACT_DIR=./outfit_recommendation/artifacts
OUTFIT_CLIP_MODEL=openai/clip-vit-base-patch32
```

Thư mục dataset và artifact đều là dữ liệu local, không được commit lên Git.
