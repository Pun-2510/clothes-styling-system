# Model comparison experiments

Thư mục này chứa protocol đánh giá hiện tại cho catalog diverse 5.000 sản phẩm
trên ba seed độc lập: `42`, `123` và `2026`.

Tất cả lệnh trong tài liệu này chạy từ thư mục gốc `dacntt/`.

## 1. Ba nhóm so sánh

### Core model and modality comparison

`compare_full_gallery.py` tạo ba nhóm:

- `controlled_baselines`: ResNet18 image-only, SBERT text-only, pretrained
  CLIP, fine-tuned CLIP và ResNet18 + SBERT RRF.
- `clip_finetuning`: pretrained và fine-tuned CLIP trong paired image→text và
  text→image retrieval.
- `serving_ablation`: các score component và pretrained/fine-tuned multimodal
  raw fusion dưới cùng trọng số.

### Category-aware comparison

`compare_category_aware.py` giữ nguyên fine-tuned image cosine score và so sánh:

- Không sử dụng category.
- Hard predicted-category filter.
- Threshold soft category.
- Confidence-weighted soft category đang dùng khi serving.

### System pipeline comparison

`compare_system_pipeline.py` kiểm tra ba query mode và combined ablation:

- Image-only deployed.
- Text-only deployed.
- Combined deployed.
- Raw fusion, attribute contribution, category contribution và full ranking.

Mọi bảng báo cáo Precision@K, Recall@K, F1@K và NDCG@K tại
`K = 1, 5, 10`.

## 2. Kết quả đã hoàn tất

```text
results/diverse_seed_42/
results/diverse_seed_123/
results/diverse_seed_2026/
```

Trong mỗi seed:

```text
core_pretrained_multimodal_v1/
category_aware_v1/
system_pipeline_v1/
retrieval_weights.json
```

Tổng hợp ba seed:

```text
results/aggregate_diverse_3_seeds_v1/
├── core_pretrained_multimodal/
├── category_aware/
└── system_pipeline/
```

Mỗi thư mục tổng hợp có:

- `mean_std_summary.csv`: mean và sample standard deviation.
- `per_seed_results.csv`: toàn bộ số liệu từng seed.
- `aggregation.json`: provenance, catalog hash, query count và gallery size.

Không gộp bộ `aggregate_3_seeds/` dùng random sampling với bộ diverse vì hai
protocol có category quota, catalog và test queries khác nhau.

## 3. Kiểm tra nhanh code

```powershell
.\project\.venv\Scripts\python.exe -m experiments.model_comparison.compare_full_gallery --self-test
.\project\.venv\Scripts\python.exe -m experiments.model_comparison.compare_category_aware --self-test
.\project\.venv\Scripts\python.exe -m experiments.model_comparison.compare_system_pipeline --self-test
```

## 4. Chạy lại seed 42

Các output dưới đây dùng hậu tố `rerun_v1` để không ghi đè kết quả báo cáo.

### 4.1. Core comparison

```powershell
.\project\.venv\Scripts\python.exe -m experiments.model_comparison.compare_full_gallery `
  --run-dir .\project\runs\clip_finetune_diverse_seed_42 `
  --catalog-csv .\project\data\processed_diverse\products.csv `
  --expected-gallery-size 5000 `
  --finetuned-image-embeddings .\project\embeddings_diverse_seed_42\image_embeddings.npy `
  --cache-dir .\experiments\model_comparison\cache\diverse_seed_42 `
  --resnet-source project-trained `
  --device cpu `
  --batch-size 16 `
  --ks 1 5 10 `
  --clip-image-weight 0.5 `
  --clip-text-image-weight 0.5 `
  --output-dir .\experiments\model_comparison\results\diverse_seed_42\core_pretrained_multimodal_rerun_v1
```

### 4.2. Category-aware comparison

```powershell
.\project\.venv\Scripts\python.exe -m experiments.model_comparison.compare_category_aware `
  --run-dir .\project\runs\clip_finetune_diverse_seed_42 `
  --catalog-csv .\project\data\processed_diverse\products.csv `
  --image-embeddings .\project\embeddings_diverse_seed_42\image_embeddings.npy `
  --expected-gallery-size 5000 `
  --ks 1 5 10 `
  --category-top-per-class 3 `
  --category-confidence-floor 0.50 `
  --category-confidence-threshold 0.60 `
  --category-bonus 0.20 `
  --output-dir .\experiments\model_comparison\results\diverse_seed_42\category_aware_rerun_v1
```

### 4.3. System pipeline comparison

```powershell
$weights42 = (Get-Content .\experiments\model_comparison\results\diverse_seed_42\retrieval_weights.json -Raw | ConvertFrom-Json).weights

.\project\.venv\Scripts\python.exe -m experiments.model_comparison.compare_system_pipeline `
  --run-dir .\project\runs\clip_finetune_diverse_seed_42 `
  --catalog-csv .\project\data\processed_diverse\products.csv `
  --image-embeddings .\project\embeddings_diverse_seed_42\image_embeddings.npy `
  --text-embeddings .\project\embeddings_diverse_seed_42\text_embeddings.npy `
  --expected-gallery-size 5000 `
  --device cpu `
  --batch-size 16 `
  --ks 1 5 10 `
  --image-weight $weights42.COMBINED_IMAGE_WEIGHT `
  --text-image-weight $weights42.TEXT_RETRIEVAL_IMAGE_WEIGHT `
  --attribute-weight $weights42.COMBINED_ATTRIBUTE_WEIGHT `
  --image-category-bonus $weights42.CATEGORY_BONUS `
  --combined-category-bonus $weights42.COMBINED_CATEGORY_WEIGHT `
  --category-top-per-class 3 `
  --category-confidence-floor 0.50 `
  --category-confidence-threshold 0.60 `
  --output-dir .\experiments\model_comparison\results\diverse_seed_42\system_pipeline_rerun_v1
```

## 5. Chạy lại seed 123

### 5.1. Core comparison

```powershell
.\project\.venv\Scripts\python.exe -m experiments.model_comparison.compare_full_gallery `
  --run-dir .\project\runs\clip_finetune_diverse_seed_123 `
  --catalog-csv .\project\data\processed_diverse_seed_123\products.csv `
  --expected-gallery-size 5000 `
  --finetuned-image-embeddings .\project\embeddings_diverse_seed_123\image_embeddings.npy `
  --cache-dir .\experiments\model_comparison\cache\diverse_seed_123 `
  --resnet-source project-trained `
  --device cpu `
  --batch-size 16 `
  --ks 1 5 10 `
  --clip-image-weight 0.5 `
  --clip-text-image-weight 0.5 `
  --output-dir .\experiments\model_comparison\results\diverse_seed_123\core_pretrained_multimodal_rerun_v1
```

### 5.2. Category-aware comparison

```powershell
.\project\.venv\Scripts\python.exe -m experiments.model_comparison.compare_category_aware `
  --run-dir .\project\runs\clip_finetune_diverse_seed_123 `
  --catalog-csv .\project\data\processed_diverse_seed_123\products.csv `
  --image-embeddings .\project\embeddings_diverse_seed_123\image_embeddings.npy `
  --expected-gallery-size 5000 `
  --ks 1 5 10 `
  --category-top-per-class 3 `
  --category-confidence-floor 0.50 `
  --category-confidence-threshold 0.60 `
  --category-bonus 0.20 `
  --output-dir .\experiments\model_comparison\results\diverse_seed_123\category_aware_rerun_v1
```

### 5.3. System pipeline comparison

```powershell
$weights123 = (Get-Content .\experiments\model_comparison\results\diverse_seed_123\retrieval_weights.json -Raw | ConvertFrom-Json).weights

.\project\.venv\Scripts\python.exe -m experiments.model_comparison.compare_system_pipeline `
  --run-dir .\project\runs\clip_finetune_diverse_seed_123 `
  --catalog-csv .\project\data\processed_diverse_seed_123\products.csv `
  --image-embeddings .\project\embeddings_diverse_seed_123\image_embeddings.npy `
  --text-embeddings .\project\embeddings_diverse_seed_123\text_embeddings.npy `
  --expected-gallery-size 5000 `
  --device cpu `
  --batch-size 16 `
  --ks 1 5 10 `
  --image-weight $weights123.COMBINED_IMAGE_WEIGHT `
  --text-image-weight $weights123.TEXT_RETRIEVAL_IMAGE_WEIGHT `
  --attribute-weight $weights123.COMBINED_ATTRIBUTE_WEIGHT `
  --image-category-bonus $weights123.CATEGORY_BONUS `
  --combined-category-bonus $weights123.COMBINED_CATEGORY_WEIGHT `
  --category-top-per-class 3 `
  --category-confidence-floor 0.50 `
  --category-confidence-threshold 0.60 `
  --output-dir .\experiments\model_comparison\results\diverse_seed_123\system_pipeline_rerun_v1
```

## 6. Chạy lại seed 2026

### 6.1. Core comparison

```powershell
.\project\.venv\Scripts\python.exe -m experiments.model_comparison.compare_full_gallery `
  --run-dir .\project\runs\clip_finetune_diverse_seed_2026 `
  --catalog-csv .\project\data\processed_diverse_seed_2026\products.csv `
  --expected-gallery-size 5000 `
  --finetuned-image-embeddings .\project\embeddings_diverse_seed_2026\image_embeddings.npy `
  --cache-dir .\experiments\model_comparison\cache\diverse_seed_2026 `
  --resnet-source project-trained `
  --device cpu `
  --batch-size 16 `
  --ks 1 5 10 `
  --clip-image-weight 0.5 `
  --clip-text-image-weight 0.5 `
  --output-dir .\experiments\model_comparison\results\diverse_seed_2026\core_pretrained_multimodal_rerun_v1
```

### 6.2. Category-aware comparison

```powershell
.\project\.venv\Scripts\python.exe -m experiments.model_comparison.compare_category_aware `
  --run-dir .\project\runs\clip_finetune_diverse_seed_2026 `
  --catalog-csv .\project\data\processed_diverse_seed_2026\products.csv `
  --image-embeddings .\project\embeddings_diverse_seed_2026\image_embeddings.npy `
  --expected-gallery-size 5000 `
  --ks 1 5 10 `
  --category-top-per-class 3 `
  --category-confidence-floor 0.50 `
  --category-confidence-threshold 0.60 `
  --category-bonus 0.20 `
  --output-dir .\experiments\model_comparison\results\diverse_seed_2026\category_aware_rerun_v1
```

### 6.3. System pipeline comparison

```powershell
$weights2026 = (Get-Content .\experiments\model_comparison\results\diverse_seed_2026\retrieval_weights.json -Raw | ConvertFrom-Json).weights

.\project\.venv\Scripts\python.exe -m experiments.model_comparison.compare_system_pipeline `
  --run-dir .\project\runs\clip_finetune_diverse_seed_2026 `
  --catalog-csv .\project\data\processed_diverse_seed_2026\products.csv `
  --image-embeddings .\project\embeddings_diverse_seed_2026\image_embeddings.npy `
  --text-embeddings .\project\embeddings_diverse_seed_2026\text_embeddings.npy `
  --expected-gallery-size 5000 `
  --device cpu `
  --batch-size 16 `
  --ks 1 5 10 `
  --image-weight $weights2026.COMBINED_IMAGE_WEIGHT `
  --text-image-weight $weights2026.TEXT_RETRIEVAL_IMAGE_WEIGHT `
  --attribute-weight $weights2026.COMBINED_ATTRIBUTE_WEIGHT `
  --image-category-bonus $weights2026.CATEGORY_BONUS `
  --combined-category-bonus $weights2026.COMBINED_CATEGORY_WEIGHT `
  --category-top-per-class 3 `
  --category-confidence-floor 0.50 `
  --category-confidence-threshold 0.60 `
  --output-dir .\experiments\model_comparison\results\diverse_seed_2026\system_pipeline_rerun_v1
```

## 7. Tổng hợp ba seed vừa chạy lại

### 7.1. Core comparison

```powershell
.\project\.venv\Scripts\python.exe -m experiments.model_comparison.aggregate_seeds `
  --inputs `
    .\experiments\model_comparison\results\diverse_seed_42\core_pretrained_multimodal_rerun_v1 `
    .\experiments\model_comparison\results\diverse_seed_123\core_pretrained_multimodal_rerun_v1 `
    .\experiments\model_comparison\results\diverse_seed_2026\core_pretrained_multimodal_rerun_v1 `
  --output-dir .\experiments\model_comparison\results\aggregate_diverse_3_seeds_rerun_v1\core_pretrained_multimodal
```

### 7.2. Category-aware comparison

```powershell
.\project\.venv\Scripts\python.exe -m experiments.model_comparison.aggregate_seeds `
  --inputs `
    .\experiments\model_comparison\results\diverse_seed_42\category_aware_rerun_v1 `
    .\experiments\model_comparison\results\diverse_seed_123\category_aware_rerun_v1 `
    .\experiments\model_comparison\results\diverse_seed_2026\category_aware_rerun_v1 `
  --output-dir .\experiments\model_comparison\results\aggregate_diverse_3_seeds_rerun_v1\category_aware
```

### 7.3. System pipeline comparison

```powershell
.\project\.venv\Scripts\python.exe -m experiments.model_comparison.aggregate_seeds `
  --inputs `
    .\experiments\model_comparison\results\diverse_seed_42\system_pipeline_rerun_v1 `
    .\experiments\model_comparison\results\diverse_seed_123\system_pipeline_rerun_v1 `
    .\experiments\model_comparison\results\diverse_seed_2026\system_pipeline_rerun_v1 `
  --output-dir .\experiments\model_comparison\results\aggregate_diverse_3_seeds_rerun_v1\system_pipeline
```

`aggregate_seeds.py` chỉ đọc kết quả có sẵn; nó không prepare dataset, fine-tune,
sinh embeddings hoặc chạy lại comparison.
