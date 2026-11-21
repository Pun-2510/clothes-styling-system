# Model comparison experiments

> **Phạm vi riêng:** thư mục này chỉ dùng cho đánh giá nghiên cứu và kết quả
> nhiều seed. Website trong `project/` không đọc catalog, embeddings, checkpoint
> hoặc weight JSON từ `experiments/`. Pipeline vận hành và lệnh validation dùng
> cho website nằm trong `project/README.txt` và chạy bằng
> `python -m src.tune_retrieval_weights` từ thư mục `project/`.

Các lệnh bên dưới được chạy từ thư mục gốc repository vì chúng cần đồng thời
truy cập code thí nghiệm và artifact của seed được đánh giá. Tên thư mục clone
không cần là `dacntt`.

Thư mục này chứa protocol đánh giá hiện tại cho catalog diverse 5.000 sản phẩm
trên ba seed độc lập: `42`, `123` và `2026`.

Tất cả lệnh trong tài liệu này chạy từ thư mục gốc `dacntt/`.

## 1. Ba nhóm so sánh

### Core model and modality comparison

`compare_full_gallery.py` tạo ba nhóm:

- `controlled_baselines`: TF-IDF, pretrained BERT, SBERT và CLIP cho text-only;
  ResNet18, ImageNet EfficientNet-B0, ImageNet ViT-B/16 và CLIP cho image-only;
  cùng với ResNet18 + SBERT RRF cho multimodal độc lập.
- `clip_finetuning`: pretrained và fine-tuned CLIP trong paired image→text và
  text→image retrieval.
- `serving_ablation`: các score component và pretrained/fine-tuned multimodal
  raw fusion dưới cùng trọng số.

### Category-aware comparison

`compare_category_aware.py` giữ nguyên fine-tuned image cosine score và so sánh:

- Không sử dụng category.
- Hard predicted-category filter.
- Threshold soft category.
- Confidence-weighted soft category.
- Dynamic relation policy dùng prompt ensemble và quan hệ Top-2 category,
  tương ứng với image-only serving hiện tại.

### System pipeline comparison

`compare_system_pipeline.py` kiểm tra ba query mode và combined ablation:

- Image-only deployed.
- Text-only deployed.
- Combined deployed.
- Raw CLIP fusion, TF-IDF metadata refinement, category contribution và full
  ranking.

Mọi bảng báo cáo Precision@K, Recall@K, F1@K, NDCG@K và MAP@K tại
`K = 1, 5, 10`.

MAP@K thưởng cho hệ thống đưa các sản phẩm relevant lên sớm và liên tục trong
danh sách Top-K. Với paired retrieval chỉ có một relevant target, MAP@K tương
đương reciprocal rank bị giới hạn tại K.

## 2. Kết quả đã hoàn tất

```text
results/diverse_seed_42/
results/diverse_seed_123/
results/diverse_seed_2026/
```

Trong mỗi seed:

```text
core_extended_pretrained_map_v1/
weight_tuning_tfidf_prompt_v3/
category_aware_tfidf_prompt_v3/
system_pipeline_tfidf_prompt_v3/
```

Tổng hợp ba seed:

```text
results/aggregate_diverse_3_seeds_map_v1/
└── core_extended_pretrained/

results/aggregate_diverse_3_seeds_tfidf_prompt_v3/
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

Cài các dependency của benchmark trước lần chạy đầu tiên:

```powershell
.\project\.venv\Scripts\python.exe -m pip install -r .\experiments\model_comparison\requirements.txt
```

```powershell
.\project\.venv\Scripts\python.exe -m experiments.model_comparison.compare_full_gallery --self-test
.\project\.venv\Scripts\python.exe -m experiments.model_comparison.compare_category_aware --self-test
.\project\.venv\Scripts\python.exe -m experiments.model_comparison.compare_system_pipeline --self-test
```

## 4. Chạy lại seed 42

Core comparison giữ hậu tố `map_v1`. Category-aware và System Pipeline hiện
dùng hậu tố `tfidf_prompt_v3`. Mỗi output đều tách khỏi protocol cũ vì thư mục
kết quả của chương trình phải rỗng trước khi chạy.

### 4.1. Core comparison

```powershell
.\project\.venv\Scripts\python.exe -m experiments.model_comparison.compare_full_gallery `
  --run-dir .\project\runs\clip_finetune_diverse_seed_42 `
  --catalog-csv .\project\data\processed_diverse\products.csv `
  --expected-gallery-size 5000 `
  --finetuned-image-embeddings .\project\embeddings_diverse_seed_42\image_embeddings.npy `
  --cache-dir .\experiments\model_comparison\cache\diverse_seed_42 `
  --resnet-source imagenet `
  --bert-model google-bert/bert-base-uncased `
  --bert-max-length 128 `
  --tfidf-max-features 20000 `
  --device cpu `
  --batch-size 16 `
  --ks 1 5 10 `
  --clip-image-weight 0.5 `
  --clip-text-image-weight 0.5 `
  --output-dir .\experiments\model_comparison\results\diverse_seed_42\core_extended_pretrained_map_v1
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
  --category-prototype-weight 1.0 `
  --category-relation-neighbors 5 `
  --related-top1-bonus-factor 0.5 `
  --related-top2-bonus-factor 0.2 `
  --unrelated-top1-bonus-factor 0.1 `
  --output-dir .\experiments\model_comparison\results\diverse_seed_42\category_aware_tfidf_prompt_v3
```

### 4.3. System pipeline comparison

```powershell
$weights42 = (Get-Content .\experiments\model_comparison\results\diverse_seed_42\weight_tuning_tfidf_prompt_v3\retrieval_weights.json -Raw | ConvertFrom-Json).weights

.\project\.venv\Scripts\python.exe -m experiments.model_comparison.compare_system_pipeline `
  --run-dir .\project\runs\clip_finetune_diverse_seed_42 `
  --catalog-csv .\project\data\processed_diverse\products.csv `
  --image-embeddings .\project\embeddings_diverse_seed_42\image_embeddings.npy `
  --text-embeddings .\project\embeddings_diverse_seed_42\text_embeddings.npy `
  --metadata-index-dir .\project\metadata_index_diverse_seed_42 `
  --expected-gallery-size 5000 `
  --device cpu `
  --batch-size 16 `
  --ks 1 5 10 `
  --image-weight $weights42.COMBINED_IMAGE_WEIGHT `
  --text-image-weight $weights42.TEXT_RETRIEVAL_IMAGE_WEIGHT `
  --text-metadata-weight $weights42.TEXT_METADATA_WEIGHT `
  --combined-metadata-weight $weights42.COMBINED_METADATA_WEIGHT `
  --image-category-bonus $weights42.CATEGORY_BONUS `
  --combined-category-bonus $weights42.COMBINED_CATEGORY_WEIGHT `
  --category-top-per-class 3 `
  --category-confidence-floor 0.50 `
  --category-confidence-threshold 0.60 `
  --category-prototype-weight 1.0 `
  --category-relation-neighbors 5 `
  --related-top1-bonus-factor 0.5 `
  --related-top2-bonus-factor 0.2 `
  --unrelated-top1-bonus-factor 0.1 `
  --output-dir .\experiments\model_comparison\results\diverse_seed_42\system_pipeline_tfidf_prompt_v3
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
  --resnet-source imagenet `
  --bert-model google-bert/bert-base-uncased `
  --bert-max-length 128 `
  --tfidf-max-features 20000 `
  --device cpu `
  --batch-size 16 `
  --ks 1 5 10 `
  --clip-image-weight 0.5 `
  --clip-text-image-weight 0.5 `
  --output-dir .\experiments\model_comparison\results\diverse_seed_123\core_extended_pretrained_map_v1
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
  --category-prototype-weight 1.0 `
  --category-relation-neighbors 5 `
  --related-top1-bonus-factor 0.5 `
  --related-top2-bonus-factor 0.2 `
  --unrelated-top1-bonus-factor 0.1 `
  --output-dir .\experiments\model_comparison\results\diverse_seed_123\category_aware_tfidf_prompt_v3
```

### 5.3. System pipeline comparison

```powershell
$weights123 = (Get-Content .\experiments\model_comparison\results\diverse_seed_123\weight_tuning_tfidf_prompt_v3\retrieval_weights.json -Raw | ConvertFrom-Json).weights

.\project\.venv\Scripts\python.exe -m experiments.model_comparison.compare_system_pipeline `
  --run-dir .\project\runs\clip_finetune_diverse_seed_123 `
  --catalog-csv .\project\data\processed_diverse_seed_123\products.csv `
  --image-embeddings .\project\embeddings_diverse_seed_123\image_embeddings.npy `
  --text-embeddings .\project\embeddings_diverse_seed_123\text_embeddings.npy `
  --metadata-index-dir .\project\metadata_index_diverse_seed_123 `
  --expected-gallery-size 5000 `
  --device cpu `
  --batch-size 16 `
  --ks 1 5 10 `
  --image-weight $weights123.COMBINED_IMAGE_WEIGHT `
  --text-image-weight $weights123.TEXT_RETRIEVAL_IMAGE_WEIGHT `
  --text-metadata-weight $weights123.TEXT_METADATA_WEIGHT `
  --combined-metadata-weight $weights123.COMBINED_METADATA_WEIGHT `
  --image-category-bonus $weights123.CATEGORY_BONUS `
  --combined-category-bonus $weights123.COMBINED_CATEGORY_WEIGHT `
  --category-top-per-class 3 `
  --category-confidence-floor 0.50 `
  --category-confidence-threshold 0.60 `
  --category-prototype-weight 1.0 `
  --category-relation-neighbors 5 `
  --related-top1-bonus-factor 0.5 `
  --related-top2-bonus-factor 0.2 `
  --unrelated-top1-bonus-factor 0.1 `
  --output-dir .\experiments\model_comparison\results\diverse_seed_123\system_pipeline_tfidf_prompt_v3
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
  --resnet-source imagenet `
  --bert-model google-bert/bert-base-uncased `
  --bert-max-length 128 `
  --tfidf-max-features 20000 `
  --device cpu `
  --batch-size 16 `
  --ks 1 5 10 `
  --clip-image-weight 0.5 `
  --clip-text-image-weight 0.5 `
  --output-dir .\experiments\model_comparison\results\diverse_seed_2026\core_extended_pretrained_map_v1
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
  --category-prototype-weight 1.0 `
  --category-relation-neighbors 5 `
  --related-top1-bonus-factor 0.5 `
  --related-top2-bonus-factor 0.2 `
  --unrelated-top1-bonus-factor 0.1 `
  --output-dir .\experiments\model_comparison\results\diverse_seed_2026\category_aware_tfidf_prompt_v3
```

### 6.3. System pipeline comparison

```powershell
$weights2026 = (Get-Content .\experiments\model_comparison\results\diverse_seed_2026\weight_tuning_tfidf_prompt_v3\retrieval_weights.json -Raw | ConvertFrom-Json).weights

.\project\.venv\Scripts\python.exe -m experiments.model_comparison.compare_system_pipeline `
  --run-dir .\project\runs\clip_finetune_diverse_seed_2026 `
  --catalog-csv .\project\data\processed_diverse_seed_2026\products.csv `
  --image-embeddings .\project\embeddings_diverse_seed_2026\image_embeddings.npy `
  --text-embeddings .\project\embeddings_diverse_seed_2026\text_embeddings.npy `
  --metadata-index-dir .\project\metadata_index_diverse_seed_2026 `
  --expected-gallery-size 5000 `
  --device cpu `
  --batch-size 16 `
  --ks 1 5 10 `
  --image-weight $weights2026.COMBINED_IMAGE_WEIGHT `
  --text-image-weight $weights2026.TEXT_RETRIEVAL_IMAGE_WEIGHT `
  --text-metadata-weight $weights2026.TEXT_METADATA_WEIGHT `
  --combined-metadata-weight $weights2026.COMBINED_METADATA_WEIGHT `
  --image-category-bonus $weights2026.CATEGORY_BONUS `
  --combined-category-bonus $weights2026.COMBINED_CATEGORY_WEIGHT `
  --category-top-per-class 3 `
  --category-confidence-floor 0.50 `
  --category-confidence-threshold 0.60 `
  --category-prototype-weight 1.0 `
  --category-relation-neighbors 5 `
  --related-top1-bonus-factor 0.5 `
  --related-top2-bonus-factor 0.2 `
  --unrelated-top1-bonus-factor 0.1 `
  --output-dir .\experiments\model_comparison\results\diverse_seed_2026\system_pipeline_tfidf_prompt_v3
```

## 7. Tổng hợp ba seed vừa chạy lại

### 7.1. Core comparison

```powershell
.\project\.venv\Scripts\python.exe -m experiments.model_comparison.aggregate_seeds `
  --inputs `
    .\experiments\model_comparison\results\diverse_seed_42\core_extended_pretrained_map_v1 `
    .\experiments\model_comparison\results\diverse_seed_123\core_extended_pretrained_map_v1 `
    .\experiments\model_comparison\results\diverse_seed_2026\core_extended_pretrained_map_v1 `
  --output-dir .\experiments\model_comparison\results\aggregate_diverse_3_seeds_map_v1\core_extended_pretrained
```

### 7.2. Category-aware comparison

```powershell
.\project\.venv\Scripts\python.exe -m experiments.model_comparison.aggregate_seeds `
  --inputs `
    .\experiments\model_comparison\results\diverse_seed_42\category_aware_tfidf_prompt_v3 `
    .\experiments\model_comparison\results\diverse_seed_123\category_aware_tfidf_prompt_v3 `
    .\experiments\model_comparison\results\diverse_seed_2026\category_aware_tfidf_prompt_v3 `
  --output-dir .\experiments\model_comparison\results\aggregate_diverse_3_seeds_tfidf_prompt_v3\category_aware
```

### 7.3. System pipeline comparison

```powershell
.\project\.venv\Scripts\python.exe -m experiments.model_comparison.aggregate_seeds `
  --inputs `
    .\experiments\model_comparison\results\diverse_seed_42\system_pipeline_tfidf_prompt_v3 `
    .\experiments\model_comparison\results\diverse_seed_123\system_pipeline_tfidf_prompt_v3 `
    .\experiments\model_comparison\results\diverse_seed_2026\system_pipeline_tfidf_prompt_v3 `
  --output-dir .\experiments\model_comparison\results\aggregate_diverse_3_seeds_tfidf_prompt_v3\system_pipeline
```

`aggregate_seeds.py` chỉ đọc kết quả có sẵn; nó không prepare dataset, fine-tune,
sinh embeddings hoặc chạy lại comparison.
