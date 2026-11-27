# Model comparison experiments

> **Phạm vi riêng:** thư mục này chỉ dùng cho đánh giá nghiên cứu và kết quả
> nhiều seed. Website trong `project/` không đọc catalog, embeddings, checkpoint
> hoặc weight JSON từ `experiments/`. Pipeline vận hành và lệnh validation dùng
> cho website nằm trong `project/readme.txt` và chạy bằng
> `python -m src.tune_retrieval_weights` từ thư mục `project/`.

Các lệnh bên dưới được chạy từ thư mục gốc repository vì chúng cần đồng thời
truy cập code thí nghiệm và artifact của seed được đánh giá. Tên thư mục clone
không cần là `dacntt`.

Thư mục này chứa protocol đánh giá trên ba catalog diverse, mỗi catalog
5.000 sản phẩm với seed `42`, `123` hoặc `2026`. Các lệnh Dynamic ở phần
trước được giữ để tái lập thí nghiệm lịch sử; pipeline website hiện dùng Hard
Category. Xem các mục Hard Category và frozen CLIP phía cuối tài liệu để chạy
protocol hiện tại. Bảng tổng hợp đã kiểm chứng nằm tại
`docs/reports/evidence/final_experiments/model_comparison_k10.csv`.

Tất cả lệnh trong tài liệu này chạy từ thư mục gốc repository (bất kể tên
thư mục được chọn khi clone).

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
  là lựa chọn linh hoạt cho image-only. Hard Category là mặc định trên website.

### System pipeline comparison

`compare_system_pipeline.py` kiểm tra ba query mode và combined ablation:

- Image-only deployed.
- Text-only deployed.
- Combined deployed.
- Raw CLIP fusion, category-gated TF-IDF metadata refinement, category
  contribution và full ranking. Trong combined query, TF-IDF chỉ tăng điểm
  trong category được chọn; image-only và combined dùng hard category pool.

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

Core comparison giữ hậu tố `map_v1` và Category-aware giữ hậu tố
`tfidf_prompt_v3`. System Pipeline dùng hậu tố `category_gated_v4` vì combined
TF-IDF hiện chỉ tăng điểm cho category được ảnh hỗ trợ hoặc một category được
text nêu rõ. Trước mỗi System Pipeline run phải chọn lại weights bằng
`src.tune_retrieval_weights`; không tái sử dụng weights `tfidf_prompt_v3`.
Mỗi output phải dùng thư mục mới vì chương trình không ghi đè kết quả cũ.

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
Push-Location .\project
.\.venv\Scripts\python.exe -m src.tune_retrieval_weights `
  --run-dir runs/clip_finetune_diverse_seed_42 `
  --catalog-csv data/processed_diverse/products.csv `
  --image-embeddings embeddings_diverse_seed_42/image_embeddings.npy `
  --text-embeddings embeddings_diverse_seed_42/text_embeddings.npy `
  --metadata-index-dir metadata_index_diverse_seed_42 `
  --expected-gallery-size 5000 `
  --output-dir ../experiments/model_comparison/results/diverse_seed_42/weight_tuning_category_gated_v4 `
  --weights-output ../experiments/model_comparison/results/diverse_seed_42/weight_tuning_category_gated_v4/retrieval_weights.json `
  --device cpu `
  --batch-size 16 `
  --objective-k 10
Pop-Location

$weights42 = (Get-Content .\experiments\model_comparison\results\diverse_seed_42\weight_tuning_category_gated_v4\retrieval_weights.json -Raw | ConvertFrom-Json).weights

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
  --output-dir .\experiments\model_comparison\results\diverse_seed_42\system_pipeline_category_gated_v4
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
Push-Location .\project
.\.venv\Scripts\python.exe -m src.tune_retrieval_weights `
  --run-dir runs/clip_finetune_diverse_seed_123 `
  --catalog-csv data/processed_diverse_seed_123/products.csv `
  --image-embeddings embeddings_diverse_seed_123/image_embeddings.npy `
  --text-embeddings embeddings_diverse_seed_123/text_embeddings.npy `
  --metadata-index-dir metadata_index_diverse_seed_123 `
  --expected-gallery-size 5000 `
  --output-dir ../experiments/model_comparison/results/diverse_seed_123/weight_tuning_category_gated_v4 `
  --weights-output ../experiments/model_comparison/results/diverse_seed_123/weight_tuning_category_gated_v4/retrieval_weights.json `
  --device cpu `
  --batch-size 16 `
  --objective-k 10
Pop-Location

$weights123 = (Get-Content .\experiments\model_comparison\results\diverse_seed_123\weight_tuning_category_gated_v4\retrieval_weights.json -Raw | ConvertFrom-Json).weights

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
  --output-dir .\experiments\model_comparison\results\diverse_seed_123\system_pipeline_category_gated_v4
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
Push-Location .\project
.\.venv\Scripts\python.exe -m src.tune_retrieval_weights `
  --run-dir runs/clip_finetune_diverse_seed_2026 `
  --catalog-csv data/processed_diverse_seed_2026/products.csv `
  --image-embeddings embeddings_diverse_seed_2026/image_embeddings.npy `
  --text-embeddings embeddings_diverse_seed_2026/text_embeddings.npy `
  --metadata-index-dir metadata_index_diverse_seed_2026 `
  --expected-gallery-size 5000 `
  --output-dir ../experiments/model_comparison/results/diverse_seed_2026/weight_tuning_category_gated_v4 `
  --weights-output ../experiments/model_comparison/results/diverse_seed_2026/weight_tuning_category_gated_v4/retrieval_weights.json `
  --device cpu `
  --batch-size 16 `
  --objective-k 10
Pop-Location

$weights2026 = (Get-Content .\experiments\model_comparison\results\diverse_seed_2026\weight_tuning_category_gated_v4\retrieval_weights.json -Raw | ConvertFrom-Json).weights

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
  --output-dir .\experiments\model_comparison\results\diverse_seed_2026\system_pipeline_category_gated_v4
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
    .\experiments\model_comparison\results\diverse_seed_42\system_pipeline_category_gated_v4 `
    .\experiments\model_comparison\results\diverse_seed_123\system_pipeline_category_gated_v4 `
    .\experiments\model_comparison\results\diverse_seed_2026\system_pipeline_category_gated_v4 `
  --output-dir .\experiments\model_comparison\results\aggregate_diverse_3_seeds_category_gated_v4\system_pipeline
```

`aggregate_seeds.py` chỉ đọc kết quả có sẵn; nó không prepare dataset, fine-tune,
sinh embeddings hoặc chạy lại comparison.

## 8. Đo khi nào Dynamic Relation hữu ích hơn Hard Category

Website hiện chọn Hard Category mặc định cho image-only; Dynamic Relation là lựa chọn
linh hoạt. Hai cách dùng cùng category prediction và cùng 5.000 sản phẩm trong
catalog của từng seed. `compare_category_aware.py` ghi thêm:

- `hard_dynamic_recovery.csv`: Recovery@K là tỷ lệ truy vấn bị đoán sai category
  nhưng Dynamic vẫn tìm được ít nhất một sản phẩm đúng category trong Top-K.
- `hard_dynamic_queries.csv`: mỗi truy vấn được Hard/Dynamic tìm thấy hay bỏ lỡ,
  dùng để chọn ví dụ minh họa có thật.
- Loss@K trong file tổng hợp: tỷ lệ truy vấn đoán đúng category mà Hard tìm thấy
  sản phẩm đúng trong Top-K nhưng Dynamic bỏ lỡ. Đọc cùng Recovery@K và
  `net_hit_rate_change` để không chỉ báo cáo những trường hợp có lợi cho Dynamic.

Relevance ở đây là **cùng category với ảnh truy vấn**, không chứng minh đúng màu,
kiểu dáng hay sở thích người dùng. Query product được loại khỏi gallery.
Không cần prepare data, fine-tune hay sinh lại embeddings; cần chạy lại riêng
Category-aware comparison vì kết quả cũ không lưu ranking theo từng query.
Chạy các lệnh từ thư mục gốc của repo, mỗi seed dùng output mới:

```powershell
.\project\.venv\Scripts\python.exe -m experiments.model_comparison.compare_category_aware `
  --run-dir .\project\runs\clip_finetune_diverse_seed_42 `
  --catalog-csv .\project\data\processed_diverse\products.csv `
  --image-embeddings .\project\embeddings_diverse_seed_42\image_embeddings.npy `
  --category-cache-dir .\experiments\model_comparison\cache\diverse_seed_42 `
  --expected-gallery-size 5000 --ks 1 5 10 --category-bonus 0.20 `
  --output-dir .\experiments\model_comparison\results\diverse_seed_42\category_aware_recovery_v4
```

```powershell
.\project\.venv\Scripts\python.exe -m experiments.model_comparison.compare_category_aware `
  --run-dir .\project\runs\clip_finetune_diverse_seed_123 `
  --catalog-csv .\project\data\processed_diverse_seed_123\products.csv `
  --image-embeddings .\project\embeddings_diverse_seed_123\image_embeddings.npy `
  --category-cache-dir .\experiments\model_comparison\cache\diverse_seed_123 `
  --expected-gallery-size 5000 --ks 1 5 10 --category-bonus 0.20 `
  --output-dir .\experiments\model_comparison\results\diverse_seed_123\category_aware_recovery_v4
```

```powershell
.\project\.venv\Scripts\python.exe -m experiments.model_comparison.compare_category_aware `
  --run-dir .\project\runs\clip_finetune_diverse_seed_2026 `
  --catalog-csv .\project\data\processed_diverse_seed_2026\products.csv `
  --image-embeddings .\project\embeddings_diverse_seed_2026\image_embeddings.npy `
  --category-cache-dir .\experiments\model_comparison\cache\diverse_seed_2026 `
  --expected-gallery-size 5000 --ks 1 5 10 --category-bonus 0.20 `
  --output-dir .\experiments\model_comparison\results\diverse_seed_2026\category_aware_recovery_v4
```

Sau khi cả ba seed xong, lấy mean và standard deviation:

```powershell
.\project\.venv\Scripts\python.exe -m experiments.model_comparison.aggregate_category_recovery `
  --inputs `
    .\experiments\model_comparison\results\diverse_seed_42\category_aware_recovery_v4 `
    .\experiments\model_comparison\results\diverse_seed_123\category_aware_recovery_v4 `
    .\experiments\model_comparison\results\diverse_seed_2026\category_aware_recovery_v4 `
  --output-dir .\experiments\model_comparison\results\aggregate_diverse_3_seeds_recovery_v4
```


## Hard Category System Pipeline and FashionCLIP baseline

These are new experiment versions. Older result folders remain historical.
Use the same three diverse 5,000-product catalogs and test splits. FashionCLIP
is frozen and uses raw cosine scores without website category or TF-IDF rules.
The System Pipeline uses fine-tuned CLIP, TF-IDF and the website's Hard Category
candidate pool. These answer different questions and should have separate tables.

Run from the repository root in PowerShell. Each command handles one seed;
the `all` stage runs validation tuning, Hard Category System Pipeline, then
FashionCLIP in that order. Use `-Stage tune`, `-Stage system`, or
`-Stage fashionclip` to run only one stage. Outputs use new `hard_v1` and
`fashionclip_raw_v1` directories and never overwrite the website's
`project/retrieval_weights.json`.

```powershell
.\experiments\model_comparison\run_seed_hard_pipeline.ps1 -Seed 42 -Stage all
.\experiments\model_comparison\run_seed_hard_pipeline.ps1 -Seed 123 -Stage all
.\experiments\model_comparison\run_seed_hard_pipeline.ps1 -Seed 2026 -Stage all
```

Once all three seeds finish, aggregate the System Pipeline and frozen
FashionCLIP baseline separately:

```powershell
.\project\.venv\Scripts\python.exe -m experiments.model_comparison.aggregate_seeds `
  --inputs `
    .\experiments\model_comparison\results\diverse_seed_42\system_pipeline_hard_v1 `
    .\experiments\model_comparison\results\diverse_seed_123\system_pipeline_hard_v1 `
    .\experiments\model_comparison\results\diverse_seed_2026\system_pipeline_hard_v1 `
  --output-dir .\experiments\model_comparison\results\aggregate_system_pipeline_hard_v1

.\project\.venv\Scripts\python.exe -m experiments.model_comparison.aggregate_seeds `
  --inputs `
    .\experiments\model_comparison\results\diverse_seed_42\fashionclip_raw_v1 `
    .\experiments\model_comparison\results\diverse_seed_123\fashionclip_raw_v1 `
    .\experiments\model_comparison\results\diverse_seed_2026\fashionclip_raw_v1 `
  --output-dir .\experiments\model_comparison\results\aggregate_fashionclip_raw_v1
```

The benchmark must not be presented as a controlled model latency comparison:
FashionCLIP is a frozen raw-score baseline, while the System Pipeline includes
Hard Category and TF-IDF. Report their differing retrieval rules alongside
the scores.

For each seed:

1. Run `src.tune_retrieval_weights` on validation and write
   `--weights-output` into that seed's experiment folder. Never overwrite
   `project/retrieval_weights.json` when tuning experiment seeds.
2. Pass the selected fusion and metadata weights to
   `compare_system_pipeline` using the matching run, catalog, embeddings
   and TF-IDF index.
3. Run `compare_fashionclip` with that run and catalog. It records the
   resolved Hugging Face revision SHA. Aggregate each comparison separately
   with `aggregate_seeds`.

Example FashionCLIP command from the repository root:

```powershell
.\project\.venv\Scripts\python.exe -m experiments.model_comparison.compare_fashionclip `
  --run-dir .\project\runs\clip_finetune_diverse_seed_42 `
  --catalog-csv .\project\data\processed_diverse\products.csv `
  --expected-gallery-size 5000 --device cpu --batch-size 16 --ks 1 5 10 `
  --cache-dir .\experiments\model_comparison\cache\fashionclip_seed_42 `
  --output-dir .\experiments\model_comparison\results\diverse_seed_42\fashionclip_raw_v1
```

Use a fresh output directory for each run. The script rejects nonempty
output directories to prevent mixed results.

## Deployed API performance: Text, Image, Combined

Run this **separately from retrieval-quality experiments**. It measures warm,
sequential client response time through the public reverse proxy to the deployed
website. The script samples distinct product categories from the currently
deployed catalog. For each case, Text uses the product name, Image uses its
image, and Combined uses both. All three routes use the same Top-K value and
the same cases. Requests are interleaved across modes and cases to reduce
effects from server drift. No audience filter or Vietnamese translation is used.

Start Docker Compose from `project/` first, then run from the repository root:

```powershell
.\project\.venv\Scripts\python.exe -m experiments.model_comparison.benchmark_api_latency `
  --base-url http://localhost `
  --catalog-csv .\project\data\processed\products.csv `
  --image-dir .\project\data\data `
  --sample-size 10 --seed 42 --top-k 10 `
  --warmup 2 --repetitions 8 --pause-seconds 0.30 `
  --output-dir .\experiments\model_comparison\results\api_latency_10k_v1
```

`--sample-size` is the number of distinct-category query cases, not gallery
size. The command sends 10 × 8 = 80 measured requests **per query mode** after
warmup. Use a new output directory for each run. If the deployed catalog is
5,000 rather than 10,000 products, change the output name accordingly. Keep
the same deployment, hardware, K, and case list when comparing modes.

`performance_summary.csv` reports mean, median, and p95 client latency (ms),
backend-reported median processing time where available, and average request/
response KiB. `request_latency.csv` contains every measurement;
`benchmark_cases.csv` and `benchmark_protocol.json` document the cases and
settings. Median describes a typical request; p95 describes a slow request.
This measures deployed API response, including HTTP and routing overhead. It
does not measure model-only inference, concurrent throughput, or CPU/GPU memory.
For resource information, record `docker stats --no-stream` for the two backend
containers during a separate controlled run, and label those snapshots as
observational rather than per-query cost.

Do not merge these latency numbers into the three-seed 5,000-product quality
table when the deployed website uses a different catalog size. Present the
quality and performance tables side by side, with their catalog sizes stated.

## Additional frozen OpenCLIP and SigLIP baselines

These two baselines use the **same three 5,000-product catalogs and held-out
test splits** as the FashionCLIP comparison. OpenCLIP uses ViT-B/32 with the
`laion2b_s34b_b79k` pretrained weights; SigLIP uses
`google/siglip-base-patch16-224`. Both models are frozen. They encode each
catalog image and its product name/description once, then evaluate image-only,
text-only, combined raw-cosine, and exact image↔text paired retrieval at
K=1, 5, and 10. The combined baseline uses fixed 0.5 image/text and 0.5
text-to-image/text-to-text weights. These results measure the frozen models;
the deployed system additionally uses TF-IDF and Hard Category ranking, so
keep those rows labelled separately in any combined table.

Install the additional experiment dependency from the repository root:

```powershell
.\project\.venv\Scripts\python.exe -m pip install -r .\experiments\model_comparison\requirements.txt
```

Run **one model and one seed per command**. These commands only create new
experiment caches and result directories; they do not retrain the website
model or change `project/retrieval_weights.json`.

```powershell
.\experiments\model_comparison\run_seed_hard_pipeline.ps1 -Seed 42 -Stage openclip
.\experiments\model_comparison\run_seed_hard_pipeline.ps1 -Seed 123 -Stage openclip
.\experiments\model_comparison\run_seed_hard_pipeline.ps1 -Seed 2026 -Stage openclip

.\experiments\model_comparison\run_seed_hard_pipeline.ps1 -Seed 42 -Stage siglip
.\experiments\model_comparison\run_seed_hard_pipeline.ps1 -Seed 123 -Stage siglip
.\experiments\model_comparison\run_seed_hard_pipeline.ps1 -Seed 2026 -Stage siglip
```

Aggregate only after all three runs of a model finish:

```powershell
.\project\.venv\Scripts\python.exe -m experiments.model_comparison.aggregate_seeds `
  --inputs `
    .\experiments\model_comparison\results\diverse_seed_42\openclip_raw_v1 `
    .\experiments\model_comparison\results\diverse_seed_123\openclip_raw_v1 `
    .\experiments\model_comparison\results\diverse_seed_2026\openclip_raw_v1 `
  --output-dir .\experiments\model_comparison\results\aggregate_openclip_raw_v1

.\project\.venv\Scripts\python.exe -m experiments.model_comparison.aggregate_seeds `
  --inputs `
    .\experiments\model_comparison\results\diverse_seed_42\siglip_raw_v1 `
    .\experiments\model_comparison\results\diverse_seed_123\siglip_raw_v1 `
    .\experiments\model_comparison\results\diverse_seed_2026\siglip_raw_v1 `
  --output-dir .\experiments\model_comparison\results\aggregate_siglip_raw_v1
```

Each aggregate writes `mean_std_summary.csv` and provenance files. The
`run_seed_hard_pipeline.ps1 -Stage all` behavior remains tune + system +
FashionCLIP; OpenCLIP and SigLIP are separate because their first run needs
to download and encode a full catalog. Choose `-Device cuda` if CUDA works in
the experiment environment, and lower `-BatchSize` if GPU memory is limited.
The script refuses a nonempty output directory. For a genuine new run, use a
new output directory in a direct `compare_additional_clips` command or move
the old results aside first.
