# Evidence thí nghiệm ba seed

Thư mục này lưu bảng tổng hợp và hồ sơ chạy ngắn gọn cho ba catalog diverse,
mỗi catalog có 5.000 sản phẩm. Các seed là `42`, `123`, `2026`, tương ứng
460, 464, 462 test queries. Các phương án trong cùng seed dùng đúng một catalog,
test split và định nghĩa relevance. `mean_std_summary.csv` tính trung bình và
độ lệch chuẩn mẫu của ba seed; mỗi seed có trọng số bằng nhau.

## Bảng kết quả hiện tại

- [`model_comparison_k10.csv`](model_comparison_k10.csv): Precision, Recall,
  F1, NDCG và MAP ở K=10 cho model-only image/text/combined và ba luồng
  full Hard Category pipeline. Cột `protocol` đánh dấu rõ hai loại phương án.
- [`paired_retrieval_k10.csv`](paired_retrieval_k10.csv): truy xuất đúng cặp
  image/text theo product ID. Không gộp bảng này với đánh giá cùng category.
- `core_extended_pretrained/`: ResNet18, EfficientNet-B0, ViT, TF-IDF, BERT,
  SBERT, pretrained/fine-tuned CLIP và các baseline ban đầu.
- `fashionclip_raw/`, `openclip_raw/`, `siglip_raw/`: ba model frozen dùng raw
  cosine, không áp dụng lọc category hay TF-IDF của website.
- `system_pipeline_hard/`: ba luồng website hiện tại với Hard Category và TF-IDF.

Mỗi thư mục aggregate có `aggregation.json`, `mean_std_summary.csv` và
`per_seed_results.csv`. `run_details/seed_*/` giữ `comparison.csv`,
`comparison_by_category.csv` và `experiment.json` của mỗi lượt chạy; riêng
pipeline Hard Category còn có `metadata_comparison.csv`,
`query_diagnostics.csv` và `summary.md`. Các file `experiment.json` ghi catalog
hash, checkpoint/model, số query và cấu hình để kiểm tra tính đối chiếu.

Các bảng model-only dùng cùng-category relevance nhưng chỉ so sánh biểu diễn
và raw cosine. Full pipeline còn dùng hard category pool và TF-IDF, nên chênh
lệch giữa hai nhóm phản ánh cả mô hình lẫn quy tắc phục vụ. Bảng paired
retrieval dùng product ID làm relevance và trả lời câu hỏi khác. Các file này
là evidence báo cáo; website không đọc từ thư mục này.

## Kết quả lịch sử

`category_aware/` và `system_pipeline/` ghi lại giai đoạn Dynamic Category
Evidence, trước khi website chuyển sang Hard Category mặc định. Chúng được
giữ để truy vết quá trình phát triển và không đại diện cho pipeline triển khai
hiện tại. `core_extended_pretrained/` vẫn là baseline model-only được dùng
trong bảng K=10 hiện tại.

Nguồn aggregate lịch sử:

- `aggregate_diverse_3_seeds_map_v1/` cho `core_extended_pretrained/`.
- `aggregate_diverse_3_seeds_tfidf_prompt_v3/category_aware/` cho
  `category_aware/`.
- `aggregate_diverse_3_seeds_category_gated_v4/system_pipeline/` cho
  `system_pipeline/`.

Nguồn hiện tại: `aggregate_system_pipeline_hard_v1/`,
`aggregate_fashionclip_raw_v1/`, `aggregate_openclip_raw_v1/` và
`aggregate_siglip_raw_v1/` dưới `experiments/model_comparison/results/`.
Các cache, embeddings và checkpoint lớn chỉ lưu tại máy chạy. Hồ sơ trong
`run_details/` là bản ghi có cấu trúc, không phải log terminal tự tạo lại.

## API latency (website 10,000 products)

[`api_latency_100/performance_summary.csv`](api_latency_100/performance_summary.csv)
contains client-side HTTP latency for text-only, image-only, and combined
queries through the public reverse proxy. The benchmark uses 10 product
cases, 2 warmup rounds, and 10 measured rounds: 100 requests per mode.
`request_latency.csv` stores each measurement; `benchmark_cases.csv` and
`benchmark_protocol.json` record the cases and run settings. These are
runtime measurements on the 10,000-product website catalog, separate from
the three-seed, 5,000-product retrieval-quality comparisons above. The
80-request `api_latency_10k_v1` run was a preliminary measurement and is
not included in this final evidence set.
