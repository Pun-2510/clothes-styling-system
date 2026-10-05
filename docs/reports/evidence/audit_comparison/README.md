# Audit, comparison, and runtime evidence

This directory contains small, versioned snapshots used by the weekly report.
The original generated datasets, experiment directories, and runtime logs remain
ignored because they can be regenerated or may grow substantially.

| Evidence file | Original source | Purpose |
|---|---|---|
| `category_audit.csv` | `project/data/processed/products.category_audit.csv` | Per-category coverage and audit status for the 5,000-product catalog |
| `matched_comparison.csv` | `experiments/model_comparison/results/full_5000_v2/matched_comparison.csv` | Matched text-only, image-only, and combined retrieval metrics |
| `clip_comparison.csv` | `experiments/model_comparison/results/full_5000_v2/clip/comparison.csv` | Pretrained versus fine-tuned CLIP metrics |
| `recommendation-backend.log` | `project/logs/recommendation-backend.log` | Runtime trace for the first backend instance |
| `recommendation-backend_2.log` | `project/logs/recommendation-backend_2.log` | Runtime trace for the second backend instance |

The CSV and log files are evidence snapshots, not runtime inputs. Regenerating
the pipeline does not update these copies automatically. When results change,
replace the snapshots deliberately and review the Git diff before committing.

The two log files were checked for common credential patterns before inclusion.
They contain request identifiers, predicted categories, confidence values,
fallback decisions, similarity scores, and processing times. They do not contain
uploaded image bytes, model weights, or embeddings.
