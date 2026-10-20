# Model comparison experiment

This folder contains the controlled evaluation used to compare three input
settings on the same held-out product set:

- **Text-only:** multilingual SBERT, pretrained CLIP text encoder, and
  fine-tuned CLIP text encoder.
- **Image-only:** ResNet18 and fine-tuned CLIP vision encoder.
- **Image + text:** late fusion of the SBERT and ResNet18 rankings using
  Reciprocal Rank Fusion (RRF), plus weighted raw-cosine fusion using the
  fine-tuned CLIP image/text encoders.

The experiment loads the `test` rows saved by
`project/runs/clip_finetune_3epochs`. It does not create a new split. The query
product is removed from the gallery, and products with the same ground-truth
category are treated as relevant. Product category is deliberately omitted
from every text encoder's input to avoid leaking the evaluation label.

The project ResNet checkpoint is a **ResNet18** classifier trained on six
categories. The experiment uses its penultimate 512-dimensional feature vector;
it does not claim that the checkpoint is a 142-category classifier. Use
`--resnet-source imagenet` to evaluate standard ImageNet ResNet18 instead.

The CLIP fusion row uses the same fine-tuned checkpoint and held-out queries
and combines cosine scores directly. It intentionally does not apply category
filtering or attribute overrides; those belong to a separate system-level
ablation.

## Setup

Run these commands from the repository root (`dacntt/`):

```powershell
.\project\.venv\Scripts\python.exe -m pip install -r .\experiments\model_comparison\requirements.txt
```

Run the small metric test without loading any model:

```powershell
.\project\.venv\Scripts\python.exe -m experiments.model_comparison.compare_modalities --self-test
```

Validate the dataset and configuration:

```powershell
.\project\.venv\Scripts\python.exe -m experiments.model_comparison.compare_modalities `
  --run-dir .\project\runs\clip_finetune_3epochs `
  --dry-run
```

## Run the modality comparison

```powershell
.\project\.venv\Scripts\python.exe -m experiments.model_comparison.compare_modalities `
  --run-dir .\project\runs\clip_finetune_3epochs `
  --resnet-source project-trained `
  --device cpu `
  --batch-size 16 `
  --ks 1 5 10 `
  --clip-image-weight 0.5 `
  --clip-text-image-weight 0.5 `
  --output-dir .\experiments\model_comparison\results\modalities_5000
```

The first run downloads SBERT and pretrained CLIP if they are not already in
the Hugging Face cache. Embeddings are cached under `cache/`; later runs reuse
them only when the model identity and evaluation rows match. The fine-tuned
CLIP category-free result is an inference ablation because the checkpoint was
originally trained with category-bearing prompts.

## Run the unified 5,000-product comparisons

`compare_full_gallery` is the canonical comparison entry point. The older
`run_all` module remains only as a compatibility alias and accepts the same
arguments; it no longer depends on code inside `project/`.

Use this protocol for the report so every comparison uses the same 473
held-out test queries, the same complete 5,000-product catalog, the same K
values and the same saved model checkpoints. One command now produces three
separate result groups:

- `controlled_baselines/`: category-retrieval baselines for SBERT, ResNet18,
  pretrained CLIP, fine-tuned CLIP and SBERT + ResNet18 RRF. The query product
  is excluded and relevance means the same ground-truth category.
- `clip_finetuning/`: pretrained versus fine-tuned CLIP on text-to-image and
  image-to-text paired retrieval. The paired product remains in the catalog
  because exact `product_id` is the retrieval target.
- `serving_ablation/`: the image-only, text-to-image, text-to-text, website
  text fusion, and raw multimodal fusion configurations. Category relevance is
  applied only after unrestricted ranking.

Run:

```powershell
.\project\.venv\Scripts\python.exe -m experiments.model_comparison.compare_full_gallery `
  --run-dir .\project\runs\clip_finetune_3epochs `
  --catalog-csv .\project\data\processed\products.csv `
  --expected-gallery-size 5000 `
  --resnet-source project-trained `
  --device cpu `
  --batch-size 16 `
  --ks 1 5 10 `
  --clip-image-weight 0.5 `
  --clip-text-image-weight 0.5 `
  --output-dir .\experiments\model_comparison\results\full_gallery_5000_raw_v1
```

Run configuration validation first without loading any model:

```powershell
.\project\.venv\Scripts\python.exe -m experiments.model_comparison.compare_full_gallery `
  --run-dir .\project\runs\clip_finetune_3epochs `
  --catalog-csv .\project\data\processed\products.csv `
  --expected-gallery-size 5000 `
  --dry-run
```

The long command caches every newly generated embedding under
`cache/full_gallery_5000/` and reuses the verified fine-tuned CLIP image
embeddings in `project/embeddings/`. If the run is interrupted, rerunning the
same command with a new empty output directory reuses completed cache stages.
The output contains a consolidated `comparison.csv`, provenance files and the
three folders listed above. Every CSV records `comparison_group`, `role`,
`deployed`, `formula`, `task`, query counts and gallery size so controlled
baselines cannot be confused with website-serving configurations.

The website-aligned text row uses
`w_t * S_TI + (1 - w_t) * S_TT`. The multimodal row uses
`w_i * S_II + (1 - w_i) * (w_t * S_TI + (1 - w_t) * S_TT)`. With both command
weights set to `0.5`, the latter becomes
`0.5 * S_II + 0.25 * S_TI + 0.25 * S_TT`.

Do not merge the full-gallery numbers with the 473 x 473 numbers without
stating the protocol. The larger gallery is expected to change Precision,
Recall and F1 even though the 473 test queries are unchanged.

## Run the category-aware ranking comparison

This system-level ablation reuses the verified fine-tuned image embeddings. It
does not train CLIP or regenerate the catalog. It compares unrestricted image
ranking, the previous binary threshold, and the deployed confidence-weighted
category bonus on the same 473 queries and 5,000-product gallery.

```powershell
.\project\.venv\Scripts\python.exe -m experiments.model_comparison.compare_category_aware `
  --run-dir .\project\runs\clip_finetune_3epochs `
  --catalog-csv .\project\data\processed\products.csv `
  --image-embeddings .\project\embeddings\image_embeddings.npy `
  --expected-gallery-size 5000 `
  --ks 1 5 10 `
  --category-top-per-class 3 `
  --category-confidence-floor 0.50 `
  --category-confidence-threshold 0.60 `
  --category-bonus 0.05 `
  --output-dir .\experiments\model_comparison\results\category_aware_5000_v2
```

The output includes ranking metrics, per-category metrics, query-level category
predictions, category prediction Accuracy and Macro-F1, binary-threshold
coverage, confidence-weighted coverage, and mean category strength.

## Run the deployed query-mode and combined-search comparison

This experiment follows the website pipeline rather than evaluating only one
encoder. It compares image-only, text-only and combined image-text queries, and
then performs the combined-search ablation below:

- raw three-score fusion;
- raw fusion plus text attributes;
- raw fusion plus the category policy;
- the full deployed ranking with both attributes and category handling.

The experiment reuses `project/embeddings/` and the verified category-free
query-text cache. It does not fine-tune CLIP or regenerate the catalog.

Validate the configuration first:

```powershell
.\project\.venv\Scripts\python.exe -m experiments.model_comparison.compare_system_pipeline `
  --run-dir .\project\runs\clip_finetune_3epochs `
  --catalog-csv .\project\data\processed\products.csv `
  --expected-gallery-size 5000 `
  --dry-run
```

Run the comparison:

```powershell
.\project\.venv\Scripts\python.exe -m experiments.model_comparison.compare_system_pipeline `
  --run-dir .\project\runs\clip_finetune_3epochs `
  --catalog-csv .\project\data\processed\products.csv `
  --image-embeddings .\project\embeddings\image_embeddings.npy `
  --text-embeddings .\project\embeddings\text_embeddings.npy `
  --expected-gallery-size 5000 `
  --device cpu `
  --batch-size 16 `
  --ks 1 5 10 `
  --image-weight 0.50 `
  --text-image-weight 0.50 `
  --attribute-weight 0.25 `
  --image-category-bonus 0.05 `
  --combined-category-bonus 0.10 `
  --category-top-per-class 3 `
  --category-confidence-floor 0.50 `
  --category-confidence-threshold 0.60 `
  --output-dir .\experiments\model_comparison\results\system_pipeline_5000_v1
```

The main outputs are `comparison.csv`, `attribute_comparison.csv`,
`query_diagnostics.csv`, `summary.md`, and the separate `query_modes/` and
`combined_ablation/` tables. Category-based Precision, Recall and F1 evaluate
general retrieval relevance. The attribute table separately measures whether
retrieved products satisfy explicit color or pattern terms in the query.
