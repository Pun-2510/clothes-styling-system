# Model comparison experiment

This folder contains the controlled evaluation used to compare three input
settings on the same held-out product set:

- **Text-only:** multilingual SBERT, pretrained CLIP text encoder, and
  fine-tuned CLIP text encoder.
- **Image-only:** ResNet18 and fine-tuned CLIP vision encoder.
- **Image + text:** late fusion of the SBERT and ResNet18 rankings using
  Reciprocal Rank Fusion (RRF), plus raw-cosine and percentile score fusion
  using the fine-tuned CLIP image/text encoders.

The experiment loads the `test` rows saved by
`project/runs/clip_finetune_3epochs`. It does not create a new split. The query
product is removed from the gallery, and products with the same ground-truth
category are treated as relevant. Product category is deliberately omitted
from every text encoder's input to avoid leaking the evaluation label.

The project ResNet checkpoint is a **ResNet18** classifier trained on six
categories. The experiment uses its penultimate 512-dimensional feature vector;
it does not claim that the checkpoint is a 142-category classifier. Use
`--resnet-source imagenet` to evaluate standard ImageNet ResNet18 instead.

The two CLIP fusion rows use the same fine-tuned checkpoint and the same
held-out queries. Raw fusion combines cosine scores directly. Percentile
fusion first converts each query's image and text scores to relative ranks so
that the two modalities have comparable scales. These rows intentionally do
not apply category filtering or attribute overrides; those belong to a
separate system-level ablation.

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

## Run all comparisons

This command first creates the modality table and then invokes the existing
pretrained-versus-fine-tuned CLIP comparison from `project/comparisons/`:

```powershell
.\project\.venv\Scripts\python.exe -m experiments.model_comparison.run_all `
  --run-dir .\project\runs\clip_finetune_3epochs `
  --resnet-source project-trained `
  --device cpu `
  --batch-size 16 `
  --ks 1 5 10 `
  --output-dir .\experiments\model_comparison\results\full_5000
```

The output contains two separate tables because they answer different
questions:

- `modalities/`: SBERT, pretrained/fine-tuned CLIP text encoders, ResNet18,
  fine-tuned CLIP image-only, SBERT + ResNet18 RRF, and the two fine-tuned CLIP
  fusion baselines on same-category retrieval.
- `clip/`: pretrained CLIP versus fine-tuned CLIP on image-to-text,
  text-to-image, and image-to-image tasks.
- `matched_comparison.csv`: one consolidated table for the compatible
  category-retrieval rows, including CLIP image-only results.

Each modality run writes `comparison.csv`, `comparison.json`,
`comparison_by_category.csv`, `evaluation_catalog.csv`, `experiment.json`, and
`summary.md`. Generated results and caches are intentionally excluded from Git.

## Extend an existing completed run

If `full_5000_v1/clip` already contains the completed CLIP cross-modal result,
reuse it and run only the new matched text comparison:

```powershell
.\project\.venv\Scripts\python.exe -m experiments.model_comparison.run_all `
  --run-dir .\project\runs\clip_finetune_3epochs `
  --reuse-clip-dir .\experiments\model_comparison\results\full_5000_v1\clip `
  --resnet-source project-trained `
  --device cpu `
  --batch-size 16 `
  --ks 1 5 10 `
  --output-dir .\experiments\model_comparison\results\full_5000_v2
```

This produces `matched_comparison.csv` without repeating the already completed
pretrained-versus-fine-tuned CLIP cross-modal evaluation.

## Run the unified 5,000-product comparisons

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
  text fusion, raw multimodal fusion and percentile multimodal fusion
  configurations. Category relevance is applied only after unrestricted
  ranking.

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
  --output-dir .\experiments\model_comparison\results\full_gallery_5000_v2
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
`w_t * S_TI + (1 - w_t) * S_TT`. The multimodal rows use
`w_i * S_II + (1 - w_i) * (w_t * S_TI + (1 - w_t) * S_TT)`. With both command
weights set to `0.5`, the latter becomes
`0.5 * S_II + 0.25 * S_TI + 0.25 * S_TT`. The percentile variant converts
each component score family before applying these weights.

Do not merge the full-gallery numbers with the 473 x 473 numbers without
stating the protocol. The larger gallery is expected to change Precision,
Recall and F1 even though the 473 test queries are unchanged.
