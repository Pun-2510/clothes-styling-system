# Model comparison experiment

This folder contains the controlled evaluation used to compare three input
settings on the same held-out product set:

- **Text-only:** multilingual SBERT, pretrained CLIP text encoder, and
  fine-tuned CLIP text encoder.
- **Image-only:** ResNet18.
- **Image + text:** late fusion of the SBERT and ResNet18 rankings using
  Reciprocal Rank Fusion (RRF).

The experiment loads the `test` rows saved by
`project/runs/clip_finetune_3epochs`. It does not create a new split. The query
product is removed from the gallery, and products with the same ground-truth
category are treated as relevant. Product category is deliberately omitted
from every text encoder's input to avoid leaking the evaluation label.

The project ResNet checkpoint is a **ResNet18** classifier trained on six
categories. The experiment uses its penultimate 512-dimensional feature vector;
it does not claim that the checkpoint is a 142-category classifier. Use
`--resnet-source imagenet` to evaluate standard ImageNet ResNet18 instead.

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

- `modalities/`: SBERT, pretrained/fine-tuned CLIP text encoders, ResNet18, and
  SBERT + ResNet18 RRF on same-category retrieval.
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
