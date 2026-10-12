# Modality comparison

The benchmark used 473 held-out products and 132 categories.
The query product was excluded, and category labels were not included in any text-encoder input.

| System | Input | K | Precision | Recall | F1 | Evaluated | Skipped |
|---|---|---:|---:|---:|---:|---:|---:|
| sbert_text_only | text | 1 | 0.6692 | 0.0997 | 0.1430 | 393 | 80 |
| sbert_text_only | text | 5 | 0.5211 | 0.2683 | 0.2735 | 393 | 80 |
| sbert_text_only | text | 10 | 0.4565 | 0.4090 | 0.3370 | 393 | 80 |
| clip_pretrained_text_only | text | 1 | 0.7048 | 0.0989 | 0.1474 | 393 | 80 |
| clip_pretrained_text_only | text | 5 | 0.5720 | 0.3076 | 0.3116 | 393 | 80 |
| clip_pretrained_text_only | text | 10 | 0.4962 | 0.4590 | 0.3725 | 393 | 80 |
| clip_finetuned_text_only | text | 1 | 0.6896 | 0.0978 | 0.1440 | 393 | 80 |
| clip_finetuned_text_only | text | 5 | 0.5379 | 0.2793 | 0.2850 | 393 | 80 |
| clip_finetuned_text_only | text | 10 | 0.4547 | 0.4066 | 0.3318 | 393 | 80 |
| resnet18_image_only | image | 1 | 0.6183 | 0.0774 | 0.1179 | 393 | 80 |
| resnet18_image_only | image | 5 | 0.5211 | 0.2406 | 0.2626 | 393 | 80 |
| resnet18_image_only | image | 10 | 0.4522 | 0.3666 | 0.3179 | 393 | 80 |
| sbert_resnet18_rrf | image + text | 1 | 0.7176 | 0.0986 | 0.1470 | 393 | 80 |
| sbert_resnet18_rrf | image + text | 5 | 0.6163 | 0.3119 | 0.3219 | 393 | 80 |
| sbert_resnet18_rrf | image + text | 10 | 0.5366 | 0.4725 | 0.3903 | 393 | 80 |
| clip_finetuned_image_only | image | 1 | 0.6947 | 0.0995 | 0.1486 | 393 | 80 |
| clip_finetuned_image_only | image | 5 | 0.6178 | 0.3370 | 0.3440 | 393 | 80 |
| clip_finetuned_image_only | image | 10 | 0.5209 | 0.4871 | 0.3969 | 393 | 80 |
| clip_finetuned_raw_cosine_fusion | image + text | 1 | 0.7328 | 0.1106 | 0.1620 | 393 | 80 |
| clip_finetuned_raw_cosine_fusion | image + text | 5 | 0.6260 | 0.3503 | 0.3500 | 393 | 80 |
| clip_finetuned_raw_cosine_fusion | image + text | 10 | 0.5394 | 0.5080 | 0.4102 | 393 | 80 |
| clip_finetuned_percentile_fusion | image + text | 1 | 0.7252 | 0.1071 | 0.1582 | 393 | 80 |
| clip_finetuned_percentile_fusion | image + text | 5 | 0.6137 | 0.3455 | 0.3432 | 393 | 80 |
| clip_finetuned_percentile_fusion | image + text | 10 | 0.5361 | 0.5052 | 0.4084 | 393 | 80 |

## Interpretation boundary

This table measures same-category retrieval, which is a proxy for relevance rather than human judgement of visual style.
The combined system uses rank fusion because SBERT and ResNet18 embeddings are not in a shared vector space.
RRF uses k = 60.
CLIP raw fusion combines cosine similarities directly; CLIP percentile fusion first aligns each query's score scales.
CLIP fusion uses image weight 0.50 and text-to-image weight 0.50.
These CLIP fusion rows do not apply category filtering or attribute overrides; those require a separate system-level ablation.
The project-trained ResNet18 checkpoint was trained on six categories, so its result must not be presented as a 142-category classifier score.
The fine-tuned CLIP checkpoint was trained with category-bearing prompts; its category-free text result is therefore an inference ablation, not a category-free retraining run.
