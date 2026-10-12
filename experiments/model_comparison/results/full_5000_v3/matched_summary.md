# Matched category-retrieval comparison

All rows use the same held-out products, category relevance, self-match exclusion, and K values.
Text-only rows use product name and description without the category label.

| System | Input | K | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| sbert_text_only | text | 1 | 0.6692 | 0.0997 | 0.1430 |
| sbert_text_only | text | 5 | 0.5211 | 0.2683 | 0.2735 |
| sbert_text_only | text | 10 | 0.4565 | 0.4090 | 0.3370 |
| clip_pretrained_text_only | text | 1 | 0.7048 | 0.0989 | 0.1474 |
| clip_pretrained_text_only | text | 5 | 0.5720 | 0.3076 | 0.3116 |
| clip_pretrained_text_only | text | 10 | 0.4962 | 0.4590 | 0.3725 |
| clip_finetuned_text_only | text | 1 | 0.6896 | 0.0978 | 0.1440 |
| clip_finetuned_text_only | text | 5 | 0.5379 | 0.2793 | 0.2850 |
| clip_finetuned_text_only | text | 10 | 0.4547 | 0.4066 | 0.3318 |
| resnet18_image_only | image | 1 | 0.6183 | 0.0774 | 0.1179 |
| resnet18_image_only | image | 5 | 0.5211 | 0.2406 | 0.2626 |
| resnet18_image_only | image | 10 | 0.4522 | 0.3666 | 0.3179 |
| sbert_resnet18_rrf | image + text | 1 | 0.7176 | 0.0986 | 0.1470 |
| sbert_resnet18_rrf | image + text | 5 | 0.6163 | 0.3119 | 0.3219 |
| sbert_resnet18_rrf | image + text | 10 | 0.5366 | 0.4725 | 0.3903 |
| clip_finetuned_image_only | image | 1 | 0.6947 | 0.0995 | 0.1486 |
| clip_finetuned_image_only | image | 5 | 0.6178 | 0.3370 | 0.3440 |
| clip_finetuned_image_only | image | 10 | 0.5209 | 0.4871 | 0.3969 |
| clip_finetuned_raw_cosine_fusion | image + text | 1 | 0.7328 | 0.1106 | 0.1620 |
| clip_finetuned_raw_cosine_fusion | image + text | 5 | 0.6260 | 0.3503 | 0.3500 |
| clip_finetuned_raw_cosine_fusion | image + text | 10 | 0.5394 | 0.5080 | 0.4102 |
| clip_finetuned_percentile_fusion | image + text | 1 | 0.7252 | 0.1071 | 0.1582 |
| clip_finetuned_percentile_fusion | image + text | 5 | 0.6137 | 0.3455 | 0.3432 |
| clip_finetuned_percentile_fusion | image + text | 10 | 0.5361 | 0.5052 | 0.4084 |
| clip_pretrained_image_only | image | 1 | 0.6718 | 0.0957 | 0.1420 |
| clip_pretrained_image_only | image | 5 | 0.5863 | 0.3171 | 0.3243 |
| clip_pretrained_image_only | image | 10 | 0.4875 | 0.4649 | 0.3767 |

The RRF row requires both image and text and should not be treated as a single-input model.
The fine-tuned CLIP text result is an inference ablation because its training prompts included category.
The CLIP fusion rows use the same fine-tuned checkpoint for both modalities and exclude category/attribute post-processing.
