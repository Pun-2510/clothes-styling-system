# Full-gallery modality comparison

The benchmark used 473 held-out test queries against 5000 catalog products.
The query product was excluded. Category labels were used only after ranking to determine relevance.

| System | Input | K | Precision | Recall | F1 | Evaluated | Skipped |
|---|---|---:|---:|---:|---:|---:|---:|
| sbert_text_only | text | 1 | 0.8097 | 0.0257 | 0.0452 | 473 | 0 |
| sbert_text_only | text | 5 | 0.7125 | 0.0928 | 0.1291 | 473 | 0 |
| sbert_text_only | text | 10 | 0.6539 | 0.1430 | 0.1736 | 473 | 0 |
| clip_pretrained_text_only | text | 1 | 0.8161 | 0.0241 | 0.0428 | 473 | 0 |
| clip_pretrained_text_only | text | 5 | 0.7252 | 0.0834 | 0.1201 | 473 | 0 |
| clip_pretrained_text_only | text | 10 | 0.6732 | 0.1345 | 0.1694 | 473 | 0 |
| clip_finetuned_text_only | text | 1 | 0.8140 | 0.0220 | 0.0394 | 473 | 0 |
| clip_finetuned_text_only | text | 5 | 0.7290 | 0.0843 | 0.1199 | 473 | 0 |
| clip_finetuned_text_only | text | 10 | 0.6763 | 0.1349 | 0.1688 | 473 | 0 |
| resnet18_image_only | image | 1 | 0.6723 | 0.0148 | 0.0272 | 473 | 0 |
| resnet18_image_only | image | 5 | 0.6224 | 0.0617 | 0.0907 | 473 | 0 |
| resnet18_image_only | image | 10 | 0.5782 | 0.0980 | 0.1286 | 473 | 0 |
| clip_pretrained_image_only | image | 1 | 0.7632 | 0.0200 | 0.0358 | 473 | 0 |
| clip_pretrained_image_only | image | 5 | 0.6884 | 0.0780 | 0.1113 | 473 | 0 |
| clip_pretrained_image_only | image | 10 | 0.6488 | 0.1291 | 0.1612 | 473 | 0 |
| clip_finetuned_image_only | image | 1 | 0.7294 | 0.0204 | 0.0360 | 473 | 0 |
| clip_finetuned_image_only | image | 5 | 0.7049 | 0.0835 | 0.1173 | 473 | 0 |
| clip_finetuned_image_only | image | 10 | 0.6710 | 0.1345 | 0.1677 | 473 | 0 |
| sbert_resnet18_rrf | image + text | 1 | 0.7992 | 0.0231 | 0.0408 | 473 | 0 |
| sbert_resnet18_rrf | image + text | 5 | 0.7522 | 0.0907 | 0.1275 | 473 | 0 |
| sbert_resnet18_rrf | image + text | 10 | 0.7076 | 0.1451 | 0.1794 | 473 | 0 |
| clip_finetuned_raw_cosine_fusion | image + text | 1 | 0.8520 | 0.0248 | 0.0437 | 473 | 0 |
| clip_finetuned_raw_cosine_fusion | image + text | 5 | 0.7459 | 0.0915 | 0.1282 | 473 | 0 |
| clip_finetuned_raw_cosine_fusion | image + text | 10 | 0.7112 | 0.1506 | 0.1860 | 473 | 0 |
| clip_finetuned_percentile_fusion | image + text | 1 | 0.8118 | 0.0213 | 0.0379 | 473 | 0 |
| clip_finetuned_percentile_fusion | image + text | 5 | 0.7370 | 0.0888 | 0.1244 | 473 | 0 |
| clip_finetuned_percentile_fusion | image + text | 10 | 0.7004 | 0.1439 | 0.1791 | 473 | 0 |

## Interpretation boundary

This is a same-category retrieval proxy over an unrestricted full catalog.
The CLIP fusion rows do not apply category filtering or attribute overrides.
Results must not be mixed with the 473 x 473 test-only protocol without naming the gallery size.
