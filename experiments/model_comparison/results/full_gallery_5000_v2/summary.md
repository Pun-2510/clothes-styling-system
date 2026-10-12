# Unified 5,000-product comparison

The benchmark used 473 held-out test queries against 5000 catalog products.
All tables use the same query rows, catalog rows, K values and model checkpoints.

## A. Controlled modality baselines

| System | Role | Input | Formula | K | Precision | Recall | F1 |
|---|---|---|---|---:|---:|---:|---:|
| sbert_text_only | text baseline | text | `S_TT (SBERT)` | 1 | 0.8097 | 0.0257 | 0.0452 |
| sbert_text_only | text baseline | text | `S_TT (SBERT)` | 5 | 0.7125 | 0.0928 | 0.1291 |
| sbert_text_only | text baseline | text | `S_TT (SBERT)` | 10 | 0.6539 | 0.1430 | 0.1736 |
| clip_pretrained_text_only | text baseline | text | `S_TT (pretrained CLIP)` | 1 | 0.8161 | 0.0241 | 0.0428 |
| clip_pretrained_text_only | text baseline | text | `S_TT (pretrained CLIP)` | 5 | 0.7252 | 0.0834 | 0.1201 |
| clip_pretrained_text_only | text baseline | text | `S_TT (pretrained CLIP)` | 10 | 0.6732 | 0.1345 | 0.1694 |
| clip_finetuned_text_only | text baseline | text | `S_TT (fine-tuned CLIP)` | 1 | 0.8140 | 0.0220 | 0.0394 |
| clip_finetuned_text_only | text baseline | text | `S_TT (fine-tuned CLIP)` | 5 | 0.7290 | 0.0843 | 0.1199 |
| clip_finetuned_text_only | text baseline | text | `S_TT (fine-tuned CLIP)` | 10 | 0.6763 | 0.1349 | 0.1688 |
| resnet18_image_only | image baseline | image | `S_II (ResNet18)` | 1 | 0.6723 | 0.0148 | 0.0272 |
| resnet18_image_only | image baseline | image | `S_II (ResNet18)` | 5 | 0.6224 | 0.0617 | 0.0907 |
| resnet18_image_only | image baseline | image | `S_II (ResNet18)` | 10 | 0.5782 | 0.0980 | 0.1286 |
| clip_pretrained_image_only | image baseline | image | `S_II (pretrained CLIP)` | 1 | 0.7632 | 0.0200 | 0.0358 |
| clip_pretrained_image_only | image baseline | image | `S_II (pretrained CLIP)` | 5 | 0.6884 | 0.0780 | 0.1113 |
| clip_pretrained_image_only | image baseline | image | `S_II (pretrained CLIP)` | 10 | 0.6488 | 0.1291 | 0.1612 |
| clip_finetuned_image_only | image baseline | image | `S_II (fine-tuned CLIP)` | 1 | 0.7294 | 0.0204 | 0.0360 |
| clip_finetuned_image_only | image baseline | image | `S_II (fine-tuned CLIP)` | 5 | 0.7049 | 0.0835 | 0.1173 |
| clip_finetuned_image_only | image baseline | image | `S_II (fine-tuned CLIP)` | 10 | 0.6710 | 0.1345 | 0.1677 |
| sbert_resnet18_rrf | independent-model fusion | image + text | `RRF(S_TT, S_II)` | 1 | 0.7992 | 0.0231 | 0.0408 |
| sbert_resnet18_rrf | independent-model fusion | image + text | `RRF(S_TT, S_II)` | 5 | 0.7522 | 0.0907 | 0.1275 |
| sbert_resnet18_rrf | independent-model fusion | image + text | `RRF(S_TT, S_II)` | 10 | 0.7076 | 0.1451 | 0.1794 |

## B. Pretrained CLIP versus fine-tuned CLIP

| System | Role | Input | Formula | K | Precision | Recall | F1 |
|---|---|---|---|---:|---:|---:|---:|
| clip_pretrained_text_to_image | pretrained CLIP | text -> image | `S_TI` | 1 | 0.2093 | 0.2093 | 0.2093 |
| clip_pretrained_text_to_image | pretrained CLIP | text -> image | `S_TI` | 5 | 0.0968 | 0.4841 | 0.1614 |
| clip_pretrained_text_to_image | pretrained CLIP | text -> image | `S_TI` | 10 | 0.0609 | 0.6089 | 0.1107 |
| clip_finetuned_text_to_image | fine-tuned CLIP | text -> image | `S_TI` | 1 | 0.2600 | 0.2600 | 0.2600 |
| clip_finetuned_text_to_image | fine-tuned CLIP | text -> image | `S_TI` | 5 | 0.1104 | 0.5518 | 0.1839 |
| clip_finetuned_text_to_image | fine-tuned CLIP | text -> image | `S_TI` | 10 | 0.0704 | 0.7040 | 0.1280 |
| clip_pretrained_image_to_text | pretrained CLIP | image -> text | `S_IT` | 1 | 0.1670 | 0.1670 | 0.1670 |
| clip_pretrained_image_to_text | pretrained CLIP | image -> text | `S_IT` | 5 | 0.0871 | 0.4355 | 0.1452 |
| clip_pretrained_image_to_text | pretrained CLIP | image -> text | `S_IT` | 10 | 0.0560 | 0.5603 | 0.1019 |
| clip_finetuned_image_to_text | fine-tuned CLIP | image -> text | `S_IT` | 1 | 0.2770 | 0.2770 | 0.2770 |
| clip_finetuned_image_to_text | fine-tuned CLIP | image -> text | `S_IT` | 5 | 0.1159 | 0.5793 | 0.1931 |
| clip_finetuned_image_to_text | fine-tuned CLIP | image -> text | `S_IT` | 10 | 0.0723 | 0.7230 | 0.1315 |

## C. Website serving ablation

| System | Role | Input | Formula | K | Precision | Recall | F1 |
|---|---|---|---|---:|---:|---:|---:|
| clip_finetuned_image_serving | image serving | image | `S_II` | 1 | 0.7294 | 0.0204 | 0.0360 |
| clip_finetuned_image_serving | image serving | image | `S_II` | 5 | 0.7049 | 0.0835 | 0.1173 |
| clip_finetuned_image_serving | image serving | image | `S_II` | 10 | 0.6710 | 0.1345 | 0.1677 |
| clip_finetuned_text_to_image_component | text component | text | `S_TI` | 1 | 0.6913 | 0.0143 | 0.0267 |
| clip_finetuned_text_to_image_component | text component | text | `S_TI` | 5 | 0.6685 | 0.0659 | 0.0996 |
| clip_finetuned_text_to_image_component | text component | text | `S_TI` | 10 | 0.6440 | 0.1161 | 0.1513 |
| clip_finetuned_text_to_text_component | text component | text | `S_TT` | 1 | 0.8140 | 0.0220 | 0.0394 |
| clip_finetuned_text_to_text_component | text component | text | `S_TT` | 5 | 0.7290 | 0.0843 | 0.1199 |
| clip_finetuned_text_to_text_component | text component | text | `S_TT` | 10 | 0.6763 | 0.1349 | 0.1688 |
| clip_finetuned_text_serving_raw | text serving | text | `w_t*S_TI + (1-w_t)*S_TT` | 1 | 0.8266 | 0.0229 | 0.0409 |
| clip_finetuned_text_serving_raw | text serving | text | `w_t*S_TI + (1-w_t)*S_TT` | 5 | 0.7319 | 0.0833 | 0.1198 |
| clip_finetuned_text_serving_raw | text serving | text | `w_t*S_TI + (1-w_t)*S_TT` | 10 | 0.6882 | 0.1368 | 0.1719 |
| clip_finetuned_text_serving_percentile | text fusion ablation | text | `w_t*P(S_TI) + (1-w_t)*P(S_TT)` | 1 | 0.7801 | 0.0183 | 0.0335 |
| clip_finetuned_text_serving_percentile | text fusion ablation | text | `w_t*P(S_TI) + (1-w_t)*P(S_TT)` | 5 | 0.7137 | 0.0766 | 0.1121 |
| clip_finetuned_text_serving_percentile | text fusion ablation | text | `w_t*P(S_TI) + (1-w_t)*P(S_TT)` | 10 | 0.6732 | 0.1276 | 0.1637 |
| clip_finetuned_multimodal_raw | multimodal fusion ablation | image + text | `w_i*S_II + (1-w_i)*(w_t*S_TI + (1-w_t)*S_TT)` | 1 | 0.8520 | 0.0248 | 0.0437 |
| clip_finetuned_multimodal_raw | multimodal fusion ablation | image + text | `w_i*S_II + (1-w_i)*(w_t*S_TI + (1-w_t)*S_TT)` | 5 | 0.7459 | 0.0915 | 0.1282 |
| clip_finetuned_multimodal_raw | multimodal fusion ablation | image + text | `w_i*S_II + (1-w_i)*(w_t*S_TI + (1-w_t)*S_TT)` | 10 | 0.7112 | 0.1506 | 0.1860 |
| clip_finetuned_multimodal_percentile | multimodal serving | image + text | `w_i*P(S_II) + (1-w_i)*(w_t*P(S_TI) + (1-w_t)*P(S_TT))` | 1 | 0.8118 | 0.0213 | 0.0379 |
| clip_finetuned_multimodal_percentile | multimodal serving | image + text | `w_i*P(S_II) + (1-w_i)*(w_t*P(S_TI) + (1-w_t)*P(S_TT))` | 5 | 0.7370 | 0.0888 | 0.1244 |
| clip_finetuned_multimodal_percentile | multimodal serving | image + text | `w_i*P(S_II) + (1-w_i)*(w_t*P(S_TI) + (1-w_t)*P(S_TT))` | 10 | 0.7004 | 0.1439 | 0.1791 |

## Interpretation boundary

Controlled and serving tables exclude the query product and use same-category relevance.
The CLIP alignment table keeps the paired product and uses exact product-ID relevance.
Category labels are used only after unrestricted ranking; they are omitted from encoder text.
Serving rows do not apply category filtering, translation or attribute overrides.
Results must not be mixed with the 473 x 473 test-only protocol without naming the gallery size.
