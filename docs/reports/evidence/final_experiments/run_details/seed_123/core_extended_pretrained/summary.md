# Unified 5,000-product comparison

The benchmark used 464 held-out test queries against 5000 catalog products.
All tables use the same query rows, catalog rows, K values and model checkpoints.

## A. Controlled modality baselines

| System | Role | Input | Formula | K | Precision | Recall | F1 | NDCG | MAP |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| tfidf_text_only | text baseline | text | `cosine(TF-IDF)` | 1 | 0.6810 | 0.0234 | 0.0422 | 0.6810 | 0.6810 |
| tfidf_text_only | text baseline | text | `cosine(TF-IDF)` | 5 | 0.6440 | 0.0974 | 0.1454 | 0.6557 | 0.5876 |
| tfidf_text_only | text baseline | text | `cosine(TF-IDF)` | 10 | 0.6017 | 0.1672 | 0.2178 | 0.6328 | 0.5302 |
| bert_text_only | text baseline | text | `S_TT (pretrained BERT)` | 1 | 0.6379 | 0.0190 | 0.0352 | 0.6379 | 0.6379 |
| bert_text_only | text baseline | text | `S_TT (pretrained BERT)` | 5 | 0.5595 | 0.0722 | 0.1134 | 0.5783 | 0.5073 |
| bert_text_only | text baseline | text | `S_TT (pretrained BERT)` | 10 | 0.4905 | 0.1148 | 0.1589 | 0.5293 | 0.4281 |
| sbert_text_only | text baseline | text | `S_TT (SBERT)` | 1 | 0.6810 | 0.0235 | 0.0421 | 0.6810 | 0.6810 |
| sbert_text_only | text baseline | text | `S_TT (SBERT)` | 5 | 0.6155 | 0.0887 | 0.1328 | 0.6357 | 0.5554 |
| sbert_text_only | text baseline | text | `S_TT (SBERT)` | 10 | 0.5619 | 0.1459 | 0.1935 | 0.6007 | 0.4873 |
| clip_pretrained_text_only | text baseline | text | `S_TT (pretrained CLIP)` | 1 | 0.7177 | 0.0213 | 0.0389 | 0.7177 | 0.7177 |
| clip_pretrained_text_only | text baseline | text | `S_TT (pretrained CLIP)` | 5 | 0.6384 | 0.0834 | 0.1281 | 0.6593 | 0.5803 |
| clip_pretrained_text_only | text baseline | text | `S_TT (pretrained CLIP)` | 10 | 0.5819 | 0.1409 | 0.1913 | 0.6193 | 0.5074 |
| clip_finetuned_text_only | text baseline | text | `S_TT (fine-tuned CLIP)` | 1 | 0.7435 | 0.0219 | 0.0397 | 0.7435 | 0.7435 |
| clip_finetuned_text_only | text baseline | text | `S_TT (fine-tuned CLIP)` | 5 | 0.6349 | 0.0833 | 0.1266 | 0.6625 | 0.5897 |
| clip_finetuned_text_only | text baseline | text | `S_TT (fine-tuned CLIP)` | 10 | 0.5754 | 0.1355 | 0.1838 | 0.6189 | 0.5097 |
| resnet18_image_only | image baseline | image | `S_II (ResNet18)` | 1 | 0.6379 | 0.0184 | 0.0335 | 0.6379 | 0.6379 |
| resnet18_image_only | image baseline | image | `S_II (ResNet18)` | 5 | 0.5517 | 0.0727 | 0.1086 | 0.5727 | 0.4946 |
| resnet18_image_only | image baseline | image | `S_II (ResNet18)` | 10 | 0.5127 | 0.1193 | 0.1606 | 0.5451 | 0.4408 |
| efficientnet_b0_image_only | image baseline | image | `S_II (ImageNet EfficientNet-B0)` | 1 | 0.6530 | 0.0194 | 0.0352 | 0.6530 | 0.6530 |
| efficientnet_b0_image_only | image baseline | image | `S_II (ImageNet EfficientNet-B0)` | 5 | 0.5690 | 0.0745 | 0.1123 | 0.5903 | 0.5114 |
| efficientnet_b0_image_only | image baseline | image | `S_II (ImageNet EfficientNet-B0)` | 10 | 0.5302 | 0.1247 | 0.1681 | 0.5634 | 0.4543 |
| vit_b16_image_only | image baseline | image | `S_II (ImageNet ViT-B/16)` | 1 | 0.6422 | 0.0182 | 0.0337 | 0.6422 | 0.6422 |
| vit_b16_image_only | image baseline | image | `S_II (ImageNet ViT-B/16)` | 5 | 0.5797 | 0.0734 | 0.1131 | 0.5948 | 0.5215 |
| vit_b16_image_only | image baseline | image | `S_II (ImageNet ViT-B/16)` | 10 | 0.5313 | 0.1208 | 0.1654 | 0.5622 | 0.4617 |
| clip_pretrained_image_only | image baseline | image | `S_II (pretrained CLIP)` | 1 | 0.6573 | 0.0194 | 0.0354 | 0.6573 | 0.6573 |
| clip_pretrained_image_only | image baseline | image | `S_II (pretrained CLIP)` | 5 | 0.5884 | 0.0765 | 0.1170 | 0.6049 | 0.5193 |
| clip_pretrained_image_only | image baseline | image | `S_II (pretrained CLIP)` | 10 | 0.5414 | 0.1276 | 0.1726 | 0.5733 | 0.4563 |
| clip_finetuned_image_only | image baseline | image | `S_II (fine-tuned CLIP)` | 1 | 0.6401 | 0.0183 | 0.0333 | 0.6401 | 0.6401 |
| clip_finetuned_image_only | image baseline | image | `S_II (fine-tuned CLIP)` | 5 | 0.5944 | 0.0775 | 0.1176 | 0.6060 | 0.5213 |
| clip_finetuned_image_only | image baseline | image | `S_II (fine-tuned CLIP)` | 10 | 0.5545 | 0.1295 | 0.1762 | 0.5804 | 0.4675 |
| sbert_resnet18_rrf | independent-model fusion | image + text | `RRF(S_TT, S_II)` | 1 | 0.7220 | 0.0232 | 0.0419 | 0.7220 | 0.7220 |
| sbert_resnet18_rrf | independent-model fusion | image + text | `RRF(S_TT, S_II)` | 5 | 0.6853 | 0.0977 | 0.1456 | 0.6968 | 0.6256 |
| sbert_resnet18_rrf | independent-model fusion | image + text | `RRF(S_TT, S_II)` | 10 | 0.6407 | 0.1620 | 0.2157 | 0.6714 | 0.5703 |

## B. Pretrained CLIP versus fine-tuned CLIP

| System | Role | Input | Formula | K | Precision | Recall | F1 | NDCG | MAP |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| clip_pretrained_text_to_image | pretrained CLIP | text -> image | `S_TI` | 1 | 0.2328 | 0.2328 | 0.2328 | 0.2328 | 0.2328 |
| clip_pretrained_text_to_image | pretrained CLIP | text -> image | `S_TI` | 5 | 0.1039 | 0.5194 | 0.1731 | 0.3843 | 0.3394 |
| clip_pretrained_text_to_image | pretrained CLIP | text -> image | `S_TI` | 10 | 0.0638 | 0.6379 | 0.1160 | 0.4228 | 0.3554 |
| clip_finetuned_text_to_image | fine-tuned CLIP | text -> image | `S_TI` | 1 | 0.2996 | 0.2996 | 0.2996 | 0.2996 | 0.2996 |
| clip_finetuned_text_to_image | fine-tuned CLIP | text -> image | `S_TI` | 5 | 0.1250 | 0.6250 | 0.2083 | 0.4724 | 0.4216 |
| clip_finetuned_text_to_image | fine-tuned CLIP | text -> image | `S_TI` | 10 | 0.0765 | 0.7651 | 0.1391 | 0.5178 | 0.4404 |
| clip_pretrained_image_to_text | pretrained CLIP | image -> text | `S_IT` | 1 | 0.2155 | 0.2155 | 0.2155 | 0.2155 | 0.2155 |
| clip_pretrained_image_to_text | pretrained CLIP | image -> text | `S_IT` | 5 | 0.1013 | 0.5065 | 0.1688 | 0.3675 | 0.3216 |
| clip_pretrained_image_to_text | pretrained CLIP | image -> text | `S_IT` | 10 | 0.0647 | 0.6466 | 0.1176 | 0.4123 | 0.3398 |
| clip_finetuned_image_to_text | fine-tuned CLIP | image -> text | `S_IT` | 1 | 0.3017 | 0.3017 | 0.3017 | 0.3017 | 0.3017 |
| clip_finetuned_image_to_text | fine-tuned CLIP | image -> text | `S_IT` | 5 | 0.1241 | 0.6207 | 0.2069 | 0.4678 | 0.4173 |
| clip_finetuned_image_to_text | fine-tuned CLIP | image -> text | `S_IT` | 10 | 0.0741 | 0.7414 | 0.1348 | 0.5070 | 0.4336 |

## C. Website serving ablation

| System | Role | Input | Formula | K | Precision | Recall | F1 | NDCG | MAP |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| clip_finetuned_image_serving | image serving | image | `S_II` | 1 | 0.6401 | 0.0183 | 0.0333 | 0.6401 | 0.6401 |
| clip_finetuned_image_serving | image serving | image | `S_II` | 5 | 0.5944 | 0.0775 | 0.1176 | 0.6060 | 0.5213 |
| clip_finetuned_image_serving | image serving | image | `S_II` | 10 | 0.5545 | 0.1295 | 0.1762 | 0.5804 | 0.4675 |
| clip_finetuned_text_to_image_component | text component | text | `S_TI` | 1 | 0.5905 | 0.0147 | 0.0274 | 0.5905 | 0.5905 |
| clip_finetuned_text_to_image_component | text component | text | `S_TI` | 5 | 0.5647 | 0.0635 | 0.1024 | 0.5733 | 0.4892 |
| clip_finetuned_text_to_image_component | text component | text | `S_TI` | 10 | 0.5274 | 0.1102 | 0.1577 | 0.5479 | 0.4293 |
| clip_finetuned_text_to_text_component | text component | text | `S_TT` | 1 | 0.7435 | 0.0219 | 0.0397 | 0.7435 | 0.7435 |
| clip_finetuned_text_to_text_component | text component | text | `S_TT` | 5 | 0.6349 | 0.0833 | 0.1266 | 0.6625 | 0.5897 |
| clip_finetuned_text_to_text_component | text component | text | `S_TT` | 10 | 0.5754 | 0.1355 | 0.1838 | 0.6189 | 0.5097 |
| clip_finetuned_text_serving_raw | text serving | text | `w_t*S_TI + (1-w_t)*S_TT` | 1 | 0.7091 | 0.0211 | 0.0383 | 0.7091 | 0.7091 |
| clip_finetuned_text_serving_raw | text serving | text | `w_t*S_TI + (1-w_t)*S_TT` | 5 | 0.6478 | 0.0826 | 0.1271 | 0.6667 | 0.5954 |
| clip_finetuned_text_serving_raw | text serving | text | `w_t*S_TI + (1-w_t)*S_TT` | 10 | 0.5894 | 0.1383 | 0.1887 | 0.6259 | 0.5201 |
| clip_pretrained_multimodal_raw | multimodal baseline | image + text | `w_i*S_II + (1-w_i)*(w_t*S_TI + (1-w_t)*S_TT)` | 1 | 0.7522 | 0.0249 | 0.0444 | 0.7522 | 0.7522 |
| clip_pretrained_multimodal_raw | multimodal baseline | image + text | `w_i*S_II + (1-w_i)*(w_t*S_TI + (1-w_t)*S_TT)` | 5 | 0.6711 | 0.0919 | 0.1382 | 0.6921 | 0.6163 |
| clip_pretrained_multimodal_raw | multimodal baseline | image + text | `w_i*S_II + (1-w_i)*(w_t*S_TI + (1-w_t)*S_TT)` | 10 | 0.6338 | 0.1565 | 0.2109 | 0.6668 | 0.5598 |
| clip_finetuned_multimodal_raw | multimodal serving | image + text | `w_i*S_II + (1-w_i)*(w_t*S_TI + (1-w_t)*S_TT)` | 1 | 0.7091 | 0.0211 | 0.0386 | 0.7091 | 0.7091 |
| clip_finetuned_multimodal_raw | multimodal serving | image + text | `w_i*S_II + (1-w_i)*(w_t*S_TI + (1-w_t)*S_TT)` | 5 | 0.6612 | 0.0891 | 0.1343 | 0.6770 | 0.6011 |
| clip_finetuned_multimodal_raw | multimodal serving | image + text | `w_i*S_II + (1-w_i)*(w_t*S_TI + (1-w_t)*S_TT)` | 10 | 0.6119 | 0.1478 | 0.2006 | 0.6448 | 0.5371 |

## Interpretation boundary

Controlled and serving tables exclude the query product and use same-category relevance.
The CLIP alignment table keeps the paired product and uses exact product-ID relevance.
Category labels are used only after unrestricted ranking; they are omitted from encoder text.
Serving rows do not apply category filtering, translation or attribute overrides.
Results must not be mixed with the 473 x 473 test-only protocol without naming the gallery size.
