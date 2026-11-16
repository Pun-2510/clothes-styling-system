# Unified 5,000-product comparison

The benchmark used 460 held-out test queries against 5000 catalog products.
All tables use the same query rows, catalog rows, K values and model checkpoints.

## A. Controlled modality baselines

| System | Role | Input | Formula | K | Precision | Recall | F1 | NDCG | MAP |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| tfidf_text_only | text baseline | text | `cosine(TF-IDF)` | 1 | 0.7022 | 0.0241 | 0.0432 | 0.7022 | 0.7022 |
| tfidf_text_only | text baseline | text | `cosine(TF-IDF)` | 5 | 0.6374 | 0.0987 | 0.1452 | 0.6555 | 0.5745 |
| tfidf_text_only | text baseline | text | `cosine(TF-IDF)` | 10 | 0.5941 | 0.1628 | 0.2125 | 0.6297 | 0.5158 |
| bert_text_only | text baseline | text | `S_TT (pretrained BERT)` | 1 | 0.6783 | 0.0222 | 0.0398 | 0.6783 | 0.6783 |
| bert_text_only | text baseline | text | `S_TT (pretrained BERT)` | 5 | 0.5739 | 0.0774 | 0.1174 | 0.6001 | 0.5241 |
| bert_text_only | text baseline | text | `S_TT (pretrained BERT)` | 10 | 0.5004 | 0.1205 | 0.1606 | 0.5475 | 0.4363 |
| sbert_text_only | text baseline | text | `S_TT (SBERT)` | 1 | 0.7130 | 0.0243 | 0.0435 | 0.7130 | 0.7130 |
| sbert_text_only | text baseline | text | `S_TT (SBERT)` | 5 | 0.6304 | 0.0920 | 0.1345 | 0.6544 | 0.5734 |
| sbert_text_only | text baseline | text | `S_TT (SBERT)` | 10 | 0.5630 | 0.1457 | 0.1898 | 0.6096 | 0.4942 |
| clip_pretrained_text_only | text baseline | text | `S_TT (pretrained CLIP)` | 1 | 0.7196 | 0.0218 | 0.0398 | 0.7196 | 0.7196 |
| clip_pretrained_text_only | text baseline | text | `S_TT (pretrained CLIP)` | 5 | 0.6404 | 0.0839 | 0.1283 | 0.6606 | 0.5786 |
| clip_pretrained_text_only | text baseline | text | `S_TT (pretrained CLIP)` | 10 | 0.5889 | 0.1399 | 0.1905 | 0.6252 | 0.5109 |
| clip_finetuned_text_only | text baseline | text | `S_TT (fine-tuned CLIP)` | 1 | 0.7435 | 0.0218 | 0.0398 | 0.7435 | 0.7435 |
| clip_finetuned_text_only | text baseline | text | `S_TT (fine-tuned CLIP)` | 5 | 0.6465 | 0.0820 | 0.1258 | 0.6700 | 0.5928 |
| clip_finetuned_text_only | text baseline | text | `S_TT (fine-tuned CLIP)` | 10 | 0.5848 | 0.1362 | 0.1862 | 0.6262 | 0.5171 |
| resnet18_image_only | image baseline | image | `S_II (ResNet18)` | 1 | 0.6174 | 0.0171 | 0.0317 | 0.6174 | 0.6174 |
| resnet18_image_only | image baseline | image | `S_II (ResNet18)` | 5 | 0.5583 | 0.0679 | 0.1038 | 0.5769 | 0.4940 |
| resnet18_image_only | image baseline | image | `S_II (ResNet18)` | 10 | 0.5102 | 0.1139 | 0.1532 | 0.5439 | 0.4348 |
| efficientnet_b0_image_only | image baseline | image | `S_II (ImageNet EfficientNet-B0)` | 1 | 0.6891 | 0.0194 | 0.0355 | 0.6891 | 0.6891 |
| efficientnet_b0_image_only | image baseline | image | `S_II (ImageNet EfficientNet-B0)` | 5 | 0.5787 | 0.0721 | 0.1095 | 0.6035 | 0.5242 |
| efficientnet_b0_image_only | image baseline | image | `S_II (ImageNet EfficientNet-B0)` | 10 | 0.5428 | 0.1218 | 0.1632 | 0.5773 | 0.4693 |
| vit_b16_image_only | image baseline | image | `S_II (ImageNet ViT-B/16)` | 1 | 0.6783 | 0.0194 | 0.0353 | 0.6783 | 0.6783 |
| vit_b16_image_only | image baseline | image | `S_II (ImageNet ViT-B/16)` | 5 | 0.5752 | 0.0736 | 0.1105 | 0.6002 | 0.5218 |
| vit_b16_image_only | image baseline | image | `S_II (ImageNet ViT-B/16)` | 10 | 0.5378 | 0.1198 | 0.1619 | 0.5729 | 0.4673 |
| clip_pretrained_image_only | image baseline | image | `S_II (pretrained CLIP)` | 1 | 0.6804 | 0.0206 | 0.0373 | 0.6804 | 0.6804 |
| clip_pretrained_image_only | image baseline | image | `S_II (pretrained CLIP)` | 5 | 0.5996 | 0.0772 | 0.1178 | 0.6180 | 0.5341 |
| clip_pretrained_image_only | image baseline | image | `S_II (pretrained CLIP)` | 10 | 0.5524 | 0.1264 | 0.1713 | 0.5858 | 0.4716 |
| clip_finetuned_image_only | image baseline | image | `S_II (fine-tuned CLIP)` | 1 | 0.6957 | 0.0185 | 0.0346 | 0.6957 | 0.6957 |
| clip_finetuned_image_only | image baseline | image | `S_II (fine-tuned CLIP)` | 5 | 0.6100 | 0.0783 | 0.1193 | 0.6289 | 0.5518 |
| clip_finetuned_image_only | image baseline | image | `S_II (fine-tuned CLIP)` | 10 | 0.5676 | 0.1299 | 0.1755 | 0.5999 | 0.4920 |
| sbert_resnet18_rrf | independent-model fusion | image + text | `RRF(S_TT, S_II)` | 1 | 0.7739 | 0.0242 | 0.0437 | 0.7739 | 0.7739 |
| sbert_resnet18_rrf | independent-model fusion | image + text | `RRF(S_TT, S_II)` | 5 | 0.6817 | 0.0946 | 0.1407 | 0.7045 | 0.6270 |
| sbert_resnet18_rrf | independent-model fusion | image + text | `RRF(S_TT, S_II)` | 10 | 0.6391 | 0.1570 | 0.2091 | 0.6768 | 0.5687 |

## B. Pretrained CLIP versus fine-tuned CLIP

| System | Role | Input | Formula | K | Precision | Recall | F1 | NDCG | MAP |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| clip_pretrained_text_to_image | pretrained CLIP | text -> image | `S_TI` | 1 | 0.2435 | 0.2435 | 0.2435 | 0.2435 | 0.2435 |
| clip_pretrained_text_to_image | pretrained CLIP | text -> image | `S_TI` | 5 | 0.1017 | 0.5087 | 0.1696 | 0.3814 | 0.3392 |
| clip_pretrained_text_to_image | pretrained CLIP | text -> image | `S_TI` | 10 | 0.0650 | 0.6500 | 0.1182 | 0.4274 | 0.3584 |
| clip_finetuned_text_to_image | fine-tuned CLIP | text -> image | `S_TI` | 1 | 0.2913 | 0.2913 | 0.2913 | 0.2913 | 0.2913 |
| clip_finetuned_text_to_image | fine-tuned CLIP | text -> image | `S_TI` | 5 | 0.1209 | 0.6043 | 0.2014 | 0.4529 | 0.4029 |
| clip_finetuned_text_to_image | fine-tuned CLIP | text -> image | `S_TI` | 10 | 0.0739 | 0.7391 | 0.1344 | 0.4965 | 0.4209 |
| clip_pretrained_image_to_text | pretrained CLIP | image -> text | `S_IT` | 1 | 0.2370 | 0.2370 | 0.2370 | 0.2370 | 0.2370 |
| clip_pretrained_image_to_text | pretrained CLIP | image -> text | `S_IT` | 5 | 0.0978 | 0.4891 | 0.1630 | 0.3685 | 0.3284 |
| clip_pretrained_image_to_text | pretrained CLIP | image -> text | `S_IT` | 10 | 0.0630 | 0.6304 | 0.1146 | 0.4135 | 0.3466 |
| clip_finetuned_image_to_text | fine-tuned CLIP | image -> text | `S_IT` | 1 | 0.3043 | 0.3043 | 0.3043 | 0.3043 | 0.3043 |
| clip_finetuned_image_to_text | fine-tuned CLIP | image -> text | `S_IT` | 5 | 0.1239 | 0.6196 | 0.2065 | 0.4672 | 0.4169 |
| clip_finetuned_image_to_text | fine-tuned CLIP | image -> text | `S_IT` | 10 | 0.0759 | 0.7587 | 0.1379 | 0.5122 | 0.4355 |

## C. Website serving ablation

| System | Role | Input | Formula | K | Precision | Recall | F1 | NDCG | MAP |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| clip_finetuned_image_serving | image serving | image | `S_II` | 1 | 0.6957 | 0.0185 | 0.0346 | 0.6957 | 0.6957 |
| clip_finetuned_image_serving | image serving | image | `S_II` | 5 | 0.6100 | 0.0783 | 0.1193 | 0.6289 | 0.5518 |
| clip_finetuned_image_serving | image serving | image | `S_II` | 10 | 0.5676 | 0.1299 | 0.1755 | 0.5999 | 0.4920 |
| clip_finetuned_text_to_image_component | text component | text | `S_TI` | 1 | 0.5957 | 0.0139 | 0.0261 | 0.5957 | 0.5957 |
| clip_finetuned_text_to_image_component | text component | text | `S_TI` | 5 | 0.5813 | 0.0653 | 0.1049 | 0.5879 | 0.4994 |
| clip_finetuned_text_to_image_component | text component | text | `S_TI` | 10 | 0.5348 | 0.1133 | 0.1609 | 0.5573 | 0.4384 |
| clip_finetuned_text_to_text_component | text component | text | `S_TT` | 1 | 0.7435 | 0.0218 | 0.0398 | 0.7435 | 0.7435 |
| clip_finetuned_text_to_text_component | text component | text | `S_TT` | 5 | 0.6465 | 0.0820 | 0.1258 | 0.6700 | 0.5928 |
| clip_finetuned_text_to_text_component | text component | text | `S_TT` | 10 | 0.5848 | 0.1362 | 0.1862 | 0.6262 | 0.5171 |
| clip_finetuned_text_serving_raw | text serving | text | `w_t*S_TI + (1-w_t)*S_TT` | 1 | 0.7435 | 0.0210 | 0.0384 | 0.7435 | 0.7435 |
| clip_finetuned_text_serving_raw | text serving | text | `w_t*S_TI + (1-w_t)*S_TT` | 5 | 0.6574 | 0.0834 | 0.1284 | 0.6793 | 0.6035 |
| clip_finetuned_text_serving_raw | text serving | text | `w_t*S_TI + (1-w_t)*S_TT` | 10 | 0.5963 | 0.1382 | 0.1891 | 0.6361 | 0.5274 |
| clip_pretrained_multimodal_raw | multimodal baseline | image + text | `w_i*S_II + (1-w_i)*(w_t*S_TI + (1-w_t)*S_TT)` | 1 | 0.7543 | 0.0217 | 0.0397 | 0.7543 | 0.7543 |
| clip_pretrained_multimodal_raw | multimodal baseline | image + text | `w_i*S_II + (1-w_i)*(w_t*S_TI + (1-w_t)*S_TT)` | 5 | 0.6883 | 0.0914 | 0.1385 | 0.7065 | 0.6325 |
| clip_pretrained_multimodal_raw | multimodal baseline | image + text | `w_i*S_II + (1-w_i)*(w_t*S_TI + (1-w_t)*S_TT)` | 10 | 0.6389 | 0.1526 | 0.2081 | 0.6736 | 0.5665 |
| clip_finetuned_multimodal_raw | multimodal serving | image + text | `w_i*S_II + (1-w_i)*(w_t*S_TI + (1-w_t)*S_TT)` | 1 | 0.7348 | 0.0215 | 0.0393 | 0.7348 | 0.7348 |
| clip_finetuned_multimodal_raw | multimodal serving | image + text | `w_i*S_II + (1-w_i)*(w_t*S_TI + (1-w_t)*S_TT)` | 5 | 0.6813 | 0.0881 | 0.1350 | 0.6963 | 0.6225 |
| clip_finetuned_multimodal_raw | multimodal serving | image + text | `w_i*S_II + (1-w_i)*(w_t*S_TI + (1-w_t)*S_TT)` | 10 | 0.6350 | 0.1506 | 0.2046 | 0.6670 | 0.5629 |

## Interpretation boundary

Controlled and serving tables exclude the query product and use same-category relevance.
The CLIP alignment table keeps the paired product and uses exact product-ID relevance.
Category labels are used only after unrestricted ranking; they are omitted from encoder text.
Serving rows do not apply category filtering, translation or attribute overrides.
Results must not be mixed with the 473 x 473 test-only protocol without naming the gallery size.
