# Unified 5,000-product comparison

The benchmark used 462 held-out test queries against 5000 catalog products.
All tables use the same query rows, catalog rows, K values and model checkpoints.

## A. Controlled modality baselines

| System | Role | Input | Formula | K | Precision | Recall | F1 | NDCG | MAP |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| tfidf_text_only | text baseline | text | `cosine(TF-IDF)` | 1 | 0.6472 | 0.0234 | 0.0417 | 0.6472 | 0.6472 |
| tfidf_text_only | text baseline | text | `cosine(TF-IDF)` | 5 | 0.6281 | 0.0953 | 0.1405 | 0.6398 | 0.5647 |
| tfidf_text_only | text baseline | text | `cosine(TF-IDF)` | 10 | 0.5892 | 0.1626 | 0.2090 | 0.6198 | 0.5106 |
| bert_text_only | text baseline | text | `S_TT (pretrained BERT)` | 1 | 0.6385 | 0.0205 | 0.0367 | 0.6385 | 0.6385 |
| bert_text_only | text baseline | text | `S_TT (pretrained BERT)` | 5 | 0.5641 | 0.0748 | 0.1142 | 0.5844 | 0.5060 |
| bert_text_only | text baseline | text | `S_TT (pretrained BERT)` | 10 | 0.4890 | 0.1169 | 0.1567 | 0.5319 | 0.4214 |
| sbert_text_only | text baseline | text | `S_TT (SBERT)` | 1 | 0.7273 | 0.0236 | 0.0426 | 0.7273 | 0.7273 |
| sbert_text_only | text baseline | text | `S_TT (SBERT)` | 5 | 0.6437 | 0.0921 | 0.1368 | 0.6656 | 0.5851 |
| sbert_text_only | text baseline | text | `S_TT (SBERT)` | 10 | 0.5896 | 0.1488 | 0.1978 | 0.6287 | 0.5185 |
| clip_pretrained_text_only | text baseline | text | `S_TT (pretrained CLIP)` | 1 | 0.7229 | 0.0222 | 0.0402 | 0.7229 | 0.7229 |
| clip_pretrained_text_only | text baseline | text | `S_TT (pretrained CLIP)` | 5 | 0.6307 | 0.0826 | 0.1251 | 0.6517 | 0.5705 |
| clip_pretrained_text_only | text baseline | text | `S_TT (pretrained CLIP)` | 10 | 0.5812 | 0.1390 | 0.1877 | 0.6172 | 0.5032 |
| clip_finetuned_text_only | text baseline | text | `S_TT (fine-tuned CLIP)` | 1 | 0.7489 | 0.0227 | 0.0415 | 0.7489 | 0.7489 |
| clip_finetuned_text_only | text baseline | text | `S_TT (fine-tuned CLIP)` | 5 | 0.6697 | 0.0887 | 0.1338 | 0.6906 | 0.6106 |
| clip_finetuned_text_only | text baseline | text | `S_TT (fine-tuned CLIP)` | 10 | 0.6074 | 0.1438 | 0.1949 | 0.6478 | 0.5384 |
| resnet18_image_only | image baseline | image | `S_II (ResNet18)` | 1 | 0.6147 | 0.0160 | 0.0297 | 0.6147 | 0.6147 |
| resnet18_image_only | image baseline | image | `S_II (ResNet18)` | 5 | 0.5437 | 0.0637 | 0.0987 | 0.5614 | 0.4824 |
| resnet18_image_only | image baseline | image | `S_II (ResNet18)` | 10 | 0.5037 | 0.1105 | 0.1505 | 0.5327 | 0.4261 |
| efficientnet_b0_image_only | image baseline | image | `S_II (ImageNet EfficientNet-B0)` | 1 | 0.6407 | 0.0182 | 0.0332 | 0.6407 | 0.6407 |
| efficientnet_b0_image_only | image baseline | image | `S_II (ImageNet EfficientNet-B0)` | 5 | 0.5511 | 0.0694 | 0.1049 | 0.5718 | 0.4953 |
| efficientnet_b0_image_only | image baseline | image | `S_II (ImageNet EfficientNet-B0)` | 10 | 0.5208 | 0.1180 | 0.1595 | 0.5500 | 0.4437 |
| vit_b16_image_only | image baseline | image | `S_II (ImageNet ViT-B/16)` | 1 | 0.6082 | 0.0171 | 0.0310 | 0.6082 | 0.6082 |
| vit_b16_image_only | image baseline | image | `S_II (ImageNet ViT-B/16)` | 5 | 0.5597 | 0.0687 | 0.1054 | 0.5750 | 0.5006 |
| vit_b16_image_only | image baseline | image | `S_II (ImageNet ViT-B/16)` | 10 | 0.5251 | 0.1144 | 0.1583 | 0.5514 | 0.4519 |
| clip_pretrained_image_only | image baseline | image | `S_II (pretrained CLIP)` | 1 | 0.6342 | 0.0177 | 0.0324 | 0.6342 | 0.6342 |
| clip_pretrained_image_only | image baseline | image | `S_II (pretrained CLIP)` | 5 | 0.5805 | 0.0716 | 0.1106 | 0.5957 | 0.5138 |
| clip_pretrained_image_only | image baseline | image | `S_II (pretrained CLIP)` | 10 | 0.5429 | 0.1213 | 0.1671 | 0.5697 | 0.4613 |
| clip_finetuned_image_only | image baseline | image | `S_II (fine-tuned CLIP)` | 1 | 0.6190 | 0.0160 | 0.0297 | 0.6190 | 0.6190 |
| clip_finetuned_image_only | image baseline | image | `S_II (fine-tuned CLIP)` | 5 | 0.5861 | 0.0719 | 0.1104 | 0.5974 | 0.5223 |
| clip_finetuned_image_only | image baseline | image | `S_II (fine-tuned CLIP)` | 10 | 0.5463 | 0.1244 | 0.1691 | 0.5716 | 0.4678 |
| sbert_resnet18_rrf | independent-model fusion | image + text | `RRF(S_TT, S_II)` | 1 | 0.7381 | 0.0210 | 0.0386 | 0.7381 | 0.7381 |
| sbert_resnet18_rrf | independent-model fusion | image + text | `RRF(S_TT, S_II)` | 5 | 0.6887 | 0.0909 | 0.1377 | 0.7022 | 0.6322 |
| sbert_resnet18_rrf | independent-model fusion | image + text | `RRF(S_TT, S_II)` | 10 | 0.6392 | 0.1535 | 0.2069 | 0.6707 | 0.5684 |

## B. Pretrained CLIP versus fine-tuned CLIP

| System | Role | Input | Formula | K | Precision | Recall | F1 | NDCG | MAP |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| clip_pretrained_text_to_image | pretrained CLIP | text -> image | `S_TI` | 1 | 0.1991 | 0.1991 | 0.1991 | 0.1991 | 0.1991 |
| clip_pretrained_text_to_image | pretrained CLIP | text -> image | `S_TI` | 5 | 0.1000 | 0.5000 | 0.1667 | 0.3572 | 0.3098 |
| clip_pretrained_text_to_image | pretrained CLIP | text -> image | `S_TI` | 10 | 0.0636 | 0.6364 | 0.1157 | 0.4012 | 0.3279 |
| clip_finetuned_text_to_image | fine-tuned CLIP | text -> image | `S_TI` | 1 | 0.2814 | 0.2814 | 0.2814 | 0.2814 | 0.2814 |
| clip_finetuned_text_to_image | fine-tuned CLIP | text -> image | `S_TI` | 5 | 0.1216 | 0.6082 | 0.2027 | 0.4592 | 0.4093 |
| clip_finetuned_text_to_image | fine-tuned CLIP | text -> image | `S_TI` | 10 | 0.0723 | 0.7229 | 0.1314 | 0.4966 | 0.4248 |
| clip_pretrained_image_to_text | pretrained CLIP | image -> text | `S_IT` | 1 | 0.2035 | 0.2035 | 0.2035 | 0.2035 | 0.2035 |
| clip_pretrained_image_to_text | pretrained CLIP | image -> text | `S_IT` | 5 | 0.0952 | 0.4762 | 0.1587 | 0.3468 | 0.3040 |
| clip_pretrained_image_to_text | pretrained CLIP | image -> text | `S_IT` | 10 | 0.0608 | 0.6082 | 0.1106 | 0.3900 | 0.3221 |
| clip_finetuned_image_to_text | fine-tuned CLIP | image -> text | `S_IT` | 1 | 0.2857 | 0.2857 | 0.2857 | 0.2857 | 0.2857 |
| clip_finetuned_image_to_text | fine-tuned CLIP | image -> text | `S_IT` | 5 | 0.1208 | 0.6039 | 0.2013 | 0.4517 | 0.4014 |
| clip_finetuned_image_to_text | fine-tuned CLIP | image -> text | `S_IT` | 10 | 0.0732 | 0.7316 | 0.1330 | 0.4929 | 0.4184 |

## C. Website serving ablation

| System | Role | Input | Formula | K | Precision | Recall | F1 | NDCG | MAP |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| clip_finetuned_image_serving | image serving | image | `S_II` | 1 | 0.6190 | 0.0160 | 0.0297 | 0.6190 | 0.6190 |
| clip_finetuned_image_serving | image serving | image | `S_II` | 5 | 0.5861 | 0.0719 | 0.1104 | 0.5974 | 0.5223 |
| clip_finetuned_image_serving | image serving | image | `S_II` | 10 | 0.5463 | 0.1244 | 0.1691 | 0.5716 | 0.4678 |
| clip_finetuned_text_to_image_component | text component | text | `S_TI` | 1 | 0.6104 | 0.0144 | 0.0271 | 0.6104 | 0.6104 |
| clip_finetuned_text_to_image_component | text component | text | `S_TI` | 5 | 0.5667 | 0.0599 | 0.0977 | 0.5769 | 0.4871 |
| clip_finetuned_text_to_image_component | text component | text | `S_TI` | 10 | 0.5370 | 0.1093 | 0.1565 | 0.5570 | 0.4380 |
| clip_finetuned_text_to_text_component | text component | text | `S_TT` | 1 | 0.7489 | 0.0227 | 0.0415 | 0.7489 | 0.7489 |
| clip_finetuned_text_to_text_component | text component | text | `S_TT` | 5 | 0.6697 | 0.0887 | 0.1338 | 0.6906 | 0.6106 |
| clip_finetuned_text_to_text_component | text component | text | `S_TT` | 10 | 0.6074 | 0.1438 | 0.1949 | 0.6478 | 0.5384 |
| clip_finetuned_text_serving_raw | text serving | text | `w_t*S_TI + (1-w_t)*S_TT` | 1 | 0.7576 | 0.0223 | 0.0408 | 0.7576 | 0.7576 |
| clip_finetuned_text_serving_raw | text serving | text | `w_t*S_TI + (1-w_t)*S_TT` | 5 | 0.6632 | 0.0863 | 0.1307 | 0.6849 | 0.6032 |
| clip_finetuned_text_serving_raw | text serving | text | `w_t*S_TI + (1-w_t)*S_TT` | 10 | 0.6175 | 0.1447 | 0.1964 | 0.6529 | 0.5424 |
| clip_pretrained_multimodal_raw | multimodal baseline | image + text | `w_i*S_II + (1-w_i)*(w_t*S_TI + (1-w_t)*S_TT)` | 1 | 0.7511 | 0.0220 | 0.0398 | 0.7511 | 0.7511 |
| clip_pretrained_multimodal_raw | multimodal baseline | image + text | `w_i*S_II + (1-w_i)*(w_t*S_TI + (1-w_t)*S_TT)` | 5 | 0.6814 | 0.0874 | 0.1327 | 0.6988 | 0.6232 |
| clip_pretrained_multimodal_raw | multimodal baseline | image + text | `w_i*S_II + (1-w_i)*(w_t*S_TI + (1-w_t)*S_TT)` | 10 | 0.6284 | 0.1472 | 0.2008 | 0.6627 | 0.5563 |
| clip_finetuned_multimodal_raw | multimodal serving | image + text | `w_i*S_II + (1-w_i)*(w_t*S_TI + (1-w_t)*S_TT)` | 1 | 0.7554 | 0.0215 | 0.0390 | 0.7554 | 0.7554 |
| clip_finetuned_multimodal_raw | multimodal serving | image + text | `w_i*S_II + (1-w_i)*(w_t*S_TI + (1-w_t)*S_TT)` | 5 | 0.6714 | 0.0861 | 0.1308 | 0.6916 | 0.6166 |
| clip_finetuned_multimodal_raw | multimodal serving | image + text | `w_i*S_II + (1-w_i)*(w_t*S_TI + (1-w_t)*S_TT)` | 10 | 0.6240 | 0.1447 | 0.1977 | 0.6587 | 0.5538 |

## Interpretation boundary

Controlled and serving tables exclude the query product and use same-category relevance.
The CLIP alignment table keeps the paired product and uses exact product-ID relevance.
Category labels are used only after unrestricted ranking; they are omitted from encoder text.
Serving rows do not apply category filtering, translation or attribute overrides.
Results must not be mixed with the 473 x 473 test-only protocol without naming the gallery size.
