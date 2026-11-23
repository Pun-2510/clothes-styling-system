# Deployed query-mode and combined-search comparison

The benchmark used 462 held-out queries against 5000 catalog products.
The query product was excluded before category prediction and ranking.
Query text omits the ground-truth category column; runtime catalog text uses the same category-bearing embeddings as the website.

## End-to-end query modes

| System | Role | Input | Formula | K | Precision | Recall | F1 | NDCG | MAP |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| image_only_deployed | deployed image query | image | `S_II + lambda_I*g(h)*w(c_i)` | 1 | 0.6515 | 0.0166 | 0.0309 | 0.6515 | 0.6515 |
| image_only_deployed | deployed image query | image | `S_II + lambda_I*g(h)*w(c_i)` | 5 | 0.6104 | 0.0721 | 0.1132 | 0.6223 | 0.5632 |
| image_only_deployed | deployed image query | image | `S_II + lambda_I*g(h)*w(c_i)` | 10 | 0.5801 | 0.1269 | 0.1776 | 0.6023 | 0.5172 |
| text_only_deployed | deployed text query | text | `CLIP text fusion + TF-IDF metadata; exact category => hard candidate pool` | 1 | 0.7576 | 0.0269 | 0.0477 | 0.7576 | 0.7576 |
| text_only_deployed | deployed text query | text | `CLIP text fusion + TF-IDF metadata; exact category => hard candidate pool` | 5 | 0.7437 | 0.1085 | 0.1624 | 0.7510 | 0.7127 |
| text_only_deployed | deployed text query | text | `CLIP text fusion + TF-IDF metadata; exact category => hard candidate pool` | 10 | 0.7154 | 0.1839 | 0.2514 | 0.7340 | 0.6797 |
| combined_full_deployed | deployed combined query | image + text | `raw cosine fusion + category-gated TF-IDF + explicit-text hard pool or dynamic image category bonus` | 1 | 0.7771 | 0.0242 | 0.0433 | 0.7771 | 0.7771 |
| combined_full_deployed | deployed combined query | image + text | `raw cosine fusion + category-gated TF-IDF + explicit-text hard pool or dynamic image category bonus` | 5 | 0.7355 | 0.1030 | 0.1553 | 0.7459 | 0.7115 |
| combined_full_deployed | deployed combined query | image + text | `raw cosine fusion + category-gated TF-IDF + explicit-text hard pool or dynamic image category bonus` | 10 | 0.7182 | 0.1781 | 0.2463 | 0.7341 | 0.6853 |

## Combined-search ablation

| System | Role | Input | Formula | K | Precision | Recall | F1 | NDCG | MAP |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| combined_base | raw fusion baseline | image + text | `w_i*S_II + (1-w_i)*(w_t*S_TI + (1-w_t)*S_TT)` | 1 | 0.7381 | 0.0212 | 0.0384 | 0.7381 | 0.7381 |
| combined_base | raw fusion baseline | image + text | `w_i*S_II + (1-w_i)*(w_t*S_TI + (1-w_t)*S_TT)` | 5 | 0.6645 | 0.0839 | 0.1284 | 0.6843 | 0.6124 |
| combined_base | raw fusion baseline | image + text | `w_i*S_II + (1-w_i)*(w_t*S_TI + (1-w_t)*S_TT)` | 10 | 0.6184 | 0.1433 | 0.1958 | 0.6525 | 0.5503 |
| combined_metadata | TF-IDF metadata contribution | image + text | `raw cosine fusion + category-gated TF-IDF refinement` | 1 | 0.8009 | 0.0237 | 0.0429 | 0.8009 | 0.8009 |
| combined_metadata | TF-IDF metadata contribution | image + text | `raw cosine fusion + category-gated TF-IDF refinement` | 5 | 0.7208 | 0.0936 | 0.1430 | 0.7414 | 0.6840 |
| combined_metadata | TF-IDF metadata contribution | image + text | `raw cosine fusion + category-gated TF-IDF refinement` | 10 | 0.6719 | 0.1592 | 0.2179 | 0.7076 | 0.6205 |
| combined_category | category contribution | image + text | `raw fusion + explicit-text hard pool or dynamic image category bonus` | 1 | 0.7727 | 0.0240 | 0.0430 | 0.7727 | 0.7727 |
| combined_category | category contribution | image + text | `raw fusion + explicit-text hard pool or dynamic image category bonus` | 5 | 0.7329 | 0.1020 | 0.1537 | 0.7433 | 0.7080 |
| combined_category | category contribution | image + text | `raw fusion + explicit-text hard pool or dynamic image category bonus` | 10 | 0.7126 | 0.1758 | 0.2432 | 0.7293 | 0.6793 |
| combined_full | full deployed ranking | image + text | `raw cosine fusion + category-gated TF-IDF + explicit-text hard pool or dynamic image category bonus` | 1 | 0.7771 | 0.0242 | 0.0433 | 0.7771 | 0.7771 |
| combined_full | full deployed ranking | image + text | `raw cosine fusion + category-gated TF-IDF + explicit-text hard pool or dynamic image category bonus` | 5 | 0.7355 | 0.1030 | 0.1553 | 0.7459 | 0.7115 |
| combined_full | full deployed ranking | image + text | `raw cosine fusion + category-gated TF-IDF + explicit-text hard pool or dynamic image category bonus` | 10 | 0.7182 | 0.1781 | 0.2463 | 0.7341 | 0.6853 |

## TF-IDF metadata diagnostics

| System | K | Metadata queries | Mean metadata match | Metadata hit rate |
|---|---:|---:|---:|---:|
| combined_base | 1 | 462 | 0.2326 | 0.9957 |
| combined_base | 5 | 462 | 0.1775 | 0.9931 |
| combined_base | 10 | 462 | 0.1520 | 0.9877 |
| combined_metadata | 1 | 462 | 0.2957 | 1.0000 |
| combined_metadata | 5 | 462 | 0.2123 | 0.9957 |
| combined_metadata | 10 | 462 | 0.1727 | 0.9913 |
| combined_category | 1 | 462 | 0.2141 | 0.9957 |
| combined_category | 5 | 462 | 0.1704 | 0.9905 |
| combined_category | 10 | 462 | 0.1452 | 0.9844 |
| combined_full | 1 | 462 | 0.2793 | 1.0000 |
| combined_full | 5 | 462 | 0.2016 | 0.9948 |
| combined_full | 10 | 462 | 0.1655 | 0.9887 |
