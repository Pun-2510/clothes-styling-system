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
| combined_full_deployed | deployed combined query | image + text | `raw cosine fusion + TF-IDF metadata + explicit-text hard pool or dynamic image category bonus` | 1 | 0.7186 | 0.0259 | 0.0457 | 0.7186 | 0.7186 |
| combined_full_deployed | deployed combined query | image + text | `raw cosine fusion + TF-IDF metadata + explicit-text hard pool or dynamic image category bonus` | 5 | 0.7139 | 0.1075 | 0.1595 | 0.7176 | 0.6761 |
| combined_full_deployed | deployed combined query | image + text | `raw cosine fusion + TF-IDF metadata + explicit-text hard pool or dynamic image category bonus` | 10 | 0.6950 | 0.1827 | 0.2480 | 0.7085 | 0.6506 |

## Combined-search ablation

| System | Role | Input | Formula | K | Precision | Recall | F1 | NDCG | MAP |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| combined_base | raw fusion baseline | image + text | `w_i*S_II + (1-w_i)*(w_t*S_TI + (1-w_t)*S_TT)` | 1 | 0.7684 | 0.0221 | 0.0400 | 0.7684 | 0.7684 |
| combined_base | raw fusion baseline | image + text | `w_i*S_II + (1-w_i)*(w_t*S_TI + (1-w_t)*S_TT)` | 5 | 0.6788 | 0.0872 | 0.1327 | 0.7003 | 0.6285 |
| combined_base | raw fusion baseline | image + text | `w_i*S_II + (1-w_i)*(w_t*S_TI + (1-w_t)*S_TT)` | 10 | 0.6305 | 0.1468 | 0.2005 | 0.6666 | 0.5637 |
| combined_metadata | TF-IDF metadata contribution | image + text | `raw cosine fusion + TF-IDF metadata refinement` | 1 | 0.6342 | 0.0211 | 0.0377 | 0.6342 | 0.6342 |
| combined_metadata | TF-IDF metadata contribution | image + text | `raw cosine fusion + TF-IDF metadata refinement` | 5 | 0.6052 | 0.0873 | 0.1290 | 0.6160 | 0.5340 |
| combined_metadata | TF-IDF metadata contribution | image + text | `raw cosine fusion + TF-IDF metadata refinement` | 10 | 0.5751 | 0.1479 | 0.1939 | 0.6006 | 0.4873 |
| combined_category | category contribution | image + text | `raw fusion + explicit-text hard pool or dynamic image category bonus` | 1 | 0.7814 | 0.0242 | 0.0434 | 0.7814 | 0.7814 |
| combined_category | category contribution | image + text | `raw fusion + explicit-text hard pool or dynamic image category bonus` | 5 | 0.7368 | 0.1028 | 0.1549 | 0.7478 | 0.7099 |
| combined_category | category contribution | image + text | `raw fusion + explicit-text hard pool or dynamic image category bonus` | 10 | 0.7119 | 0.1765 | 0.2438 | 0.7305 | 0.6772 |
| combined_full | full deployed ranking | image + text | `raw cosine fusion + TF-IDF metadata + explicit-text hard pool or dynamic image category bonus` | 1 | 0.7186 | 0.0259 | 0.0457 | 0.7186 | 0.7186 |
| combined_full | full deployed ranking | image + text | `raw cosine fusion + TF-IDF metadata + explicit-text hard pool or dynamic image category bonus` | 5 | 0.7139 | 0.1075 | 0.1595 | 0.7176 | 0.6761 |
| combined_full | full deployed ranking | image + text | `raw cosine fusion + TF-IDF metadata + explicit-text hard pool or dynamic image category bonus` | 10 | 0.6950 | 0.1827 | 0.2480 | 0.7085 | 0.6506 |

## TF-IDF metadata diagnostics

| System | K | Metadata queries | Mean metadata match | Metadata hit rate |
|---|---:|---:|---:|---:|
| combined_base | 1 | 462 | 0.2351 | 0.9978 |
| combined_base | 5 | 462 | 0.1817 | 0.9952 |
| combined_base | 10 | 462 | 0.1548 | 0.9913 |
| combined_metadata | 1 | 462 | 0.4478 | 1.0000 |
| combined_metadata | 5 | 462 | 0.3281 | 1.0000 |
| combined_metadata | 10 | 462 | 0.2712 | 0.9998 |
| combined_category | 1 | 462 | 0.2164 | 0.9978 |
| combined_category | 5 | 462 | 0.1723 | 0.9926 |
| combined_category | 10 | 462 | 0.1471 | 0.9870 |
| combined_full | 1 | 462 | 0.3942 | 1.0000 |
| combined_full | 5 | 462 | 0.2814 | 0.9983 |
| combined_full | 10 | 462 | 0.2304 | 0.9960 |
