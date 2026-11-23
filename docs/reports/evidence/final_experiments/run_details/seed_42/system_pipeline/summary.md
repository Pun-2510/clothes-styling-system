# Deployed query-mode and combined-search comparison

The benchmark used 460 held-out queries against 5000 catalog products.
The query product was excluded before category prediction and ranking.
Query text omits the ground-truth category column; runtime catalog text uses the same category-bearing embeddings as the website.

## End-to-end query modes

| System | Role | Input | Formula | K | Precision | Recall | F1 | NDCG | MAP |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| image_only_deployed | deployed image query | image | `S_II + lambda_I*g(h)*w(c_i)` | 1 | 0.7130 | 0.0191 | 0.0356 | 0.7130 | 0.7130 |
| image_only_deployed | deployed image query | image | `S_II + lambda_I*g(h)*w(c_i)` | 5 | 0.6470 | 0.0822 | 0.1260 | 0.6627 | 0.6034 |
| image_only_deployed | deployed image query | image | `S_II + lambda_I*g(h)*w(c_i)` | 10 | 0.6054 | 0.1372 | 0.1885 | 0.6347 | 0.5467 |
| text_only_deployed | deployed text query | text | `CLIP text fusion + TF-IDF metadata; exact category => hard candidate pool` | 1 | 0.7717 | 0.0251 | 0.0451 | 0.7717 | 0.7717 |
| text_only_deployed | deployed text query | text | `CLIP text fusion + TF-IDF metadata; exact category => hard candidate pool` | 5 | 0.7374 | 0.1049 | 0.1574 | 0.7471 | 0.7068 |
| text_only_deployed | deployed text query | text | `CLIP text fusion + TF-IDF metadata; exact category => hard candidate pool` | 10 | 0.7120 | 0.1794 | 0.2448 | 0.7320 | 0.6753 |
| combined_full_deployed | deployed combined query | image + text | `raw cosine fusion + category-gated TF-IDF + explicit-text hard pool or dynamic image category bonus` | 1 | 0.7891 | 0.0238 | 0.0431 | 0.7891 | 0.7891 |
| combined_full_deployed | deployed combined query | image + text | `raw cosine fusion + category-gated TF-IDF + explicit-text hard pool or dynamic image category bonus` | 5 | 0.7557 | 0.1043 | 0.1577 | 0.7639 | 0.7304 |
| combined_full_deployed | deployed combined query | image + text | `raw cosine fusion + category-gated TF-IDF + explicit-text hard pool or dynamic image category bonus` | 10 | 0.7307 | 0.1791 | 0.2468 | 0.7482 | 0.6995 |

## Combined-search ablation

| System | Role | Input | Formula | K | Precision | Recall | F1 | NDCG | MAP |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| combined_base | raw fusion baseline | image + text | `w_i*S_II + (1-w_i)*(w_t*S_TI + (1-w_t)*S_TT)` | 1 | 0.7630 | 0.0220 | 0.0401 | 0.7630 | 0.7630 |
| combined_base | raw fusion baseline | image + text | `w_i*S_II + (1-w_i)*(w_t*S_TI + (1-w_t)*S_TT)` | 5 | 0.6809 | 0.0881 | 0.1346 | 0.7008 | 0.6267 |
| combined_base | raw fusion baseline | image + text | `w_i*S_II + (1-w_i)*(w_t*S_TI + (1-w_t)*S_TT)` | 10 | 0.6404 | 0.1505 | 0.2048 | 0.6736 | 0.5703 |
| combined_metadata | TF-IDF metadata contribution | image + text | `raw cosine fusion + category-gated TF-IDF refinement` | 1 | 0.8152 | 0.0247 | 0.0449 | 0.8152 | 0.8152 |
| combined_metadata | TF-IDF metadata contribution | image + text | `raw cosine fusion + category-gated TF-IDF refinement` | 5 | 0.7417 | 0.0981 | 0.1497 | 0.7600 | 0.7031 |
| combined_metadata | TF-IDF metadata contribution | image + text | `raw cosine fusion + category-gated TF-IDF refinement` | 10 | 0.6911 | 0.1641 | 0.2238 | 0.7269 | 0.6403 |
| combined_category | category contribution | image + text | `raw fusion + explicit-text hard pool or dynamic image category bonus` | 1 | 0.7870 | 0.0237 | 0.0430 | 0.7870 | 0.7870 |
| combined_category | category contribution | image + text | `raw fusion + explicit-text hard pool or dynamic image category bonus` | 5 | 0.7465 | 0.1028 | 0.1553 | 0.7569 | 0.7208 |
| combined_category | category contribution | image + text | `raw fusion + explicit-text hard pool or dynamic image category bonus` | 10 | 0.7239 | 0.1775 | 0.2444 | 0.7423 | 0.6920 |
| combined_full | full deployed ranking | image + text | `raw cosine fusion + category-gated TF-IDF + explicit-text hard pool or dynamic image category bonus` | 1 | 0.7891 | 0.0238 | 0.0431 | 0.7891 | 0.7891 |
| combined_full | full deployed ranking | image + text | `raw cosine fusion + category-gated TF-IDF + explicit-text hard pool or dynamic image category bonus` | 5 | 0.7557 | 0.1043 | 0.1577 | 0.7639 | 0.7304 |
| combined_full | full deployed ranking | image + text | `raw cosine fusion + category-gated TF-IDF + explicit-text hard pool or dynamic image category bonus` | 10 | 0.7307 | 0.1791 | 0.2468 | 0.7482 | 0.6995 |

## TF-IDF metadata diagnostics

| System | K | Metadata queries | Mean metadata match | Metadata hit rate |
|---|---:|---:|---:|---:|
| combined_base | 1 | 460 | 0.2253 | 0.9957 |
| combined_base | 5 | 460 | 0.1677 | 0.9913 |
| combined_base | 10 | 460 | 0.1411 | 0.9854 |
| combined_metadata | 1 | 460 | 0.2850 | 0.9978 |
| combined_metadata | 5 | 460 | 0.1950 | 0.9939 |
| combined_metadata | 10 | 460 | 0.1597 | 0.9898 |
| combined_category | 1 | 460 | 0.2070 | 0.9935 |
| combined_category | 5 | 460 | 0.1550 | 0.9857 |
| combined_category | 10 | 460 | 0.1317 | 0.9802 |
| combined_full | 1 | 460 | 0.2705 | 0.9957 |
| combined_full | 5 | 460 | 0.1828 | 0.9896 |
| combined_full | 10 | 460 | 0.1497 | 0.9850 |
