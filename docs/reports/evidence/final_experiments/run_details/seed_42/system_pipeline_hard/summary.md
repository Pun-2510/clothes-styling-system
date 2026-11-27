# Deployed query-mode and combined-search comparison

The benchmark used 460 held-out queries against 5000 catalog products.
The query product was excluded before category prediction and ranking.
Query text omits the ground-truth category column; runtime catalog text uses the same category-bearing embeddings as the website.

## End-to-end query modes

| System | Role | Input | Formula | K | Precision | Recall | F1 | NDCG | MAP |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| image_only_deployed | deployed image query | image | `S_II within predicted-category hard pool` | 1 | 0.7065 | 0.0184 | 0.0344 | 0.7065 | 0.7065 |
| image_only_deployed | deployed image query | image | `S_II within predicted-category hard pool` | 5 | 0.7065 | 0.0901 | 0.1412 | 0.7065 | 0.7065 |
| image_only_deployed | deployed image query | image | `S_II within predicted-category hard pool` | 10 | 0.7065 | 0.1647 | 0.2347 | 0.7065 | 0.7065 |
| text_only_deployed | deployed text query | text | `CLIP text fusion + TF-IDF metadata; exact category => hard candidate pool` | 1 | 0.7717 | 0.0251 | 0.0451 | 0.7717 | 0.7717 |
| text_only_deployed | deployed text query | text | `CLIP text fusion + TF-IDF metadata; exact category => hard candidate pool` | 5 | 0.7374 | 0.1049 | 0.1574 | 0.7471 | 0.7068 |
| text_only_deployed | deployed text query | text | `CLIP text fusion + TF-IDF metadata; exact category => hard candidate pool` | 10 | 0.7120 | 0.1794 | 0.2448 | 0.7320 | 0.6753 |
| combined_full_deployed | deployed combined query | image + text | `raw cosine fusion + category-gated TF-IDF + hard pool (explicit text category else image prediction)` | 1 | 0.7587 | 0.0213 | 0.0394 | 0.7587 | 0.7587 |
| combined_full_deployed | deployed combined query | image + text | `raw cosine fusion + category-gated TF-IDF + hard pool (explicit text category else image prediction)` | 5 | 0.7587 | 0.1025 | 0.1582 | 0.7587 | 0.7587 |
| combined_full_deployed | deployed combined query | image + text | `raw cosine fusion + category-gated TF-IDF + hard pool (explicit text category else image prediction)` | 10 | 0.7587 | 0.1830 | 0.2602 | 0.7587 | 0.7587 |

## Combined-search ablation

| System | Role | Input | Formula | K | Precision | Recall | F1 | NDCG | MAP |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| combined_base | raw fusion baseline | image + text | `w_i*S_II + (1-w_i)*(w_t*S_TI + (1-w_t)*S_TT)` | 1 | 0.7630 | 0.0220 | 0.0401 | 0.7630 | 0.7630 |
| combined_base | raw fusion baseline | image + text | `w_i*S_II + (1-w_i)*(w_t*S_TI + (1-w_t)*S_TT)` | 5 | 0.6809 | 0.0881 | 0.1346 | 0.7008 | 0.6267 |
| combined_base | raw fusion baseline | image + text | `w_i*S_II + (1-w_i)*(w_t*S_TI + (1-w_t)*S_TT)` | 10 | 0.6404 | 0.1505 | 0.2048 | 0.6736 | 0.5703 |
| combined_metadata | TF-IDF metadata contribution | image + text | `raw cosine fusion + category-gated TF-IDF refinement` | 1 | 0.7978 | 0.0229 | 0.0423 | 0.7978 | 0.7978 |
| combined_metadata | TF-IDF metadata contribution | image + text | `raw cosine fusion + category-gated TF-IDF refinement` | 5 | 0.7600 | 0.0993 | 0.1532 | 0.7709 | 0.7302 |
| combined_metadata | TF-IDF metadata contribution | image + text | `raw cosine fusion + category-gated TF-IDF refinement` | 10 | 0.7087 | 0.1695 | 0.2312 | 0.7401 | 0.6703 |
| combined_category | category contribution | image + text | `raw fusion + hard pool (explicit text category else image prediction)` | 1 | 0.7587 | 0.0213 | 0.0394 | 0.7587 | 0.7587 |
| combined_category | category contribution | image + text | `raw fusion + hard pool (explicit text category else image prediction)` | 5 | 0.7587 | 0.1025 | 0.1582 | 0.7587 | 0.7587 |
| combined_category | category contribution | image + text | `raw fusion + hard pool (explicit text category else image prediction)` | 10 | 0.7587 | 0.1830 | 0.2602 | 0.7587 | 0.7587 |
| combined_full | full deployed ranking | image + text | `raw cosine fusion + category-gated TF-IDF + hard pool (explicit text category else image prediction)` | 1 | 0.7587 | 0.0213 | 0.0394 | 0.7587 | 0.7587 |
| combined_full | full deployed ranking | image + text | `raw cosine fusion + category-gated TF-IDF + hard pool (explicit text category else image prediction)` | 5 | 0.7587 | 0.1025 | 0.1582 | 0.7587 | 0.7587 |
| combined_full | full deployed ranking | image + text | `raw cosine fusion + category-gated TF-IDF + hard pool (explicit text category else image prediction)` | 10 | 0.7587 | 0.1830 | 0.2602 | 0.7587 | 0.7587 |

## TF-IDF metadata diagnostics

| System | K | Metadata queries | Mean metadata match | Metadata hit rate |
|---|---:|---:|---:|---:|
| combined_base | 1 | 460 | 0.2253 | 0.9957 |
| combined_base | 5 | 460 | 0.1677 | 0.9913 |
| combined_base | 10 | 460 | 0.1411 | 0.9854 |
| combined_metadata | 1 | 460 | 0.3107 | 0.9978 |
| combined_metadata | 5 | 460 | 0.2132 | 0.9943 |
| combined_metadata | 10 | 460 | 0.1697 | 0.9902 |
| combined_category | 1 | 460 | 0.1940 | 0.9935 |
| combined_category | 5 | 460 | 0.1508 | 0.9848 |
| combined_category | 10 | 460 | 0.1297 | 0.9783 |
| combined_full | 1 | 460 | 0.2881 | 0.9957 |
| combined_full | 5 | 460 | 0.1955 | 0.9896 |
| combined_full | 10 | 460 | 0.1573 | 0.9846 |
