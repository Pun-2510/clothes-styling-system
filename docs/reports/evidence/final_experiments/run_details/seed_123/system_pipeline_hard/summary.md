# Deployed query-mode and combined-search comparison

The benchmark used 464 held-out queries against 5000 catalog products.
The query product was excluded before category prediction and ranking.
Query text omits the ground-truth category column; runtime catalog text uses the same category-bearing embeddings as the website.

## End-to-end query modes

| System | Role | Input | Formula | K | Precision | Recall | F1 | NDCG | MAP |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| image_only_deployed | deployed image query | image | `S_II within predicted-category hard pool` | 1 | 0.7026 | 0.0174 | 0.0327 | 0.7026 | 0.7026 |
| image_only_deployed | deployed image query | image | `S_II within predicted-category hard pool` | 5 | 0.7026 | 0.0852 | 0.1359 | 0.7026 | 0.7026 |
| image_only_deployed | deployed image query | image | `S_II within predicted-category hard pool` | 10 | 0.7026 | 0.1583 | 0.2289 | 0.7026 | 0.7026 |
| text_only_deployed | deployed text query | text | `CLIP text fusion + TF-IDF metadata; exact category => hard candidate pool` | 1 | 0.7392 | 0.0244 | 0.0438 | 0.7392 | 0.7392 |
| text_only_deployed | deployed text query | text | `CLIP text fusion + TF-IDF metadata; exact category => hard candidate pool` | 5 | 0.7379 | 0.1080 | 0.1617 | 0.7399 | 0.6995 |
| text_only_deployed | deployed text query | text | `CLIP text fusion + TF-IDF metadata; exact category => hard candidate pool` | 10 | 0.7108 | 0.1814 | 0.2492 | 0.7252 | 0.6693 |
| combined_full_deployed | deployed combined query | image + text | `raw cosine fusion + category-gated TF-IDF + hard pool (explicit text category else image prediction)` | 1 | 0.7392 | 0.0214 | 0.0392 | 0.7392 | 0.7392 |
| combined_full_deployed | deployed combined query | image + text | `raw cosine fusion + category-gated TF-IDF + hard pool (explicit text category else image prediction)` | 5 | 0.7392 | 0.1013 | 0.1555 | 0.7392 | 0.7392 |
| combined_full_deployed | deployed combined query | image + text | `raw cosine fusion + category-gated TF-IDF + hard pool (explicit text category else image prediction)` | 10 | 0.7392 | 0.1792 | 0.2546 | 0.7392 | 0.7392 |

## Combined-search ablation

| System | Role | Input | Formula | K | Precision | Recall | F1 | NDCG | MAP |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| combined_base | raw fusion baseline | image + text | `w_i*S_II + (1-w_i)*(w_t*S_TI + (1-w_t)*S_TT)` | 1 | 0.6789 | 0.0201 | 0.0366 | 0.6789 | 0.6789 |
| combined_base | raw fusion baseline | image + text | `w_i*S_II + (1-w_i)*(w_t*S_TI + (1-w_t)*S_TT)` | 5 | 0.6388 | 0.0858 | 0.1298 | 0.6540 | 0.5757 |
| combined_base | raw fusion baseline | image + text | `w_i*S_II + (1-w_i)*(w_t*S_TI + (1-w_t)*S_TT)` | 10 | 0.5948 | 0.1428 | 0.1942 | 0.6251 | 0.5159 |
| combined_metadata | TF-IDF metadata contribution | image + text | `raw cosine fusion + category-gated TF-IDF refinement` | 1 | 0.7565 | 0.0208 | 0.0386 | 0.7565 | 0.7565 |
| combined_metadata | TF-IDF metadata contribution | image + text | `raw cosine fusion + category-gated TF-IDF refinement` | 5 | 0.7228 | 0.0966 | 0.1472 | 0.7332 | 0.6907 |
| combined_metadata | TF-IDF metadata contribution | image + text | `raw cosine fusion + category-gated TF-IDF refinement` | 10 | 0.6841 | 0.1650 | 0.2259 | 0.7102 | 0.6420 |
| combined_category | category contribution | image + text | `raw fusion + hard pool (explicit text category else image prediction)` | 1 | 0.7392 | 0.0214 | 0.0392 | 0.7392 | 0.7392 |
| combined_category | category contribution | image + text | `raw fusion + hard pool (explicit text category else image prediction)` | 5 | 0.7392 | 0.1013 | 0.1555 | 0.7392 | 0.7392 |
| combined_category | category contribution | image + text | `raw fusion + hard pool (explicit text category else image prediction)` | 10 | 0.7392 | 0.1792 | 0.2546 | 0.7392 | 0.7392 |
| combined_full | full deployed ranking | image + text | `raw cosine fusion + category-gated TF-IDF + hard pool (explicit text category else image prediction)` | 1 | 0.7392 | 0.0214 | 0.0392 | 0.7392 | 0.7392 |
| combined_full | full deployed ranking | image + text | `raw cosine fusion + category-gated TF-IDF + hard pool (explicit text category else image prediction)` | 5 | 0.7392 | 0.1013 | 0.1555 | 0.7392 | 0.7392 |
| combined_full | full deployed ranking | image + text | `raw cosine fusion + category-gated TF-IDF + hard pool (explicit text category else image prediction)` | 10 | 0.7392 | 0.1792 | 0.2546 | 0.7392 | 0.7392 |

## TF-IDF metadata diagnostics

| System | K | Metadata queries | Mean metadata match | Metadata hit rate |
|---|---:|---:|---:|---:|
| combined_base | 1 | 464 | 0.1974 | 0.9849 |
| combined_base | 5 | 464 | 0.1601 | 0.9862 |
| combined_base | 10 | 464 | 0.1390 | 0.9815 |
| combined_metadata | 1 | 464 | 0.3011 | 0.9957 |
| combined_metadata | 5 | 464 | 0.2132 | 0.9897 |
| combined_metadata | 10 | 464 | 0.1748 | 0.9858 |
| combined_category | 1 | 464 | 0.1765 | 0.9763 |
| combined_category | 5 | 464 | 0.1473 | 0.9793 |
| combined_category | 10 | 464 | 0.1342 | 0.9732 |
| combined_full | 1 | 464 | 0.2884 | 0.9892 |
| combined_full | 5 | 464 | 0.2010 | 0.9853 |
| combined_full | 10 | 464 | 0.1662 | 0.9777 |
