# Deployed query-mode and combined-search comparison

The benchmark used 462 held-out queries against 5000 catalog products.
The query product was excluded before category prediction and ranking.
Query text omits the ground-truth category column; runtime catalog text uses the same category-bearing embeddings as the website.

## End-to-end query modes

| System | Role | Input | Formula | K | Precision | Recall | F1 | NDCG | MAP |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| image_only_deployed | deployed image query | image | `S_II within predicted-category hard pool` | 1 | 0.6818 | 0.0168 | 0.0316 | 0.6818 | 0.6818 |
| image_only_deployed | deployed image query | image | `S_II within predicted-category hard pool` | 5 | 0.6818 | 0.0821 | 0.1313 | 0.6818 | 0.6818 |
| image_only_deployed | deployed image query | image | `S_II within predicted-category hard pool` | 10 | 0.6818 | 0.1527 | 0.2216 | 0.6818 | 0.6818 |
| text_only_deployed | deployed text query | text | `CLIP text fusion + TF-IDF metadata; exact category => hard candidate pool` | 1 | 0.7576 | 0.0269 | 0.0477 | 0.7576 | 0.7576 |
| text_only_deployed | deployed text query | text | `CLIP text fusion + TF-IDF metadata; exact category => hard candidate pool` | 5 | 0.7437 | 0.1085 | 0.1624 | 0.7510 | 0.7127 |
| text_only_deployed | deployed text query | text | `CLIP text fusion + TF-IDF metadata; exact category => hard candidate pool` | 10 | 0.7154 | 0.1839 | 0.2514 | 0.7340 | 0.6797 |
| combined_full_deployed | deployed combined query | image + text | `raw cosine fusion + category-gated TF-IDF + hard pool (explicit text category else image prediction)` | 1 | 0.7338 | 0.0214 | 0.0389 | 0.7338 | 0.7338 |
| combined_full_deployed | deployed combined query | image + text | `raw cosine fusion + category-gated TF-IDF + hard pool (explicit text category else image prediction)` | 5 | 0.7338 | 0.0982 | 0.1520 | 0.7338 | 0.7338 |
| combined_full_deployed | deployed combined query | image + text | `raw cosine fusion + category-gated TF-IDF + hard pool (explicit text category else image prediction)` | 10 | 0.7338 | 0.1748 | 0.2506 | 0.7338 | 0.7338 |

## Combined-search ablation

| System | Role | Input | Formula | K | Precision | Recall | F1 | NDCG | MAP |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| combined_base | raw fusion baseline | image + text | `w_i*S_II + (1-w_i)*(w_t*S_TI + (1-w_t)*S_TT)` | 1 | 0.7359 | 0.0212 | 0.0383 | 0.7359 | 0.7359 |
| combined_base | raw fusion baseline | image + text | `w_i*S_II + (1-w_i)*(w_t*S_TI + (1-w_t)*S_TT)` | 5 | 0.6706 | 0.0859 | 0.1306 | 0.6883 | 0.6163 |
| combined_base | raw fusion baseline | image + text | `w_i*S_II + (1-w_i)*(w_t*S_TI + (1-w_t)*S_TT)` | 10 | 0.6186 | 0.1434 | 0.1960 | 0.6530 | 0.5511 |
| combined_metadata | TF-IDF metadata contribution | image + text | `raw cosine fusion + category-gated TF-IDF refinement` | 1 | 0.7814 | 0.0231 | 0.0416 | 0.7814 | 0.7814 |
| combined_metadata | TF-IDF metadata contribution | image + text | `raw cosine fusion + category-gated TF-IDF refinement` | 5 | 0.7342 | 0.0929 | 0.1439 | 0.7483 | 0.7025 |
| combined_metadata | TF-IDF metadata contribution | image + text | `raw cosine fusion + category-gated TF-IDF refinement` | 10 | 0.6918 | 0.1625 | 0.2229 | 0.7217 | 0.6497 |
| combined_category | category contribution | image + text | `raw fusion + hard pool (explicit text category else image prediction)` | 1 | 0.7338 | 0.0214 | 0.0389 | 0.7338 | 0.7338 |
| combined_category | category contribution | image + text | `raw fusion + hard pool (explicit text category else image prediction)` | 5 | 0.7338 | 0.0982 | 0.1520 | 0.7338 | 0.7338 |
| combined_category | category contribution | image + text | `raw fusion + hard pool (explicit text category else image prediction)` | 10 | 0.7338 | 0.1748 | 0.2506 | 0.7338 | 0.7338 |
| combined_full | full deployed ranking | image + text | `raw cosine fusion + category-gated TF-IDF + hard pool (explicit text category else image prediction)` | 1 | 0.7338 | 0.0214 | 0.0389 | 0.7338 | 0.7338 |
| combined_full | full deployed ranking | image + text | `raw cosine fusion + category-gated TF-IDF + hard pool (explicit text category else image prediction)` | 5 | 0.7338 | 0.0982 | 0.1520 | 0.7338 | 0.7338 |
| combined_full | full deployed ranking | image + text | `raw cosine fusion + category-gated TF-IDF + hard pool (explicit text category else image prediction)` | 10 | 0.7338 | 0.1748 | 0.2506 | 0.7338 | 0.7338 |

## TF-IDF metadata diagnostics

| System | K | Metadata queries | Mean metadata match | Metadata hit rate |
|---|---:|---:|---:|---:|
| combined_base | 1 | 462 | 0.2332 | 0.9957 |
| combined_base | 5 | 462 | 0.1783 | 0.9935 |
| combined_base | 10 | 462 | 0.1523 | 0.9879 |
| combined_metadata | 1 | 462 | 0.3183 | 1.0000 |
| combined_metadata | 5 | 462 | 0.2223 | 0.9952 |
| combined_metadata | 10 | 462 | 0.1808 | 0.9926 |
| combined_category | 1 | 462 | 0.2049 | 0.9957 |
| combined_category | 5 | 462 | 0.1612 | 0.9870 |
| combined_category | 10 | 462 | 0.1396 | 0.9802 |
| combined_full | 1 | 462 | 0.2930 | 1.0000 |
| combined_full | 5 | 462 | 0.2042 | 0.9939 |
| combined_full | 10 | 462 | 0.1676 | 0.9874 |
