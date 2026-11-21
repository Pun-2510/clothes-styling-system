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
| combined_full_deployed | deployed combined query | image + text | `raw cosine fusion + TF-IDF metadata + explicit-text hard pool or dynamic image category bonus` | 1 | 0.7739 | 0.0253 | 0.0454 | 0.7739 | 0.7739 |
| combined_full_deployed | deployed combined query | image + text | `raw cosine fusion + TF-IDF metadata + explicit-text hard pool or dynamic image category bonus` | 5 | 0.7252 | 0.1039 | 0.1566 | 0.7363 | 0.6910 |
| combined_full_deployed | deployed combined query | image + text | `raw cosine fusion + TF-IDF metadata + explicit-text hard pool or dynamic image category bonus` | 10 | 0.7070 | 0.1807 | 0.2461 | 0.7257 | 0.6636 |

## Combined-search ablation

| System | Role | Input | Formula | K | Precision | Recall | F1 | NDCG | MAP |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| combined_base | raw fusion baseline | image + text | `w_i*S_II + (1-w_i)*(w_t*S_TI + (1-w_t)*S_TT)` | 1 | 0.7587 | 0.0217 | 0.0397 | 0.7587 | 0.7587 |
| combined_base | raw fusion baseline | image + text | `w_i*S_II + (1-w_i)*(w_t*S_TI + (1-w_t)*S_TT)` | 5 | 0.6922 | 0.0898 | 0.1375 | 0.7086 | 0.6361 |
| combined_base | raw fusion baseline | image + text | `w_i*S_II + (1-w_i)*(w_t*S_TI + (1-w_t)*S_TT)` | 10 | 0.6409 | 0.1514 | 0.2066 | 0.6749 | 0.5711 |
| combined_metadata | TF-IDF metadata contribution | image + text | `raw cosine fusion + TF-IDF metadata refinement` | 1 | 0.6783 | 0.0230 | 0.0411 | 0.6783 | 0.6783 |
| combined_metadata | TF-IDF metadata contribution | image + text | `raw cosine fusion + TF-IDF metadata refinement` | 5 | 0.6074 | 0.0868 | 0.1298 | 0.6247 | 0.5390 |
| combined_metadata | TF-IDF metadata contribution | image + text | `raw cosine fusion + TF-IDF metadata refinement` | 10 | 0.5730 | 0.1474 | 0.1940 | 0.6045 | 0.4851 |
| combined_category | category contribution | image + text | `raw fusion + explicit-text hard pool or dynamic image category bonus` | 1 | 0.7870 | 0.0235 | 0.0428 | 0.7870 | 0.7870 |
| combined_category | category contribution | image + text | `raw fusion + explicit-text hard pool or dynamic image category bonus` | 5 | 0.7465 | 0.1026 | 0.1552 | 0.7558 | 0.7187 |
| combined_category | category contribution | image + text | `raw fusion + explicit-text hard pool or dynamic image category bonus` | 10 | 0.7200 | 0.1766 | 0.2434 | 0.7390 | 0.6867 |
| combined_full | full deployed ranking | image + text | `raw cosine fusion + TF-IDF metadata + explicit-text hard pool or dynamic image category bonus` | 1 | 0.7739 | 0.0253 | 0.0454 | 0.7739 | 0.7739 |
| combined_full | full deployed ranking | image + text | `raw cosine fusion + TF-IDF metadata + explicit-text hard pool or dynamic image category bonus` | 5 | 0.7252 | 0.1039 | 0.1566 | 0.7363 | 0.6910 |
| combined_full | full deployed ranking | image + text | `raw cosine fusion + TF-IDF metadata + explicit-text hard pool or dynamic image category bonus` | 10 | 0.7070 | 0.1807 | 0.2461 | 0.7257 | 0.6636 |

## TF-IDF metadata diagnostics

| System | K | Metadata queries | Mean metadata match | Metadata hit rate |
|---|---:|---:|---:|---:|
| combined_base | 1 | 460 | 0.2343 | 0.9935 |
| combined_base | 5 | 460 | 0.1745 | 0.9922 |
| combined_base | 10 | 460 | 0.1456 | 0.9898 |
| combined_metadata | 1 | 460 | 0.4349 | 1.0000 |
| combined_metadata | 5 | 460 | 0.3130 | 1.0000 |
| combined_metadata | 10 | 460 | 0.2524 | 0.9993 |
| combined_category | 1 | 460 | 0.2164 | 0.9913 |
| combined_category | 5 | 460 | 0.1599 | 0.9878 |
| combined_category | 10 | 460 | 0.1347 | 0.9826 |
| combined_full | 1 | 460 | 0.3825 | 0.9978 |
| combined_full | 5 | 460 | 0.2659 | 0.9970 |
| combined_full | 10 | 460 | 0.2136 | 0.9943 |
