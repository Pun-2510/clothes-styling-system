# Deployed query-mode and combined-search comparison

The benchmark used 464 held-out queries against 5000 catalog products.
The query product was excluded before category prediction and ranking.
Query text omits the ground-truth category column; runtime catalog text uses the same category-bearing embeddings as the website.

## End-to-end query modes

| System | Role | Input | Formula | K | Precision | Recall | F1 | NDCG | MAP |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| image_only_deployed | deployed image query | image | `S_II + lambda_I*g(h)*w(c_i)` | 1 | 0.6659 | 0.0177 | 0.0328 | 0.6659 | 0.6659 |
| image_only_deployed | deployed image query | image | `S_II + lambda_I*g(h)*w(c_i)` | 5 | 0.6272 | 0.0807 | 0.1229 | 0.6371 | 0.5729 |
| image_only_deployed | deployed image query | image | `S_II + lambda_I*g(h)*w(c_i)` | 10 | 0.5920 | 0.1372 | 0.1879 | 0.6151 | 0.5215 |
| text_only_deployed | deployed text query | text | `CLIP text fusion + TF-IDF metadata; exact category => hard candidate pool` | 1 | 0.7392 | 0.0244 | 0.0438 | 0.7392 | 0.7392 |
| text_only_deployed | deployed text query | text | `CLIP text fusion + TF-IDF metadata; exact category => hard candidate pool` | 5 | 0.7379 | 0.1080 | 0.1617 | 0.7399 | 0.6995 |
| text_only_deployed | deployed text query | text | `CLIP text fusion + TF-IDF metadata; exact category => hard candidate pool` | 10 | 0.7108 | 0.1814 | 0.2492 | 0.7252 | 0.6693 |
| combined_full_deployed | deployed combined query | image + text | `raw cosine fusion + category-gated TF-IDF + explicit-text hard pool or dynamic image category bonus` | 1 | 0.7478 | 0.0239 | 0.0430 | 0.7478 | 0.7478 |
| combined_full_deployed | deployed combined query | image + text | `raw cosine fusion + category-gated TF-IDF + explicit-text hard pool or dynamic image category bonus` | 5 | 0.7341 | 0.1039 | 0.1573 | 0.7391 | 0.7015 |
| combined_full_deployed | deployed combined query | image + text | `raw cosine fusion + category-gated TF-IDF + explicit-text hard pool or dynamic image category bonus` | 10 | 0.7151 | 0.1794 | 0.2484 | 0.7281 | 0.6756 |

## Combined-search ablation

| System | Role | Input | Formula | K | Precision | Recall | F1 | NDCG | MAP |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| combined_base | raw fusion baseline | image + text | `w_i*S_II + (1-w_i)*(w_t*S_TI + (1-w_t)*S_TT)` | 1 | 0.6789 | 0.0201 | 0.0366 | 0.6789 | 0.6789 |
| combined_base | raw fusion baseline | image + text | `w_i*S_II + (1-w_i)*(w_t*S_TI + (1-w_t)*S_TT)` | 5 | 0.6388 | 0.0858 | 0.1298 | 0.6540 | 0.5757 |
| combined_base | raw fusion baseline | image + text | `w_i*S_II + (1-w_i)*(w_t*S_TI + (1-w_t)*S_TT)` | 10 | 0.5948 | 0.1428 | 0.1942 | 0.6251 | 0.5159 |
| combined_metadata | TF-IDF metadata contribution | image + text | `raw cosine fusion + category-gated TF-IDF refinement` | 1 | 0.7586 | 0.0222 | 0.0407 | 0.7586 | 0.7586 |
| combined_metadata | TF-IDF metadata contribution | image + text | `raw cosine fusion + category-gated TF-IDF refinement` | 5 | 0.7030 | 0.0938 | 0.1431 | 0.7184 | 0.6572 |
| combined_metadata | TF-IDF metadata contribution | image + text | `raw cosine fusion + category-gated TF-IDF refinement` | 10 | 0.6614 | 0.1610 | 0.2194 | 0.6920 | 0.6027 |
| combined_category | category contribution | image + text | `raw fusion + explicit-text hard pool or dynamic image category bonus` | 1 | 0.7478 | 0.0239 | 0.0430 | 0.7478 | 0.7478 |
| combined_category | category contribution | image + text | `raw fusion + explicit-text hard pool or dynamic image category bonus` | 5 | 0.7293 | 0.1033 | 0.1562 | 0.7357 | 0.6955 |
| combined_category | category contribution | image + text | `raw fusion + explicit-text hard pool or dynamic image category bonus` | 10 | 0.7052 | 0.1767 | 0.2443 | 0.7209 | 0.6643 |
| combined_full | full deployed ranking | image + text | `raw cosine fusion + category-gated TF-IDF + explicit-text hard pool or dynamic image category bonus` | 1 | 0.7478 | 0.0239 | 0.0430 | 0.7478 | 0.7478 |
| combined_full | full deployed ranking | image + text | `raw cosine fusion + category-gated TF-IDF + explicit-text hard pool or dynamic image category bonus` | 5 | 0.7341 | 0.1039 | 0.1573 | 0.7391 | 0.7015 |
| combined_full | full deployed ranking | image + text | `raw cosine fusion + category-gated TF-IDF + explicit-text hard pool or dynamic image category bonus` | 10 | 0.7151 | 0.1794 | 0.2484 | 0.7281 | 0.6756 |

## TF-IDF metadata diagnostics

| System | K | Metadata queries | Mean metadata match | Metadata hit rate |
|---|---:|---:|---:|---:|
| combined_base | 1 | 464 | 0.1974 | 0.9849 |
| combined_base | 5 | 464 | 0.1601 | 0.9862 |
| combined_base | 10 | 464 | 0.1390 | 0.9815 |
| combined_metadata | 1 | 464 | 0.2734 | 0.9957 |
| combined_metadata | 5 | 464 | 0.1986 | 0.9897 |
| combined_metadata | 10 | 464 | 0.1660 | 0.9858 |
| combined_category | 1 | 464 | 0.1865 | 0.9741 |
| combined_category | 5 | 464 | 0.1541 | 0.9784 |
| combined_category | 10 | 464 | 0.1366 | 0.9740 |
| combined_full | 1 | 464 | 0.2645 | 0.9892 |
| combined_full | 5 | 464 | 0.1922 | 0.9849 |
| combined_full | 10 | 464 | 0.1602 | 0.9779 |
