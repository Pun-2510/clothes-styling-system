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
| combined_full_deployed | deployed combined query | image + text | `raw cosine fusion + TF-IDF metadata + explicit-text hard pool or dynamic image category bonus` | 1 | 0.7198 | 0.0249 | 0.0447 | 0.7198 | 0.7198 |
| combined_full_deployed | deployed combined query | image + text | `raw cosine fusion + TF-IDF metadata + explicit-text hard pool or dynamic image category bonus` | 5 | 0.7134 | 0.1073 | 0.1601 | 0.7160 | 0.6751 |
| combined_full_deployed | deployed combined query | image + text | `raw cosine fusion + TF-IDF metadata + explicit-text hard pool or dynamic image category bonus` | 10 | 0.6940 | 0.1813 | 0.2474 | 0.7071 | 0.6500 |

## Combined-search ablation

| System | Role | Input | Formula | K | Precision | Recall | F1 | NDCG | MAP |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| combined_base | raw fusion baseline | image + text | `w_i*S_II + (1-w_i)*(w_t*S_TI + (1-w_t)*S_TT)` | 1 | 0.7177 | 0.0206 | 0.0378 | 0.7177 | 0.7177 |
| combined_base | raw fusion baseline | image + text | `w_i*S_II + (1-w_i)*(w_t*S_TI + (1-w_t)*S_TT)` | 5 | 0.6547 | 0.0868 | 0.1318 | 0.6745 | 0.6011 |
| combined_base | raw fusion baseline | image + text | `w_i*S_II + (1-w_i)*(w_t*S_TI + (1-w_t)*S_TT)` | 10 | 0.6114 | 0.1450 | 0.1973 | 0.6445 | 0.5371 |
| combined_metadata | TF-IDF metadata contribution | image + text | `raw cosine fusion + TF-IDF metadata refinement` | 1 | 0.6293 | 0.0217 | 0.0390 | 0.6293 | 0.6293 |
| combined_metadata | TF-IDF metadata contribution | image + text | `raw cosine fusion + TF-IDF metadata refinement` | 5 | 0.6000 | 0.0877 | 0.1308 | 0.6086 | 0.5340 |
| combined_metadata | TF-IDF metadata contribution | image + text | `raw cosine fusion + TF-IDF metadata refinement` | 10 | 0.5636 | 0.1445 | 0.1927 | 0.5887 | 0.4817 |
| combined_category | category contribution | image + text | `raw fusion + explicit-text hard pool or dynamic image category bonus` | 1 | 0.7737 | 0.0240 | 0.0433 | 0.7737 | 0.7737 |
| combined_category | category contribution | image + text | `raw fusion + explicit-text hard pool or dynamic image category bonus` | 5 | 0.7332 | 0.1040 | 0.1567 | 0.7433 | 0.7052 |
| combined_category | category contribution | image + text | `raw fusion + explicit-text hard pool or dynamic image category bonus` | 10 | 0.7084 | 0.1749 | 0.2426 | 0.7261 | 0.6687 |
| combined_full | full deployed ranking | image + text | `raw cosine fusion + TF-IDF metadata + explicit-text hard pool or dynamic image category bonus` | 1 | 0.7198 | 0.0249 | 0.0447 | 0.7198 | 0.7198 |
| combined_full | full deployed ranking | image + text | `raw cosine fusion + TF-IDF metadata + explicit-text hard pool or dynamic image category bonus` | 5 | 0.7134 | 0.1073 | 0.1601 | 0.7160 | 0.6751 |
| combined_full | full deployed ranking | image + text | `raw cosine fusion + TF-IDF metadata + explicit-text hard pool or dynamic image category bonus` | 10 | 0.6940 | 0.1813 | 0.2474 | 0.7071 | 0.6500 |

## TF-IDF metadata diagnostics

| System | K | Metadata queries | Mean metadata match | Metadata hit rate |
|---|---:|---:|---:|---:|
| combined_base | 1 | 464 | 0.2095 | 0.9935 |
| combined_base | 5 | 464 | 0.1669 | 0.9884 |
| combined_base | 10 | 464 | 0.1431 | 0.9877 |
| combined_metadata | 1 | 464 | 0.4397 | 1.0000 |
| combined_metadata | 5 | 464 | 0.3242 | 0.9996 |
| combined_metadata | 10 | 464 | 0.2668 | 0.9998 |
| combined_category | 1 | 464 | 0.1951 | 0.9871 |
| combined_category | 5 | 464 | 0.1587 | 0.9806 |
| combined_category | 10 | 464 | 0.1387 | 0.9794 |
| combined_full | 1 | 464 | 0.3905 | 0.9978 |
| combined_full | 5 | 464 | 0.2834 | 0.9957 |
| combined_full | 10 | 464 | 0.2316 | 0.9924 |
