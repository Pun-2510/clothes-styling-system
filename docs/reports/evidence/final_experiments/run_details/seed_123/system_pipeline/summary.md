# Deployed query-mode and combined-search comparison

The benchmark used 464 held-out queries against 5000 catalog products.
The query product was excluded before category prediction and ranking.
Query text omits the ground-truth category column; runtime catalog text uses the same category-bearing embeddings as the website.

## End-to-end query modes

| System | Role | Input | Formula | K | Precision | Recall | F1 | NDCG | MAP |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| image_only_deployed | deployed image query | image | `S_II + lambda_I*g(h)*w(c_i)` | 1 | 0.6703 | 0.0178 | 0.0331 | 0.6703 | 0.6703 |
| image_only_deployed | deployed image query | image | `S_II + lambda_I*g(h)*w(c_i)` | 5 | 0.6284 | 0.0812 | 0.1234 | 0.6394 | 0.5748 |
| image_only_deployed | deployed image query | image | `S_II + lambda_I*g(h)*w(c_i)` | 10 | 0.5933 | 0.1381 | 0.1891 | 0.6170 | 0.5234 |
| text_only_deployed | deployed text query | text | `w_t*S_TI + (1-w_t)*S_TT; exact category => hard pool` | 1 | 0.7651 | 0.0230 | 0.0417 | 0.7651 | 0.7651 |
| text_only_deployed | deployed text query | text | `w_t*S_TI + (1-w_t)*S_TT; exact category => hard pool` | 5 | 0.7203 | 0.1000 | 0.1514 | 0.7322 | 0.6928 |
| text_only_deployed | deployed text query | text | `w_t*S_TI + (1-w_t)*S_TT; exact category => hard pool` | 10 | 0.6974 | 0.1705 | 0.2367 | 0.7156 | 0.6585 |
| combined_full_deployed | deployed combined query | image + text | `raw fusion + attribute + category policy` | 1 | 0.7328 | 0.0221 | 0.0403 | 0.7328 | 0.7328 |
| combined_full_deployed | deployed combined query | image + text | `raw fusion + attribute + category policy` | 5 | 0.7371 | 0.1032 | 0.1567 | 0.7380 | 0.7157 |
| combined_full_deployed | deployed combined query | image + text | `raw fusion + attribute + category policy` | 10 | 0.7209 | 0.1793 | 0.2485 | 0.7302 | 0.6969 |

## Combined-search ablation

| System | Role | Input | Formula | K | Precision | Recall | F1 | NDCG | MAP |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| combined_base | raw fusion baseline | image + text | `w_i*S_II + (1-w_i)*(w_t*S_TI + (1-w_t)*S_TT)` | 1 | 0.7112 | 0.0214 | 0.0390 | 0.7112 | 0.7112 |
| combined_base | raw fusion baseline | image + text | `w_i*S_II + (1-w_i)*(w_t*S_TI + (1-w_t)*S_TT)` | 5 | 0.6599 | 0.0888 | 0.1345 | 0.6742 | 0.5988 |
| combined_base | raw fusion baseline | image + text | `w_i*S_II + (1-w_i)*(w_t*S_TI + (1-w_t)*S_TT)` | 10 | 0.6116 | 0.1496 | 0.2022 | 0.6433 | 0.5371 |
| combined_attribute | attribute contribution | image + text | `raw fusion + attribute adjustment` | 1 | 0.6746 | 0.0205 | 0.0374 | 0.6746 | 0.6746 |
| combined_attribute | attribute contribution | image + text | `raw fusion + attribute adjustment` | 5 | 0.6358 | 0.0856 | 0.1289 | 0.6495 | 0.5723 |
| combined_attribute | attribute contribution | image + text | `raw fusion + attribute adjustment` | 10 | 0.5851 | 0.1423 | 0.1914 | 0.6171 | 0.5076 |
| combined_category | category contribution | image + text | `raw fusion + category policy` | 1 | 0.7414 | 0.0227 | 0.0412 | 0.7414 | 0.7414 |
| combined_category | category contribution | image + text | `raw fusion + category policy` | 5 | 0.7409 | 0.1046 | 0.1581 | 0.7426 | 0.7212 |
| combined_category | category contribution | image + text | `raw fusion + category policy` | 10 | 0.7280 | 0.1817 | 0.2515 | 0.7370 | 0.7063 |
| combined_full | full deployed ranking | image + text | `raw fusion + attribute + category policy` | 1 | 0.7328 | 0.0221 | 0.0403 | 0.7328 | 0.7328 |
| combined_full | full deployed ranking | image + text | `raw fusion + attribute + category policy` | 5 | 0.7371 | 0.1032 | 0.1567 | 0.7380 | 0.7157 |
| combined_full | full deployed ranking | image + text | `raw fusion + attribute + category policy` | 10 | 0.7209 | 0.1793 | 0.2485 | 0.7302 | 0.6969 |

## Attribute diagnostics

| System | K | Attribute queries | Mean attribute match | All attributes match |
|---|---:|---:|---:|---:|
| combined_base | 1 | 404 | 0.6356 | 0.4505 |
| combined_base | 5 | 404 | 0.6008 | 0.4183 |
| combined_base | 10 | 404 | 0.5600 | 0.3782 |
| combined_attribute | 1 | 404 | 0.7757 | 0.5891 |
| combined_attribute | 5 | 404 | 0.7127 | 0.5257 |
| combined_attribute | 10 | 404 | 0.6744 | 0.4814 |
| combined_category | 1 | 404 | 0.5968 | 0.4084 |
| combined_category | 5 | 404 | 0.5522 | 0.3777 |
| combined_category | 10 | 404 | 0.5024 | 0.3296 |
| combined_full | 1 | 404 | 0.7422 | 0.5520 |
| combined_full | 5 | 404 | 0.6546 | 0.4718 |
| combined_full | 10 | 404 | 0.5925 | 0.4071 |
