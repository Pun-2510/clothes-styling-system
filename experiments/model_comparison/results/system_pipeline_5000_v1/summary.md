# Deployed query-mode and combined-search comparison

The benchmark used 473 held-out queries against 5000 catalog products.
The query product was excluded before category prediction and ranking.
Query text omits the ground-truth category column; runtime catalog text uses the same category-bearing embeddings as the website.

## End-to-end query modes

| System | Role | Input | Formula | K | Precision | Recall | F1 |
|---|---|---|---|---:|---:|---:|---:|
| image_only_deployed | deployed image query | image | `S_II + lambda_I*g(h)*I[c_i=c_hat]` | 1 | 0.7548 | 0.0200 | 0.0357 |
| image_only_deployed | deployed image query | image | `S_II + lambda_I*g(h)*I[c_i=c_hat]` | 5 | 0.7311 | 0.0868 | 0.1227 |
| image_only_deployed | deployed image query | image | `S_II + lambda_I*g(h)*I[c_i=c_hat]` | 10 | 0.7074 | 0.1433 | 0.1795 |
| text_only_deployed | deployed text query | text | `w_t*S_TI + (1-w_t)*S_TT` | 1 | 0.8309 | 0.0233 | 0.0416 |
| text_only_deployed | deployed text query | text | `w_t*S_TI + (1-w_t)*S_TT` | 5 | 0.7378 | 0.0851 | 0.1219 |
| text_only_deployed | deployed text query | text | `w_t*S_TI + (1-w_t)*S_TT` | 10 | 0.6943 | 0.1393 | 0.1748 |
| combined_full_deployed | deployed combined query | image + text | `raw fusion + attribute + category policy` | 1 | 0.6744 | 0.0228 | 0.0404 |
| combined_full_deployed | deployed combined query | image + text | `raw fusion + attribute + category policy` | 5 | 0.6672 | 0.1032 | 0.1445 |
| combined_full_deployed | deployed combined query | image + text | `raw fusion + attribute + category policy` | 10 | 0.6613 | 0.1721 | 0.2175 |

## Combined-search ablation

| System | Role | Input | Formula | K | Precision | Recall | F1 |
|---|---|---|---|---:|---:|---:|---:|
| combined_base | raw fusion baseline | image + text | `w_i*S_II + (1-w_i)*(w_t*S_TI + (1-w_t)*S_TT)` | 1 | 0.8541 | 0.0255 | 0.0448 |
| combined_base | raw fusion baseline | image + text | `w_i*S_II + (1-w_i)*(w_t*S_TI + (1-w_t)*S_TT)` | 5 | 0.7501 | 0.0921 | 0.1292 |
| combined_base | raw fusion baseline | image + text | `w_i*S_II + (1-w_i)*(w_t*S_TI + (1-w_t)*S_TT)` | 10 | 0.7148 | 0.1507 | 0.1866 |
| combined_attribute | attribute contribution | image + text | `raw fusion + attribute adjustment` | 1 | 0.7484 | 0.0189 | 0.0339 |
| combined_attribute | attribute contribution | image + text | `raw fusion + attribute adjustment` | 5 | 0.6427 | 0.0628 | 0.0933 |
| combined_attribute | attribute contribution | image + text | `raw fusion + attribute adjustment` | 10 | 0.5934 | 0.1008 | 0.1323 |
| combined_category | category contribution | image + text | `raw fusion + category policy` | 1 | 0.6786 | 0.0229 | 0.0406 |
| combined_category | category contribution | image + text | `raw fusion + category policy` | 5 | 0.6719 | 0.1045 | 0.1464 |
| combined_category | category contribution | image + text | `raw fusion + category policy` | 10 | 0.6674 | 0.1755 | 0.2218 |
| combined_full | full deployed ranking | image + text | `raw fusion + attribute + category policy` | 1 | 0.6744 | 0.0228 | 0.0404 |
| combined_full | full deployed ranking | image + text | `raw fusion + attribute + category policy` | 5 | 0.6672 | 0.1032 | 0.1445 |
| combined_full | full deployed ranking | image + text | `raw fusion + attribute + category policy` | 10 | 0.6613 | 0.1721 | 0.2175 |

## Attribute diagnostics

| System | K | Attribute queries | Mean attribute match | All attributes match |
|---|---:|---:|---:|---:|
| combined_base | 1 | 437 | 0.7251 | 0.5332 |
| combined_base | 5 | 437 | 0.6562 | 0.4545 |
| combined_base | 10 | 437 | 0.6273 | 0.4227 |
| combined_attribute | 1 | 437 | 0.9576 | 0.8764 |
| combined_attribute | 5 | 437 | 0.9250 | 0.8082 |
| combined_attribute | 10 | 437 | 0.9068 | 0.7760 |
| combined_category | 1 | 437 | 0.6611 | 0.4760 |
| combined_category | 5 | 437 | 0.5614 | 0.3767 |
| combined_category | 10 | 437 | 0.5163 | 0.3388 |
| combined_full | 1 | 437 | 0.8575 | 0.7185 |
| combined_full | 5 | 437 | 0.7497 | 0.5799 |
| combined_full | 10 | 437 | 0.6750 | 0.5038 |
