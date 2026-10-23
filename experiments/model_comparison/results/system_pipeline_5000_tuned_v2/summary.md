# Deployed query-mode and combined-search comparison

The benchmark used 473 held-out queries against 5000 catalog products.
The query product was excluded before category prediction and ranking.
Query text omits the ground-truth category column; runtime catalog text uses the same category-bearing embeddings as the website.

## End-to-end query modes

| System | Role | Input | Formula | K | Precision | Recall | F1 |
|---|---|---|---|---:|---:|---:|---:|
| image_only_deployed | deployed image query | image | `S_II + lambda_I*g(h)*I[c_i=c_hat]` | 1 | 0.7526 | 0.0177 | 0.0327 |
| image_only_deployed | deployed image query | image | `S_II + lambda_I*g(h)*I[c_i=c_hat]` | 5 | 0.7480 | 0.0863 | 0.1258 |
| image_only_deployed | deployed image query | image | `S_II + lambda_I*g(h)*I[c_i=c_hat]` | 10 | 0.7347 | 0.1522 | 0.1929 |
| text_only_deployed | deployed text query | text | `w_t*S_TI + (1-w_t)*S_TT` | 1 | 0.8351 | 0.0236 | 0.0423 |
| text_only_deployed | deployed text query | text | `w_t*S_TI + (1-w_t)*S_TT` | 5 | 0.7383 | 0.0859 | 0.1217 |
| text_only_deployed | deployed text query | text | `w_t*S_TI + (1-w_t)*S_TT` | 10 | 0.6892 | 0.1378 | 0.1728 |
| combined_full_deployed | deployed combined query | image + text | `raw fusion + attribute + category policy` | 1 | 0.6702 | 0.0227 | 0.0402 |
| combined_full_deployed | deployed combined query | image + text | `raw fusion + attribute + category policy` | 5 | 0.6681 | 0.1041 | 0.1458 |
| combined_full_deployed | deployed combined query | image + text | `raw fusion + attribute + category policy` | 10 | 0.6668 | 0.1765 | 0.2225 |

## Combined-search ablation

| System | Role | Input | Formula | K | Precision | Recall | F1 |
|---|---|---|---|---:|---:|---:|---:|
| combined_base | raw fusion baseline | image + text | `w_i*S_II + (1-w_i)*(w_t*S_TI + (1-w_t)*S_TT)` | 1 | 0.8436 | 0.0256 | 0.0448 |
| combined_base | raw fusion baseline | image + text | `w_i*S_II + (1-w_i)*(w_t*S_TI + (1-w_t)*S_TT)` | 5 | 0.7471 | 0.0918 | 0.1287 |
| combined_base | raw fusion baseline | image + text | `w_i*S_II + (1-w_i)*(w_t*S_TI + (1-w_t)*S_TT)` | 10 | 0.7093 | 0.1500 | 0.1850 |
| combined_attribute | attribute contribution | image + text | `raw fusion + attribute adjustment` | 1 | 0.8372 | 0.0245 | 0.0434 |
| combined_attribute | attribute contribution | image + text | `raw fusion + attribute adjustment` | 5 | 0.7328 | 0.0857 | 0.1218 |
| combined_attribute | attribute contribution | image + text | `raw fusion + attribute adjustment` | 10 | 0.6920 | 0.1436 | 0.1782 |
| combined_category | category contribution | image + text | `raw fusion + category policy` | 1 | 0.6702 | 0.0227 | 0.0402 |
| combined_category | category contribution | image + text | `raw fusion + category policy` | 5 | 0.6681 | 0.1041 | 0.1457 |
| combined_category | category contribution | image + text | `raw fusion + category policy` | 10 | 0.6670 | 0.1761 | 0.2224 |
| combined_full | full deployed ranking | image + text | `raw fusion + attribute + category policy` | 1 | 0.6702 | 0.0227 | 0.0402 |
| combined_full | full deployed ranking | image + text | `raw fusion + attribute + category policy` | 5 | 0.6681 | 0.1041 | 0.1458 |
| combined_full | full deployed ranking | image + text | `raw fusion + attribute + category policy` | 10 | 0.6668 | 0.1765 | 0.2225 |

## Attribute diagnostics

| System | K | Attribute queries | Mean attribute match | All attributes match |
|---|---:|---:|---:|---:|
| combined_base | 1 | 437 | 0.7009 | 0.5011 |
| combined_base | 5 | 437 | 0.6351 | 0.4375 |
| combined_base | 10 | 437 | 0.6015 | 0.3991 |
| combined_attribute | 1 | 437 | 0.8067 | 0.6453 |
| combined_attribute | 5 | 437 | 0.7500 | 0.5597 |
| combined_attribute | 10 | 437 | 0.7075 | 0.5098 |
| combined_category | 1 | 437 | 0.6400 | 0.4554 |
| combined_category | 5 | 437 | 0.5428 | 0.3593 |
| combined_category | 10 | 437 | 0.4977 | 0.3223 |
| combined_full | 1 | 437 | 0.7416 | 0.5767 |
| combined_full | 5 | 437 | 0.6338 | 0.4504 |
| combined_full | 10 | 437 | 0.5736 | 0.3951 |
