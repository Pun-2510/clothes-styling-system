# Deployed query-mode and combined-search comparison

The benchmark used 462 held-out queries against 5000 catalog products.
The query product was excluded before category prediction and ranking.
Query text omits the ground-truth category column; runtime catalog text uses the same category-bearing embeddings as the website.

## End-to-end query modes

| System | Role | Input | Formula | K | Precision | Recall | F1 | NDCG | MAP |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| image_only_deployed | deployed image query | image | `S_II + lambda_I*g(h)*w(c_i)` | 1 | 0.6558 | 0.0168 | 0.0312 | 0.6558 | 0.6558 |
| image_only_deployed | deployed image query | image | `S_II + lambda_I*g(h)*w(c_i)` | 5 | 0.6177 | 0.0757 | 0.1167 | 0.6292 | 0.5691 |
| image_only_deployed | deployed image query | image | `S_II + lambda_I*g(h)*w(c_i)` | 10 | 0.5911 | 0.1334 | 0.1833 | 0.6122 | 0.5280 |
| text_only_deployed | deployed text query | text | `w_t*S_TI + (1-w_t)*S_TT; exact category => hard pool` | 1 | 0.7922 | 0.0256 | 0.0459 | 0.7922 | 0.7922 |
| text_only_deployed | deployed text query | text | `w_t*S_TI + (1-w_t)*S_TT; exact category => hard pool` | 5 | 0.7463 | 0.1054 | 0.1586 | 0.7579 | 0.7171 |
| text_only_deployed | deployed text query | text | `w_t*S_TI + (1-w_t)*S_TT; exact category => hard pool` | 10 | 0.7160 | 0.1779 | 0.2461 | 0.7368 | 0.6815 |
| combined_full_deployed | deployed combined query | image + text | `raw fusion + attribute + category policy` | 1 | 0.7381 | 0.0225 | 0.0408 | 0.7381 | 0.7381 |
| combined_full_deployed | deployed combined query | image + text | `raw fusion + attribute + category policy` | 5 | 0.7247 | 0.0989 | 0.1513 | 0.7281 | 0.7071 |
| combined_full_deployed | deployed combined query | image + text | `raw fusion + attribute + category policy` | 10 | 0.7043 | 0.1720 | 0.2400 | 0.7159 | 0.6867 |

## Combined-search ablation

| System | Role | Input | Formula | K | Precision | Recall | F1 | NDCG | MAP |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| combined_base | raw fusion baseline | image + text | `w_i*S_II + (1-w_i)*(w_t*S_TI + (1-w_t)*S_TT)` | 1 | 0.7143 | 0.0204 | 0.0370 | 0.7143 | 0.7143 |
| combined_base | raw fusion baseline | image + text | `w_i*S_II + (1-w_i)*(w_t*S_TI + (1-w_t)*S_TT)` | 5 | 0.6528 | 0.0823 | 0.1258 | 0.6694 | 0.5974 |
| combined_base | raw fusion baseline | image + text | `w_i*S_II + (1-w_i)*(w_t*S_TI + (1-w_t)*S_TT)` | 10 | 0.6082 | 0.1397 | 0.1915 | 0.6392 | 0.5378 |
| combined_attribute | attribute contribution | image + text | `raw fusion + attribute adjustment` | 1 | 0.6840 | 0.0182 | 0.0337 | 0.6840 | 0.6840 |
| combined_attribute | attribute contribution | image + text | `raw fusion + attribute adjustment` | 5 | 0.5952 | 0.0728 | 0.1112 | 0.6162 | 0.5375 |
| combined_attribute | attribute contribution | image + text | `raw fusion + attribute adjustment` | 10 | 0.5437 | 0.1207 | 0.1666 | 0.5780 | 0.4685 |
| combined_category | category contribution | image + text | `raw fusion + category policy` | 1 | 0.7468 | 0.0236 | 0.0422 | 0.7468 | 0.7468 |
| combined_category | category contribution | image + text | `raw fusion + category policy` | 5 | 0.7273 | 0.0998 | 0.1522 | 0.7323 | 0.7128 |
| combined_category | category contribution | image + text | `raw fusion + category policy` | 10 | 0.7214 | 0.1772 | 0.2469 | 0.7301 | 0.7036 |
| combined_full | full deployed ranking | image + text | `raw fusion + attribute + category policy` | 1 | 0.7381 | 0.0225 | 0.0408 | 0.7381 | 0.7381 |
| combined_full | full deployed ranking | image + text | `raw fusion + attribute + category policy` | 5 | 0.7247 | 0.0989 | 0.1513 | 0.7281 | 0.7071 |
| combined_full | full deployed ranking | image + text | `raw fusion + attribute + category policy` | 10 | 0.7043 | 0.1720 | 0.2400 | 0.7159 | 0.6867 |

## Attribute diagnostics

| System | K | Attribute queries | Mean attribute match | All attributes match |
|---|---:|---:|---:|---:|
| combined_base | 1 | 403 | 0.6167 | 0.4615 |
| combined_base | 5 | 403 | 0.5530 | 0.4050 |
| combined_base | 10 | 403 | 0.5152 | 0.3717 |
| combined_attribute | 1 | 403 | 0.8516 | 0.7122 |
| combined_attribute | 5 | 403 | 0.7930 | 0.6437 |
| combined_attribute | 10 | 403 | 0.7585 | 0.6062 |
| combined_category | 1 | 403 | 0.5855 | 0.4268 |
| combined_category | 5 | 403 | 0.4973 | 0.3509 |
| combined_category | 10 | 403 | 0.4490 | 0.3133 |
| combined_full | 1 | 403 | 0.8189 | 0.6675 |
| combined_full | 5 | 403 | 0.7010 | 0.5444 |
| combined_full | 10 | 403 | 0.6321 | 0.4781 |
