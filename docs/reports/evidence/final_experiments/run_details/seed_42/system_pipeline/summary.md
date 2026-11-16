# Deployed query-mode and combined-search comparison

The benchmark used 460 held-out queries against 5000 catalog products.
The query product was excluded before category prediction and ranking.
Query text omits the ground-truth category column; runtime catalog text uses the same category-bearing embeddings as the website.

## End-to-end query modes

| System | Role | Input | Formula | K | Precision | Recall | F1 | NDCG | MAP |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| image_only_deployed | deployed image query | image | `S_II + lambda_I*g(h)*w(c_i)` | 1 | 0.7174 | 0.0191 | 0.0357 | 0.7174 | 0.7174 |
| image_only_deployed | deployed image query | image | `S_II + lambda_I*g(h)*w(c_i)` | 5 | 0.6457 | 0.0822 | 0.1258 | 0.6624 | 0.6027 |
| image_only_deployed | deployed image query | image | `S_II + lambda_I*g(h)*w(c_i)` | 10 | 0.6107 | 0.1401 | 0.1902 | 0.6393 | 0.5515 |
| text_only_deployed | deployed text query | text | `w_t*S_TI + (1-w_t)*S_TT; exact category => hard pool` | 1 | 0.7783 | 0.0232 | 0.0422 | 0.7783 | 0.7783 |
| text_only_deployed | deployed text query | text | `w_t*S_TI + (1-w_t)*S_TT; exact category => hard pool` | 5 | 0.7283 | 0.1008 | 0.1514 | 0.7409 | 0.7015 |
| text_only_deployed | deployed text query | text | `w_t*S_TI + (1-w_t)*S_TT; exact category => hard pool` | 10 | 0.6996 | 0.1714 | 0.2357 | 0.7210 | 0.6646 |
| combined_full_deployed | deployed combined query | image + text | `raw fusion + attribute + category policy` | 1 | 0.7674 | 0.0213 | 0.0394 | 0.7674 | 0.7674 |
| combined_full_deployed | deployed combined query | image + text | `raw fusion + attribute + category policy` | 5 | 0.7387 | 0.0968 | 0.1503 | 0.7453 | 0.7275 |
| combined_full_deployed | deployed combined query | image + text | `raw fusion + attribute + category policy` | 10 | 0.7211 | 0.1725 | 0.2416 | 0.7344 | 0.7039 |

## Combined-search ablation

| System | Role | Input | Formula | K | Precision | Recall | F1 | NDCG | MAP |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| combined_base | raw fusion baseline | image + text | `w_i*S_II + (1-w_i)*(w_t*S_TI + (1-w_t)*S_TT)` | 1 | 0.7196 | 0.0195 | 0.0363 | 0.7196 | 0.7196 |
| combined_base | raw fusion baseline | image + text | `w_i*S_II + (1-w_i)*(w_t*S_TI + (1-w_t)*S_TT)` | 5 | 0.6535 | 0.0834 | 0.1280 | 0.6703 | 0.5967 |
| combined_base | raw fusion baseline | image + text | `w_i*S_II + (1-w_i)*(w_t*S_TI + (1-w_t)*S_TT)` | 10 | 0.6178 | 0.1444 | 0.1949 | 0.6472 | 0.5424 |
| combined_attribute | attribute contribution | image + text | `raw fusion + attribute adjustment` | 1 | 0.6804 | 0.0173 | 0.0324 | 0.6804 | 0.6804 |
| combined_attribute | attribute contribution | image + text | `raw fusion + attribute adjustment` | 5 | 0.5904 | 0.0719 | 0.1119 | 0.6098 | 0.5292 |
| combined_attribute | attribute contribution | image + text | `raw fusion + attribute adjustment` | 10 | 0.5370 | 0.1220 | 0.1661 | 0.5723 | 0.4592 |
| combined_category | category contribution | image + text | `raw fusion + category policy` | 1 | 0.7717 | 0.0213 | 0.0394 | 0.7717 | 0.7717 |
| combined_category | category contribution | image + text | `raw fusion + category policy` | 5 | 0.7570 | 0.1020 | 0.1565 | 0.7605 | 0.7454 |
| combined_category | category contribution | image + text | `raw fusion + category policy` | 10 | 0.7430 | 0.1806 | 0.2498 | 0.7538 | 0.7283 |
| combined_full | full deployed ranking | image + text | `raw fusion + attribute + category policy` | 1 | 0.7674 | 0.0213 | 0.0394 | 0.7674 | 0.7674 |
| combined_full | full deployed ranking | image + text | `raw fusion + attribute + category policy` | 5 | 0.7387 | 0.0968 | 0.1503 | 0.7453 | 0.7275 |
| combined_full | full deployed ranking | image + text | `raw fusion + attribute + category policy` | 10 | 0.7211 | 0.1725 | 0.2416 | 0.7344 | 0.7039 |

## Attribute diagnostics

| System | K | Attribute queries | Mean attribute match | All attributes match |
|---|---:|---:|---:|---:|
| combined_base | 1 | 403 | 0.6089 | 0.4392 |
| combined_base | 5 | 403 | 0.5291 | 0.3658 |
| combined_base | 10 | 403 | 0.4838 | 0.3268 |
| combined_attribute | 1 | 403 | 0.8587 | 0.7146 |
| combined_attribute | 5 | 403 | 0.8010 | 0.6328 |
| combined_attribute | 10 | 403 | 0.7614 | 0.5876 |
| combined_category | 1 | 403 | 0.5790 | 0.4094 |
| combined_category | 5 | 403 | 0.4848 | 0.3300 |
| combined_category | 10 | 403 | 0.4264 | 0.2788 |
| combined_full | 1 | 403 | 0.8204 | 0.6675 |
| combined_full | 5 | 403 | 0.7099 | 0.5400 |
| combined_full | 10 | 403 | 0.6329 | 0.4604 |
