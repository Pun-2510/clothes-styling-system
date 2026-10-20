# Category-aware ranking comparison

The benchmark used 473 held-out test queries against 5000 catalog products.
The query product was excluded before category prediction and ranking.

## Ranking results

| System | Role | Input | Formula | K | Precision | Recall | F1 |
|---|---|---|---|---:|---:|---:|---:|
| no_category | category-free baseline | image | `S_II` | 1 | 0.7294 | 0.0204 | 0.0360 |
| no_category | category-free baseline | image | `S_II` | 5 | 0.7049 | 0.0835 | 0.1173 |
| no_category | category-free baseline | image | `S_II` | 10 | 0.6710 | 0.1345 | 0.1677 |
| threshold_soft_category | previous threshold strategy | image | `S_II + lambda*I[h>=tau]*I[c_i=c_hat]` | 1 | 0.7315 | 0.0204 | 0.0360 |
| threshold_soft_category | previous threshold strategy | image | `S_II + lambda*I[h>=tau]*I[c_i=c_hat]` | 5 | 0.7078 | 0.0843 | 0.1184 |
| threshold_soft_category | previous threshold strategy | image | `S_II + lambda*I[h>=tau]*I[c_i=c_hat]` | 10 | 0.6778 | 0.1374 | 0.1711 |
| confidence_weighted_category | confidence-weighted serving | image | `S_II + lambda*g(h)*I[c_i=c_hat]` | 1 | 0.7548 | 0.0200 | 0.0357 |
| confidence_weighted_category | confidence-weighted serving | image | `S_II + lambda*g(h)*I[c_i=c_hat]` | 5 | 0.7311 | 0.0868 | 0.1227 |
| confidence_weighted_category | confidence-weighted serving | image | `S_II + lambda*g(h)*I[c_i=c_hat]` | 10 | 0.7074 | 0.1433 | 0.1795 |

## Category prediction diagnostics

- Accuracy: 0.7484
- Macro-F1: 0.5352
- Previous threshold coverage: 0.2326
- Confidence-weighted coverage: 1.0000
- Mean category strength: 0.5513
