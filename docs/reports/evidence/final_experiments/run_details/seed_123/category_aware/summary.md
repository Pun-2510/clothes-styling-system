# Category-aware ranking comparison

The benchmark used 464 held-out test queries against 5000 catalog products.
The query product was excluded before category prediction and ranking.

## Ranking results

| System | Role | Input | Formula | K | Precision | Recall | F1 | NDCG | MAP |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| no_category | category-free baseline | image | `S_II` | 1 | 0.6401 | 0.0183 | 0.0333 | 0.6401 | 0.6401 |
| no_category | category-free baseline | image | `S_II` | 5 | 0.5944 | 0.0775 | 0.1176 | 0.6060 | 0.5213 |
| no_category | category-free baseline | image | `S_II` | 10 | 0.5545 | 0.1295 | 0.1762 | 0.5804 | 0.4675 |
| hard_category_filter | hard predicted-category filter | image | `rank only c_i=c_hat` | 1 | 0.7026 | 0.0174 | 0.0327 | 0.7026 | 0.7026 |
| hard_category_filter | hard predicted-category filter | image | `rank only c_i=c_hat` | 5 | 0.7026 | 0.0852 | 0.1359 | 0.7026 | 0.7026 |
| hard_category_filter | hard predicted-category filter | image | `rank only c_i=c_hat` | 10 | 0.7026 | 0.1583 | 0.2289 | 0.7026 | 0.7026 |
| threshold_soft_category | previous threshold strategy | image | `S_II + lambda*I[h>=tau]*I[c_i=c_hat]` | 1 | 0.6422 | 0.0183 | 0.0333 | 0.6422 | 0.6422 |
| threshold_soft_category | previous threshold strategy | image | `S_II + lambda*I[h>=tau]*I[c_i=c_hat]` | 5 | 0.5987 | 0.0781 | 0.1185 | 0.6096 | 0.5263 |
| threshold_soft_category | previous threshold strategy | image | `S_II + lambda*I[h>=tau]*I[c_i=c_hat]` | 10 | 0.5638 | 0.1323 | 0.1803 | 0.5877 | 0.4779 |
| confidence_weighted_category | confidence-weighted serving | image | `S_II + lambda*g(h)*I[c_i=c_hat]` | 1 | 0.6789 | 0.0178 | 0.0331 | 0.6789 | 0.6789 |
| confidence_weighted_category | confidence-weighted serving | image | `S_II + lambda*g(h)*I[c_i=c_hat]` | 5 | 0.6556 | 0.0810 | 0.1268 | 0.6623 | 0.6261 |
| confidence_weighted_category | confidence-weighted serving | image | `S_II + lambda*g(h)*I[c_i=c_hat]` | 10 | 0.6369 | 0.1471 | 0.2034 | 0.6532 | 0.5963 |
| dynamic_relation_category | deployed prototype-and-relation policy | image | `S_II + lambda*g(h)*w(c_i)` | 1 | 0.6659 | 0.0177 | 0.0328 | 0.6659 | 0.6659 |
| dynamic_relation_category | deployed prototype-and-relation policy | image | `S_II + lambda*g(h)*w(c_i)` | 5 | 0.6272 | 0.0807 | 0.1229 | 0.6371 | 0.5729 |
| dynamic_relation_category | deployed prototype-and-relation policy | image | `S_II + lambda*g(h)*w(c_i)` | 10 | 0.5920 | 0.1372 | 0.1879 | 0.6151 | 0.5215 |

## Category prediction diagnostics

- Accuracy: 0.7026
- Macro-F1: 0.5690
- Previous threshold coverage: 0.0539
- Confidence-weighted coverage: 1.0000
- Mean category strength: 0.2914
- Dynamic route counts: {'ambiguous_related_top2': 222, 'ambiguous_unrelated_top1': 217, 'confident_top1': 25}
