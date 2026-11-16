# Category-aware ranking comparison

The benchmark used 464 held-out test queries against 5000 catalog products.
The query product was excluded before category prediction and ranking.

## Ranking results

| System | Role | Input | Formula | K | Precision | Recall | F1 | NDCG | MAP |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| no_category | category-free baseline | image | `S_II` | 1 | 0.6401 | 0.0183 | 0.0333 | 0.6401 | 0.6401 |
| no_category | category-free baseline | image | `S_II` | 5 | 0.5944 | 0.0775 | 0.1176 | 0.6060 | 0.5213 |
| no_category | category-free baseline | image | `S_II` | 10 | 0.5545 | 0.1295 | 0.1762 | 0.5804 | 0.4675 |
| hard_category_filter | hard predicted-category filter | image | `rank only c_i=c_hat` | 1 | 0.6940 | 0.0168 | 0.0316 | 0.6940 | 0.6940 |
| hard_category_filter | hard predicted-category filter | image | `rank only c_i=c_hat` | 5 | 0.6940 | 0.0822 | 0.1318 | 0.6940 | 0.6940 |
| hard_category_filter | hard predicted-category filter | image | `rank only c_i=c_hat` | 10 | 0.6940 | 0.1532 | 0.2230 | 0.6940 | 0.6940 |
| threshold_soft_category | previous threshold strategy | image | `S_II + lambda*I[h>=tau]*I[c_i=c_hat]` | 1 | 0.6422 | 0.0183 | 0.0333 | 0.6422 | 0.6422 |
| threshold_soft_category | previous threshold strategy | image | `S_II + lambda*I[h>=tau]*I[c_i=c_hat]` | 5 | 0.5978 | 0.0780 | 0.1183 | 0.6090 | 0.5255 |
| threshold_soft_category | previous threshold strategy | image | `S_II + lambda*I[h>=tau]*I[c_i=c_hat]` | 10 | 0.5625 | 0.1319 | 0.1797 | 0.5867 | 0.4765 |
| confidence_weighted_category | confidence-weighted serving | image | `S_II + lambda*g(h)*I[c_i=c_hat]` | 1 | 0.6789 | 0.0176 | 0.0327 | 0.6789 | 0.6789 |
| confidence_weighted_category | confidence-weighted serving | image | `S_II + lambda*g(h)*I[c_i=c_hat]` | 5 | 0.6595 | 0.0811 | 0.1271 | 0.6653 | 0.6296 |
| confidence_weighted_category | confidence-weighted serving | image | `S_II + lambda*g(h)*I[c_i=c_hat]` | 10 | 0.6373 | 0.1472 | 0.2036 | 0.6539 | 0.5981 |
| dynamic_relation_category | deployed prototype-and-relation policy | image | `S_II + lambda*g(h)*w(c_i)` | 1 | 0.6703 | 0.0178 | 0.0331 | 0.6703 | 0.6703 |
| dynamic_relation_category | deployed prototype-and-relation policy | image | `S_II + lambda*g(h)*w(c_i)` | 5 | 0.6284 | 0.0812 | 0.1234 | 0.6394 | 0.5748 |
| dynamic_relation_category | deployed prototype-and-relation policy | image | `S_II + lambda*g(h)*w(c_i)` | 10 | 0.5933 | 0.1381 | 0.1891 | 0.6170 | 0.5234 |

## Category prediction diagnostics

- Accuracy: 0.6940
- Macro-F1: 0.5488
- Previous threshold coverage: 0.0496
- Confidence-weighted coverage: 1.0000
- Mean category strength: 0.2983
- Dynamic route counts: {'ambiguous_related_top2': 222, 'ambiguous_unrelated_top1': 219, 'confident_top1': 23}
