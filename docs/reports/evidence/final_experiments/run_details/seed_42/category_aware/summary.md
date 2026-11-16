# Category-aware ranking comparison

The benchmark used 460 held-out test queries against 5000 catalog products.
The query product was excluded before category prediction and ranking.

## Ranking results

| System | Role | Input | Formula | K | Precision | Recall | F1 | NDCG | MAP |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| no_category | category-free baseline | image | `S_II` | 1 | 0.6957 | 0.0185 | 0.0346 | 0.6957 | 0.6957 |
| no_category | category-free baseline | image | `S_II` | 5 | 0.6100 | 0.0783 | 0.1193 | 0.6289 | 0.5518 |
| no_category | category-free baseline | image | `S_II` | 10 | 0.5676 | 0.1299 | 0.1755 | 0.5999 | 0.4920 |
| hard_category_filter | hard predicted-category filter | image | `rank only c_i=c_hat` | 1 | 0.7130 | 0.0185 | 0.0346 | 0.7130 | 0.7130 |
| hard_category_filter | hard predicted-category filter | image | `rank only c_i=c_hat` | 5 | 0.7130 | 0.0906 | 0.1421 | 0.7130 | 0.7130 |
| hard_category_filter | hard predicted-category filter | image | `rank only c_i=c_hat` | 10 | 0.7130 | 0.1658 | 0.2364 | 0.7130 | 0.7130 |
| threshold_soft_category | previous threshold strategy | image | `S_II + lambda*I[h>=tau]*I[c_i=c_hat]` | 1 | 0.6957 | 0.0185 | 0.0346 | 0.6957 | 0.6957 |
| threshold_soft_category | previous threshold strategy | image | `S_II + lambda*I[h>=tau]*I[c_i=c_hat]` | 5 | 0.6143 | 0.0791 | 0.1206 | 0.6322 | 0.5566 |
| threshold_soft_category | previous threshold strategy | image | `S_II + lambda*I[h>=tau]*I[c_i=c_hat]` | 10 | 0.5748 | 0.1328 | 0.1793 | 0.6056 | 0.5007 |
| confidence_weighted_category | confidence-weighted serving | image | `S_II + lambda*g(h)*I[c_i=c_hat]` | 1 | 0.7217 | 0.0188 | 0.0352 | 0.7217 | 0.7217 |
| confidence_weighted_category | confidence-weighted serving | image | `S_II + lambda*g(h)*I[c_i=c_hat]` | 5 | 0.6783 | 0.0858 | 0.1329 | 0.6893 | 0.6545 |
| confidence_weighted_category | confidence-weighted serving | image | `S_II + lambda*g(h)*I[c_i=c_hat]` | 10 | 0.6600 | 0.1523 | 0.2087 | 0.6797 | 0.6256 |
| dynamic_relation_category | deployed prototype-and-relation policy | image | `S_II + lambda*g(h)*w(c_i)` | 1 | 0.7174 | 0.0191 | 0.0357 | 0.7174 | 0.7174 |
| dynamic_relation_category | deployed prototype-and-relation policy | image | `S_II + lambda*g(h)*w(c_i)` | 5 | 0.6457 | 0.0822 | 0.1258 | 0.6624 | 0.6027 |
| dynamic_relation_category | deployed prototype-and-relation policy | image | `S_II + lambda*g(h)*w(c_i)` | 10 | 0.6107 | 0.1401 | 0.1902 | 0.6393 | 0.5515 |

## Category prediction diagnostics

- Accuracy: 0.7130
- Macro-F1: 0.6097
- Previous threshold coverage: 0.0370
- Confidence-weighted coverage: 1.0000
- Mean category strength: 0.2792
- Dynamic route counts: {'ambiguous_related_top2': 235, 'ambiguous_unrelated_top1': 208, 'confident_top1': 17}
