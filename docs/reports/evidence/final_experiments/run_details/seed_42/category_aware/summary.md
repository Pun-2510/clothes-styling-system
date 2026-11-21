# Category-aware ranking comparison

The benchmark used 460 held-out test queries against 5000 catalog products.
The query product was excluded before category prediction and ranking.

## Ranking results

| System | Role | Input | Formula | K | Precision | Recall | F1 | NDCG | MAP |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| no_category | category-free baseline | image | `S_II` | 1 | 0.6957 | 0.0185 | 0.0346 | 0.6957 | 0.6957 |
| no_category | category-free baseline | image | `S_II` | 5 | 0.6100 | 0.0783 | 0.1193 | 0.6289 | 0.5518 |
| no_category | category-free baseline | image | `S_II` | 10 | 0.5676 | 0.1299 | 0.1755 | 0.5999 | 0.4920 |
| hard_category_filter | hard predicted-category filter | image | `rank only c_i=c_hat` | 1 | 0.7065 | 0.0184 | 0.0344 | 0.7065 | 0.7065 |
| hard_category_filter | hard predicted-category filter | image | `rank only c_i=c_hat` | 5 | 0.7065 | 0.0901 | 0.1412 | 0.7065 | 0.7065 |
| hard_category_filter | hard predicted-category filter | image | `rank only c_i=c_hat` | 10 | 0.7065 | 0.1647 | 0.2347 | 0.7065 | 0.7065 |
| threshold_soft_category | previous threshold strategy | image | `S_II + lambda*I[h>=tau]*I[c_i=c_hat]` | 1 | 0.6957 | 0.0185 | 0.0346 | 0.6957 | 0.6957 |
| threshold_soft_category | previous threshold strategy | image | `S_II + lambda*I[h>=tau]*I[c_i=c_hat]` | 5 | 0.6143 | 0.0789 | 0.1203 | 0.6323 | 0.5567 |
| threshold_soft_category | previous threshold strategy | image | `S_II + lambda*I[h>=tau]*I[c_i=c_hat]` | 10 | 0.5739 | 0.1300 | 0.1780 | 0.6045 | 0.4998 |
| confidence_weighted_category | confidence-weighted serving | image | `S_II + lambda*g(h)*I[c_i=c_hat]` | 1 | 0.7152 | 0.0188 | 0.0351 | 0.7152 | 0.7152 |
| confidence_weighted_category | confidence-weighted serving | image | `S_II + lambda*g(h)*I[c_i=c_hat]` | 5 | 0.6778 | 0.0861 | 0.1335 | 0.6873 | 0.6526 |
| confidence_weighted_category | confidence-weighted serving | image | `S_II + lambda*g(h)*I[c_i=c_hat]` | 10 | 0.6552 | 0.1495 | 0.2070 | 0.6747 | 0.6224 |
| dynamic_relation_category | deployed prototype-and-relation policy | image | `S_II + lambda*g(h)*w(c_i)` | 1 | 0.7130 | 0.0191 | 0.0356 | 0.7130 | 0.7130 |
| dynamic_relation_category | deployed prototype-and-relation policy | image | `S_II + lambda*g(h)*w(c_i)` | 5 | 0.6470 | 0.0822 | 0.1260 | 0.6627 | 0.6034 |
| dynamic_relation_category | deployed prototype-and-relation policy | image | `S_II + lambda*g(h)*w(c_i)` | 10 | 0.6054 | 0.1372 | 0.1885 | 0.6347 | 0.5467 |

## Category prediction diagnostics

- Accuracy: 0.7065
- Macro-F1: 0.6048
- Previous threshold coverage: 0.0370
- Confidence-weighted coverage: 1.0000
- Mean category strength: 0.2819
- Dynamic route counts: {'ambiguous_unrelated_top1': 223, 'ambiguous_related_top2': 220, 'confident_top1': 17}
