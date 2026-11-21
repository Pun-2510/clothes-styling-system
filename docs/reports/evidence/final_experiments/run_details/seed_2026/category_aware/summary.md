# Category-aware ranking comparison

The benchmark used 462 held-out test queries against 5000 catalog products.
The query product was excluded before category prediction and ranking.

## Ranking results

| System | Role | Input | Formula | K | Precision | Recall | F1 | NDCG | MAP |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| no_category | category-free baseline | image | `S_II` | 1 | 0.6190 | 0.0160 | 0.0297 | 0.6190 | 0.6190 |
| no_category | category-free baseline | image | `S_II` | 5 | 0.5861 | 0.0719 | 0.1104 | 0.5974 | 0.5223 |
| no_category | category-free baseline | image | `S_II` | 10 | 0.5463 | 0.1244 | 0.1691 | 0.5716 | 0.4678 |
| hard_category_filter | hard predicted-category filter | image | `rank only c_i=c_hat` | 1 | 0.6818 | 0.0168 | 0.0316 | 0.6818 | 0.6818 |
| hard_category_filter | hard predicted-category filter | image | `rank only c_i=c_hat` | 5 | 0.6818 | 0.0821 | 0.1313 | 0.6818 | 0.6818 |
| hard_category_filter | hard predicted-category filter | image | `rank only c_i=c_hat` | 10 | 0.6818 | 0.1527 | 0.2216 | 0.6818 | 0.6818 |
| threshold_soft_category | previous threshold strategy | image | `S_II + lambda*I[h>=tau]*I[c_i=c_hat]` | 1 | 0.6190 | 0.0160 | 0.0297 | 0.6190 | 0.6190 |
| threshold_soft_category | previous threshold strategy | image | `S_II + lambda*I[h>=tau]*I[c_i=c_hat]` | 5 | 0.5874 | 0.0709 | 0.1100 | 0.5979 | 0.5240 |
| threshold_soft_category | previous threshold strategy | image | `S_II + lambda*I[h>=tau]*I[c_i=c_hat]` | 10 | 0.5491 | 0.1229 | 0.1693 | 0.5730 | 0.4715 |
| confidence_weighted_category | confidence-weighted serving | image | `S_II + lambda*g(h)*I[c_i=c_hat]` | 1 | 0.6775 | 0.0175 | 0.0326 | 0.6775 | 0.6775 |
| confidence_weighted_category | confidence-weighted serving | image | `S_II + lambda*g(h)*I[c_i=c_hat]` | 5 | 0.6550 | 0.0782 | 0.1236 | 0.6597 | 0.6281 |
| confidence_weighted_category | confidence-weighted serving | image | `S_II + lambda*g(h)*I[c_i=c_hat]` | 10 | 0.6303 | 0.1402 | 0.1975 | 0.6461 | 0.5976 |
| dynamic_relation_category | deployed prototype-and-relation policy | image | `S_II + lambda*g(h)*w(c_i)` | 1 | 0.6515 | 0.0166 | 0.0309 | 0.6515 | 0.6515 |
| dynamic_relation_category | deployed prototype-and-relation policy | image | `S_II + lambda*g(h)*w(c_i)` | 5 | 0.6104 | 0.0721 | 0.1132 | 0.6223 | 0.5632 |
| dynamic_relation_category | deployed prototype-and-relation policy | image | `S_II + lambda*g(h)*w(c_i)` | 10 | 0.5801 | 0.1269 | 0.1776 | 0.6023 | 0.5172 |

## Category prediction diagnostics

- Accuracy: 0.6818
- Macro-F1: 0.5675
- Previous threshold coverage: 0.0411
- Confidence-weighted coverage: 1.0000
- Mean category strength: 0.3025
- Dynamic route counts: {'ambiguous_unrelated_top1': 229, 'ambiguous_related_top2': 214, 'confident_top1': 19}
