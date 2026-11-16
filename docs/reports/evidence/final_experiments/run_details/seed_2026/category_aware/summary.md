# Category-aware ranking comparison

The benchmark used 462 held-out test queries against 5000 catalog products.
The query product was excluded before category prediction and ranking.

## Ranking results

| System | Role | Input | Formula | K | Precision | Recall | F1 | NDCG | MAP |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| no_category | category-free baseline | image | `S_II` | 1 | 0.6190 | 0.0160 | 0.0297 | 0.6190 | 0.6190 |
| no_category | category-free baseline | image | `S_II` | 5 | 0.5861 | 0.0719 | 0.1104 | 0.5974 | 0.5223 |
| no_category | category-free baseline | image | `S_II` | 10 | 0.5463 | 0.1244 | 0.1691 | 0.5716 | 0.4678 |
| hard_category_filter | hard predicted-category filter | image | `rank only c_i=c_hat` | 1 | 0.6753 | 0.0167 | 0.0314 | 0.6753 | 0.6753 |
| hard_category_filter | hard predicted-category filter | image | `rank only c_i=c_hat` | 5 | 0.6753 | 0.0816 | 0.1304 | 0.6753 | 0.6753 |
| hard_category_filter | hard predicted-category filter | image | `rank only c_i=c_hat` | 10 | 0.6753 | 0.1517 | 0.2202 | 0.6753 | 0.6753 |
| threshold_soft_category | previous threshold strategy | image | `S_II + lambda*I[h>=tau]*I[c_i=c_hat]` | 1 | 0.6190 | 0.0160 | 0.0297 | 0.6190 | 0.6190 |
| threshold_soft_category | previous threshold strategy | image | `S_II + lambda*I[h>=tau]*I[c_i=c_hat]` | 5 | 0.5879 | 0.0721 | 0.1108 | 0.5986 | 0.5242 |
| threshold_soft_category | previous threshold strategy | image | `S_II + lambda*I[h>=tau]*I[c_i=c_hat]` | 10 | 0.5506 | 0.1255 | 0.1707 | 0.5749 | 0.4730 |
| confidence_weighted_category | confidence-weighted serving | image | `S_II + lambda*g(h)*I[c_i=c_hat]` | 1 | 0.6688 | 0.0175 | 0.0326 | 0.6688 | 0.6688 |
| confidence_weighted_category | confidence-weighted serving | image | `S_II + lambda*g(h)*I[c_i=c_hat]` | 5 | 0.6502 | 0.0789 | 0.1244 | 0.6562 | 0.6268 |
| confidence_weighted_category | confidence-weighted serving | image | `S_II + lambda*g(h)*I[c_i=c_hat]` | 10 | 0.6288 | 0.1423 | 0.1991 | 0.6450 | 0.5984 |
| dynamic_relation_category | deployed prototype-and-relation policy | image | `S_II + lambda*g(h)*w(c_i)` | 1 | 0.6558 | 0.0168 | 0.0312 | 0.6558 | 0.6558 |
| dynamic_relation_category | deployed prototype-and-relation policy | image | `S_II + lambda*g(h)*w(c_i)` | 5 | 0.6177 | 0.0757 | 0.1167 | 0.6292 | 0.5691 |
| dynamic_relation_category | deployed prototype-and-relation policy | image | `S_II + lambda*g(h)*w(c_i)` | 10 | 0.5911 | 0.1334 | 0.1833 | 0.6122 | 0.5280 |

## Category prediction diagnostics

- Accuracy: 0.6753
- Macro-F1: 0.5626
- Previous threshold coverage: 0.0346
- Confidence-weighted coverage: 1.0000
- Mean category strength: 0.3168
- Dynamic route counts: {'ambiguous_related_top2': 236, 'ambiguous_unrelated_top1': 210, 'confident_top1': 16}
