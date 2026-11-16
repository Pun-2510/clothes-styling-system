# Final three-seed experiment evidence

This directory contains the lightweight final summaries used by Chapter 5 and
the appendices. The files were copied from:

`experiments/model_comparison/results/aggregate_diverse_3_seeds_map_v1/`

The aggregation covers seeds `42`, `123`, and `2026`. Each seed uses a
controlled 5,000-product gallery. Raw caches, embeddings, checkpoints, and
intermediate result directories remain local and are intentionally excluded
from Git.

## Contents

- `core_extended_pretrained/`: model and modality comparisons.
- `category_aware/`: category-aware ranking comparisons.
- `system_pipeline/`: end-to-end image-only, text-only, and combined queries.

Each group contains:

- `aggregation.json`: aggregation metadata and source runs.
- `mean_std_summary.csv`: mean and sample standard deviation across seeds.
- `per_seed_results.csv`: the individual values used by the aggregation.

These files are report evidence. They are not runtime inputs for the website.
