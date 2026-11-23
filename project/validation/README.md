# Project-local validation artifacts

`python -m src.tune_retrieval_weights` writes cached validation embeddings and
weight-search trial tables here. These files support the operational pipeline
inside `project/`; they are not repository-level model-comparison results.

Generated contents are ignored by Git. The final runtime configuration is
written to `project/retrieval_weights.json`.

For the current combined-query protocol, validation keeps the image as the
main product-type reference, limits the category-gated TF-IDF weight, and
selects category contribution independently. Do not reuse a weight file tuned
for the earlier full-catalog TF-IDF formula.
