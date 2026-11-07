# Project-local validation artifacts

`python -m src.tune_retrieval_weights` writes cached validation embeddings and
weight-search trial tables here. These files support the operational pipeline
inside `project/`; they are not repository-level model-comparison results.

Generated contents are ignored by Git. The final runtime configuration is
written to `project/retrieval_weights.json`.
