"""Evaluate deployed query modes and the combined-search ranking pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .compare_full_gallery import (
    DEFAULT_CACHE,
    DEFAULT_CATALOG,
    DEFAULT_FINETUNED_IMAGES,
    _annotated_category_rows,
    _annotated_rows,
    _load_and_validate_data,
    _markdown_table,
)
from .compare_modalities import (
    DEFAULT_RUN,
    PROJECT,
    _cache_signature,
    _cached_embeddings,
    _prepare_output,
)
from .encoders import encode_clip_text
from .evaluation import evaluate_query_gallery_category_rankings


DEFAULT_FINETUNED_TEXT = PROJECT / "embeddings" / "text_embeddings.npy"


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--catalog-csv", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument(
        "--image-embeddings", type=Path, default=DEFAULT_FINETUNED_IMAGES
    )
    parser.add_argument(
        "--text-embeddings", type=Path, default=DEFAULT_FINETUNED_TEXT
    )
    parser.add_argument("--query-text-cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--expected-gallery-size", type=int, default=5000)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument("--image-weight", type=float, default=0.50)
    parser.add_argument("--text-image-weight", type=float, default=0.50)
    parser.add_argument("--attribute-weight", type=float, default=0.25)
    parser.add_argument("--image-category-bonus", type=float, default=0.05)
    parser.add_argument("--combined-category-bonus", type=float, default=0.10)
    parser.add_argument("--category-top-per-class", type=int, default=3)
    parser.add_argument("--category-confidence-floor", type=float, default=0.50)
    parser.add_argument("--category-confidence-threshold", type=float, default=0.60)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def _validate_arguments(args):
    if args.expected_gallery_size < 2 or args.batch_size < 1:
        raise ValueError("Gallery size and batch size must be positive.")
    if args.category_top_per_class < 1 or any(k < 1 for k in args.ks):
        raise ValueError("Top-per-class and every K value must be positive.")
    for name in ("image_weight", "text_image_weight", "attribute_weight"):
        if not 0.0 <= getattr(args, name) <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1.")
    if args.image_category_bonus < 0.0 or args.combined_category_bonus < 0.0:
        raise ValueError("Category bonuses cannot be negative.")
    if not (
        0.0
        <= args.category_confidence_floor
        < args.category_confidence_threshold
        <= 1.0
    ):
        raise ValueError(
            "Category thresholds must satisfy 0 <= floor < threshold <= 1."
        )


def _normalized(matrix, label):
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError(f"{label} must be a finite two-dimensional matrix.")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 0.0):
        raise ValueError(f"{label} contains a zero vector.")
    return values / norms


def _load_runtime_embeddings(path, model_identity_value, catalog_csv, rows, label):
    sys.path.insert(0, str(PROJECT))
    from src.clip_runtime import validate_embedding_metadata

    validate_embedding_metadata(path, model_identity_value, catalog_csv)
    values = np.load(path, allow_pickle=False)
    if values.ndim != 2 or len(values) != rows or not np.isfinite(values).all():
        raise ValueError(f"Invalid {label}: {path}")
    return _normalized(values, label), {
        "source": str(path.resolve()),
        "rows": int(values.shape[0]),
        "dimension": int(values.shape[1]),
    }


def _rank_rows(scores, excluded_positions, candidate_groups=None):
    values = np.asarray(scores, dtype=np.float32)
    all_indices = np.arange(values.shape[1], dtype=np.int64)
    rankings = []
    for query_index, row in enumerate(values):
        candidates = all_indices[all_indices != excluded_positions[query_index]]
        if candidate_groups is not None and candidate_groups[query_index] is not None:
            requested = np.asarray(candidate_groups[query_index], dtype=np.int64)
            requested = requested[requested != excluded_positions[query_index]]
            if len(requested):
                candidates = requested
        rankings.append(candidates[np.argsort(-row[candidates], kind="stable")])
    return rankings


def _attribute_masks(products, rules):
    search_text = (
        products[["product_name", "description", "category"]]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.lower()
    )
    masks = {}
    for name, _, candidate_patterns in rules:
        matched = np.zeros(len(products), dtype=bool)
        for pattern in candidate_patterns:
            matched |= search_text.str.contains(
                pattern, regex=True, na=False
            ).to_numpy()
        masks[name] = matched.astype(np.float32)
    return masks


def _attribute_score_matrix(query_attributes, masks, gallery_size):
    scores = np.zeros((len(query_attributes), gallery_size), dtype=np.float32)
    for query_index, attributes in enumerate(query_attributes):
        if attributes:
            scores[query_index] = np.mean(
                np.stack([masks[name] for name in attributes]), axis=0
            )
    return scores


def _attribute_metrics(rankings_by_system, attribute_scores, query_attributes, ks):
    eligible = [index for index, values in enumerate(query_attributes) if values]
    rows = []
    for system, rankings in rankings_by_system.items():
        for k in ks:
            means = []
            all_matches = []
            for query_index in eligible:
                top = np.asarray(rankings[query_index], dtype=np.int64)[:k]
                if len(top):
                    selected = attribute_scores[query_index, top]
                    means.append(float(np.mean(selected)))
                    all_matches.append(float(np.mean(selected >= 1.0)))
            rows.append(
                {
                    "system": system,
                    "k": k,
                    "attribute_queries": len(eligible),
                    "mean_attribute_match": float(np.mean(means)) if means else None,
                    "all_attributes_match_rate": (
                        float(np.mean(all_matches)) if all_matches else None
                    ),
                }
            )
    return rows


def _category_policy_rankings(
    scores,
    category_sources,
    applied_categories,
    predicted_categories,
    category_strengths,
    gallery_categories,
    category_indices,
    excluded_positions,
    soft_bonus,
):
    adjusted = np.asarray(scores, dtype=np.float32).copy()
    candidate_groups = []
    for query_index, source in enumerate(category_sources):
        if source == "image":
            matches = gallery_categories == predicted_categories[query_index]
            adjusted[query_index] += (
                soft_bonus * category_strengths[query_index] * matches
            )
            candidate_groups.append(None)
        elif source in {"text", "text_image"}:
            candidate_groups.append(category_indices[applied_categories[query_index]])
        else:
            candidate_groups.append(None)
    return _rank_rows(adjusted, excluded_positions, candidate_groups)


def _write_group(output, name, rows, category_rows):
    folder = output / name
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(folder / "comparison.csv", index=False)
    pd.DataFrame(category_rows).to_csv(
        folder / "comparison_by_category.csv", index=False
    )


def self_test():
    scores = np.asarray([[0.9, 0.8, 0.7], [0.4, 0.6, 0.5]], dtype=np.float32)
    rankings = _rank_rows(scores, np.asarray([0, 1]))
    assert rankings[0].tolist() == [1, 2]
    assert rankings[1].tolist() == [2, 0]
    attrs = _attribute_metrics(
        {"base": rankings},
        np.asarray([[0.0, 1.0, 0.0], [1.0, 0.0, 0.5]], dtype=np.float32),
        [["red"], []],
        [1],
    )
    assert attrs[0]["attribute_queries"] == 1
    assert attrs[0]["mean_attribute_match"] == 1.0
    print("PASS: query routing, self exclusion and attribute diagnostics")


def main():
    args = arguments()
    if args.self_test:
        self_test()
        return
    _validate_arguments(args)

    sys.path.insert(0, str(PROJECT))
    from src.category_ranking import (
        category_confidence_strength,
        rank_category_candidates,
    )
    from src.clip_data import file_sha256
    from src.clip_runtime import model_identity, same_model_identity
    from src.query_signals import (
        ATTRIBUTE_RULES,
        categories_are_related,
        extract_query_attributes,
        infer_category_from_text,
    )

    _, queries, gallery, query_positions, experiment, training, training_path = (
        _load_and_validate_data(args, file_sha256)
    )
    checkpoint = args.run_dir / "best"
    checkpoint_identity = model_identity(checkpoint)
    if not same_model_identity(training["checkpoint_identity"], checkpoint_identity):
        raise ValueError("The fine-tuned checkpoint differs from training.json.")

    ks = sorted(set(args.ks))
    settings = {
        "protocol": "deployed query modes over held-out queries and full catalog",
        "run_dir": str(args.run_dir.resolve()),
        "catalog_csv": str(args.catalog_csv.resolve()),
        "dataset_sha256": experiment["dataset_sha256"],
        "training_json_sha256": file_sha256(training_path),
        "catalog_sha256": file_sha256(args.catalog_csv),
        "query_count": len(queries),
        "gallery_size": len(gallery),
        "ks": ks,
        "image_weight": args.image_weight,
        "text_image_weight": args.text_image_weight,
        "attribute_weight": args.attribute_weight,
        "image_category_bonus": args.image_category_bonus,
        "combined_category_bonus": args.combined_category_bonus,
        "category_top_per_class": args.category_top_per_class,
        "category_confidence_floor": args.category_confidence_floor,
        "category_confidence_threshold": args.category_confidence_threshold,
        "query_text": "product name and description; category column omitted",
        "gallery_text": "runtime product text including product, category and description",
        "self_match": "excluded before category prediction and ranking",
        "relevance": "same ground-truth category",
    }
    if args.dry_run:
        print(json.dumps(settings, indent=2, ensure_ascii=False))
        print(
            f"DRY RUN: system-pipeline {len(queries)}-query x "
            f"{len(gallery):,}-gallery configuration is valid."
        )
        return

    image_vectors, image_metadata = _load_runtime_embeddings(
        args.image_embeddings,
        checkpoint_identity,
        args.catalog_csv,
        len(gallery),
        "runtime image embeddings",
    )
    runtime_text_vectors, text_metadata = _load_runtime_embeddings(
        args.text_embeddings,
        checkpoint_identity,
        args.catalog_csv,
        len(gallery),
        "runtime text embeddings",
    )

    query_signature = _cache_signature(
        gallery,
        {
            "architecture": "CLIP text encoder",
            "identity": checkpoint_identity,
            "category_in_text": False,
        },
    )
    query_text_vectors, query_text_metadata = _cached_embeddings(
        args.query_text_cache_dir,
        "clip_finetuned_text_gallery",
        query_signature,
        lambda: encode_clip_text(
            gallery.evaluation_text.tolist(),
            checkpoint,
            args.device,
            args.batch_size,
        ),
        not args.no_cache,
    )
    query_text_vectors = _normalized(query_text_vectors, "query text embeddings")

    query_image_vectors = image_vectors[query_positions]
    query_text_inputs = query_text_vectors[query_positions]
    visual_scores = query_image_vectors @ image_vectors.T
    text_to_image_scores = query_text_inputs @ image_vectors.T
    text_to_text_scores = query_text_inputs @ runtime_text_vectors.T
    for scores in (visual_scores, text_to_image_scores, text_to_text_scores):
        scores[np.arange(len(queries)), query_positions] = -np.inf

    text_scores = (
        args.text_image_weight * text_to_image_scores
        + (1.0 - args.text_image_weight) * text_to_text_scores
    )
    combined_base_scores = (
        args.image_weight * visual_scores
        + (1.0 - args.image_weight) * text_scores
    )

    query_texts = queries.evaluation_text.astype(str).tolist()
    query_attributes = [extract_query_attributes(text) for text in query_texts]
    attribute_masks = _attribute_masks(gallery, ATTRIBUTE_RULES)
    attribute_scores = _attribute_score_matrix(
        query_attributes, attribute_masks, len(gallery)
    )
    combined_attribute_scores = combined_base_scores.copy()
    for query_index, attributes in enumerate(query_attributes):
        if attributes:
            combined_attribute_scores[query_index] = (
                (1.0 - args.attribute_weight) * combined_base_scores[query_index]
                + args.attribute_weight * attribute_scores[query_index]
            )

    gallery_categories = gallery.category.fillna("").astype(str).to_numpy()
    query_categories = queries.category.fillna("").astype(str).to_numpy()
    query_ids = queries.product_id.astype(str).to_numpy()
    gallery_ids = gallery.product_id.astype(str).to_numpy()
    category_indices = {
        str(category): np.asarray(indices, dtype=np.int64)
        for category, indices in gallery.groupby(
            gallery["category"].astype(str), sort=True
        ).groups.items()
    }
    catalog_categories = sorted(category_indices)

    predicted_categories = []
    category_confidences = []
    category_strengths = []
    text_categories = []
    applied_categories = []
    category_sources = []
    for query_index, scores in enumerate(visual_scores):
        ranked_categories, confidence = rank_category_candidates(
            scores, category_indices, args.category_top_per_class
        )
        predicted = ranked_categories[0][0]
        strength = category_confidence_strength(
            confidence,
            args.category_confidence_floor,
            args.category_confidence_threshold,
        )
        text_category = infer_category_from_text(
            query_texts[query_index], catalog_categories
        )
        if text_category is not None:
            if strength > 0.0 and categories_are_related(predicted, text_category):
                applied_category = predicted
                category_source = "text_image"
            else:
                applied_category = text_category
                category_source = "text"
        elif strength > 0.0:
            applied_category = predicted
            category_source = "image"
        else:
            applied_category = None
            category_source = None

        predicted_categories.append(predicted)
        category_confidences.append(confidence)
        category_strengths.append(strength)
        text_categories.append(text_category)
        applied_categories.append(applied_category)
        category_sources.append(category_source)

    predicted_categories = np.asarray(predicted_categories, dtype=str)
    category_confidences = np.asarray(category_confidences, dtype=np.float32)
    category_strengths = np.asarray(category_strengths, dtype=np.float32)

    image_deployed_scores = visual_scores.copy()
    for query_index, predicted in enumerate(predicted_categories):
        image_deployed_scores[query_index] += (
            args.image_category_bonus
            * category_strengths[query_index]
            * (gallery_categories == predicted)
        )

    combined_base_rankings = _rank_rows(combined_base_scores, query_positions)
    combined_attribute_rankings = _rank_rows(
        combined_attribute_scores, query_positions
    )
    combined_category_rankings = _category_policy_rankings(
        combined_base_scores,
        category_sources,
        applied_categories,
        predicted_categories,
        category_strengths,
        gallery_categories,
        category_indices,
        query_positions,
        args.combined_category_bonus,
    )
    combined_full_rankings = _category_policy_rankings(
        combined_attribute_scores,
        category_sources,
        applied_categories,
        predicted_categories,
        category_strengths,
        gallery_categories,
        category_indices,
        query_positions,
        args.combined_category_bonus,
    )

    query_mode_rankings = {
        "image_only_deployed": _rank_rows(image_deployed_scores, query_positions),
        "text_only_deployed": _rank_rows(text_scores, query_positions),
        "combined_full_deployed": combined_full_rankings,
    }
    combined_rankings = {
        "combined_base": combined_base_rankings,
        "combined_attribute": combined_attribute_rankings,
        "combined_category": combined_category_rankings,
        "combined_full": combined_full_rankings,
    }
    query_mode_specs = {
        "image_only_deployed": {
            "comparison_group": "query_mode",
            "role": "deployed image query",
            "deployed": True,
            "input": "image",
            "formula": "S_II + lambda_I*g(h)*I[c_i=c_hat]",
        },
        "text_only_deployed": {
            "comparison_group": "query_mode",
            "role": "deployed text query",
            "deployed": True,
            "input": "text",
            "formula": "w_t*S_TI + (1-w_t)*S_TT",
        },
        "combined_full_deployed": {
            "comparison_group": "query_mode",
            "role": "deployed combined query",
            "deployed": True,
            "input": "image + text",
            "formula": "raw fusion + attribute + category policy",
        },
    }
    combined_specs = {
        "combined_base": {
            "comparison_group": "combined_ablation",
            "role": "raw fusion baseline",
            "deployed": False,
            "input": "image + text",
            "formula": "w_i*S_II + (1-w_i)*(w_t*S_TI + (1-w_t)*S_TT)",
        },
        "combined_attribute": {
            "comparison_group": "combined_ablation",
            "role": "attribute contribution",
            "deployed": False,
            "input": "image + text",
            "formula": "raw fusion + attribute adjustment",
        },
        "combined_category": {
            "comparison_group": "combined_ablation",
            "role": "category contribution",
            "deployed": False,
            "input": "image + text",
            "formula": "raw fusion + category policy",
        },
        "combined_full": {
            "comparison_group": "combined_ablation",
            "role": "full deployed ranking",
            "deployed": True,
            "input": "image + text",
            "formula": "raw fusion + attribute + category policy",
        },
    }

    def evaluate(rankings):
        return {
            name: evaluate_query_gallery_category_rankings(
                values,
                query_ids,
                query_categories,
                gallery_ids,
                gallery_categories,
                ks,
            )
            for name, values in rankings.items()
        }

    query_mode_metrics = evaluate(query_mode_rankings)
    combined_metrics = evaluate(combined_rankings)
    query_mode_rows = _annotated_rows(
        query_mode_specs,
        query_mode_metrics,
        ks,
        "same-category end-to-end retrieval",
    )
    query_mode_category_rows = _annotated_category_rows(
        query_mode_specs,
        query_mode_metrics,
        ks,
        "same-category end-to-end retrieval",
    )
    combined_rows = _annotated_rows(
        combined_specs,
        combined_metrics,
        ks,
        "same-category combined-search ablation",
    )
    combined_category_rows = _annotated_category_rows(
        combined_specs,
        combined_metrics,
        ks,
        "same-category combined-search ablation",
    )
    attribute_rows = _attribute_metrics(
        combined_rankings, attribute_scores, query_attributes, ks
    )

    diagnostics = pd.DataFrame(
        {
            "product_id": query_ids,
            "actual_category": query_categories,
            "predicted_image_category": predicted_categories,
            "category_confidence": category_confidences,
            "category_strength": category_strengths,
            "text_category": [value or "" for value in text_categories],
            "applied_category": [value or "" for value in applied_categories],
            "category_source": [value or "" for value in category_sources],
            "attributes": [";".join(values) for values in query_attributes],
        }
    )
    settings["diagnostics"] = {
        "queries_with_attributes": int(sum(bool(values) for values in query_attributes)),
        "queries_with_explicit_text_category": int(
            sum(value is not None for value in text_categories)
        ),
        "category_source_counts": {
            str(name): int(count)
            for name, count in diagnostics["category_source"]
            .replace("", "not_used")
            .value_counts()
            .items()
        },
    }

    output = _prepare_output(args.output_dir)
    all_rows = query_mode_rows + combined_rows
    all_category_rows = query_mode_category_rows + combined_category_rows
    pd.DataFrame(all_rows).to_csv(output / "comparison.csv", index=False)
    pd.DataFrame(all_category_rows).to_csv(
        output / "comparison_by_category.csv", index=False
    )
    pd.DataFrame(attribute_rows).to_csv(
        output / "attribute_comparison.csv", index=False
    )
    diagnostics.to_csv(output / "query_diagnostics.csv", index=False)
    _write_group(
        output, "query_modes", query_mode_rows, query_mode_category_rows
    )
    _write_group(
        output, "combined_ablation", combined_rows, combined_category_rows
    )
    (output / "experiment.json").write_text(
        json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "settings": settings,
        "embeddings": {
            "image": image_metadata,
            "runtime_text": text_metadata,
            "query_text": query_text_metadata,
        },
        "query_mode_metrics": query_mode_metrics,
        "combined_ablation_metrics": combined_metrics,
        "attribute_metrics": attribute_rows,
    }
    (output / "comparison.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    summary = [
        "# Deployed query-mode and combined-search comparison",
        "",
        f"The benchmark used {len(queries)} held-out queries against "
        f"{len(gallery)} catalog products.",
        "The query product was excluded before category prediction and ranking.",
        "Query text omits the ground-truth category column; runtime catalog text "
        "uses the same category-bearing embeddings as the website.",
        "",
        *_markdown_table("End-to-end query modes", query_mode_rows),
        "",
        *_markdown_table("Combined-search ablation", combined_rows),
        "",
        "## Attribute diagnostics",
        "",
        "| System | K | Attribute queries | Mean attribute match | All attributes match |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in attribute_rows:
        mean_match = row["mean_attribute_match"]
        all_match = row["all_attributes_match_rate"]
        summary.append(
            f"| {row['system']} | {row['k']} | {row['attribute_queries']} | "
            f"{mean_match:.4f} | {all_match:.4f} |"
            if mean_match is not None and all_match is not None
            else (
                f"| {row['system']} | {row['k']} | "
                f"{row['attribute_queries']} | N/A | N/A |"
            )
        )
    (output / "summary.md").write_text(
        "\n".join(summary) + "\n", encoding="utf-8"
    )

    print(
        pd.DataFrame(all_rows).to_string(
            index=False, float_format=lambda value: f"{value:.4f}"
        )
    )
    print(f"Saved deployed pipeline comparison to: {output}")


if __name__ == "__main__":
    main()
