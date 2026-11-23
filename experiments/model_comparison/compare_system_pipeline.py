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
from .category_runtime import (
    CategoryPolicySettings,
    apply_dynamic_category_bonus,
    build_category_evidence,
    evidence_columns,
    load_category_prototypes,
    validate_policy_settings,
)

sys.path.insert(0, str(PROJECT))
from src.category_ranking import apply_category_gated_metadata_refinement


DEFAULT_FINETUNED_TEXT = PROJECT / "embeddings" / "text_embeddings.npy"


def _serving_defaults():
    sys.path.insert(0, str(PROJECT))
    from src.config import (
        CATEGORY_BONUS,
        CATEGORY_CONFIDENCE_FLOOR,
        CATEGORY_CONFIDENCE_THRESHOLD,
        CATEGORY_PROTOTYPE_WEIGHT,
        CATEGORY_RELATION_NEIGHBORS,
        CATEGORY_TOP_PER_CLASS,
        COMBINED_METADATA_WEIGHT,
        COMBINED_CATEGORY_WEIGHT,
        COMBINED_IMAGE_WEIGHT,
        METADATA_INDEX_DIR,
        TEXT_METADATA_WEIGHT,
        TEXT_RETRIEVAL_IMAGE_WEIGHT,
        RELATED_TOP1_BONUS_FACTOR,
        RELATED_TOP2_BONUS_FACTOR,
        UNRELATED_TOP1_BONUS_FACTOR,
    )

    return {
        "image_weight": COMBINED_IMAGE_WEIGHT,
        "text_image_weight": TEXT_RETRIEVAL_IMAGE_WEIGHT,
        "metadata_index_dir": METADATA_INDEX_DIR,
        "text_metadata_weight": TEXT_METADATA_WEIGHT,
        "combined_metadata_weight": COMBINED_METADATA_WEIGHT,
        "image_category_bonus": CATEGORY_BONUS,
        "combined_category_bonus": COMBINED_CATEGORY_WEIGHT,
        "category_top_per_class": CATEGORY_TOP_PER_CLASS,
        "category_confidence_floor": CATEGORY_CONFIDENCE_FLOOR,
        "category_confidence_threshold": CATEGORY_CONFIDENCE_THRESHOLD,
        "category_prototype_weight": CATEGORY_PROTOTYPE_WEIGHT,
        "category_relation_neighbors": CATEGORY_RELATION_NEIGHBORS,
        "related_top1_bonus_factor": RELATED_TOP1_BONUS_FACTOR,
        "related_top2_bonus_factor": RELATED_TOP2_BONUS_FACTOR,
        "unrelated_top1_bonus_factor": UNRELATED_TOP1_BONUS_FACTOR,
    }


def arguments():
    defaults = _serving_defaults()
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
    parser.add_argument(
        "--metadata-index-dir",
        type=Path,
        default=defaults["metadata_index_dir"],
        help="TF-IDF index built from the same catalog and row ordering.",
    )
    parser.add_argument("--expected-gallery-size", type=int, default=5000)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument("--query-split", choices=("validation", "test"), default="test")
    parser.add_argument("--image-weight", type=float, default=defaults["image_weight"])
    parser.add_argument(
        "--text-image-weight", type=float, default=defaults["text_image_weight"]
    )
    parser.add_argument(
        "--text-metadata-weight",
        type=float,
        default=defaults["text_metadata_weight"],
    )
    parser.add_argument(
        "--combined-metadata-weight",
        type=float,
        default=defaults["combined_metadata_weight"],
    )
    parser.add_argument(
        "--image-category-bonus",
        type=float,
        default=defaults["image_category_bonus"],
    )
    parser.add_argument(
        "--combined-category-bonus",
        type=float,
        default=defaults["combined_category_bonus"],
    )
    parser.add_argument(
        "--category-top-per-class",
        type=int,
        default=defaults["category_top_per_class"],
    )
    parser.add_argument(
        "--category-confidence-floor",
        type=float,
        default=defaults["category_confidence_floor"],
    )
    parser.add_argument(
        "--category-confidence-threshold",
        type=float,
        default=defaults["category_confidence_threshold"],
    )
    parser.add_argument(
        "--category-prototype-weight",
        type=float,
        default=defaults["category_prototype_weight"],
    )
    parser.add_argument(
        "--category-relation-neighbors",
        type=int,
        default=defaults["category_relation_neighbors"],
    )
    parser.add_argument(
        "--related-top1-bonus-factor",
        type=float,
        default=defaults["related_top1_bonus_factor"],
    )
    parser.add_argument(
        "--related-top2-bonus-factor",
        type=float,
        default=defaults["related_top2_bonus_factor"],
    )
    parser.add_argument(
        "--unrelated-top1-bonus-factor",
        type=float,
        default=defaults["unrelated_top1_bonus_factor"],
    )
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def _validate_arguments(args):
    if args.expected_gallery_size < 2 or args.batch_size < 1:
        raise ValueError("Gallery size and batch size must be positive.")
    if args.category_top_per_class < 1 or any(k < 1 for k in args.ks):
        raise ValueError("Top-per-class and every K value must be positive.")
    for name in (
        "image_weight",
        "text_image_weight",
        "text_metadata_weight",
        "combined_metadata_weight",
    ):
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
    validate_policy_settings(_policy_settings(args))


def _policy_settings(args):
    return CategoryPolicySettings(
        top_per_class=args.category_top_per_class,
        confidence_floor=args.category_confidence_floor,
        confidence_threshold=args.category_confidence_threshold,
        prototype_weight=args.category_prototype_weight,
        relation_neighbors=args.category_relation_neighbors,
        related_top1_factor=args.related_top1_bonus_factor,
        related_top2_factor=args.related_top2_bonus_factor,
        unrelated_top1_factor=args.unrelated_top1_bonus_factor,
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


def _metadata_metrics(rankings_by_system, metadata_scores, usable_queries, ks):
    eligible = np.flatnonzero(np.asarray(usable_queries, dtype=bool)).tolist()
    rows = []
    for system, rankings in rankings_by_system.items():
        for k in ks:
            means = []
            hit_rates = []
            for query_index in eligible:
                top = np.asarray(rankings[query_index], dtype=np.int64)[:k]
                if len(top):
                    selected = metadata_scores[query_index, top]
                    means.append(float(np.mean(selected)))
                    hit_rates.append(float(np.mean(selected > 0.0)))
            rows.append(
                {
                    "system": system,
                    "k": k,
                    "metadata_queries": len(eligible),
                    "mean_metadata_match": float(np.mean(means)) if means else None,
                    "metadata_hit_rate": (
                        float(np.mean(hit_rates)) if hit_rates else None
                    ),
                }
            )
    return rows


def _combined_category_route(
    explicit_categories,
    predicted_category,
    category_strength,
):
    """Mirror the deployed combined-query category precedence.

    One explicit text category is the clearest user instruction and therefore
    becomes a hard candidate pool. Multiple explicit categories are treated as
    ambiguous. Image evidence is used only when text does not name a category.
    """
    explicit = [str(value) for value in explicit_categories]
    if len(explicit) == 1:
        return explicit[0], "text"
    if len(explicit) > 1:
        return None, None
    if float(category_strength) > 0.0:
        return str(predicted_category), "image"
    return None, None


def _category_policy_rankings(
    scores,
    category_sources,
    applied_categories,
    category_evidence,
    gallery_categories,
    category_indices,
    excluded_positions,
    soft_bonus,
):
    adjusted = np.asarray(scores, dtype=np.float32).copy()
    dynamic_adjusted = apply_dynamic_category_bonus(
        adjusted,
        category_evidence,
        gallery_categories,
        soft_bonus,
    )
    candidate_groups = []
    for query_index, source in enumerate(category_sources):
        if source == "image":
            adjusted[query_index] = dynamic_adjusted[query_index]
            candidate_groups.append(None)
        elif source == "text":
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
    metadata = _metadata_metrics(
        {"base": rankings},
        np.asarray([[0.0, 1.0, 0.0], [1.0, 0.0, 0.5]], dtype=np.float32),
        np.asarray([True, False]),
        [1],
    )
    assert metadata[0]["metadata_queries"] == 1
    assert metadata[0]["mean_metadata_match"] == 1.0
    assert _combined_category_route(["Shirts"], "Tshirts", 1.0) == (
        "Shirts",
        "text",
    )
    assert _combined_category_route(["Shirts", "Tshirts"], "Shirts", 1.0) == (
        None,
        None,
    )
    assert _combined_category_route([], "Shirts", 0.5) == ("Shirts", "image")
    assert _combined_category_route([], "Shirts", 0.0) == (None, None)
    print("PASS: query routing, self exclusion and metadata diagnostics")


def main():
    args = arguments()
    if args.self_test:
        self_test()
        return
    _validate_arguments(args)

    sys.path.insert(0, str(PROJECT))
    from src.clip_data import file_sha256
    from src.clip_runtime import model_identity, same_model_identity
    from src.metadata_retrieval import MetadataRetriever
    from src.query_signals import (
        explicit_categories_from_text,
        metadata_refinement_text,
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
        "query_split": args.query_split,
        "query_count": len(queries),
        "gallery_size": len(gallery),
        "ks": ks,
        "image_weight": args.image_weight,
        "text_image_weight": args.text_image_weight,
        "text_metadata_weight": args.text_metadata_weight,
        "combined_metadata_weight": args.combined_metadata_weight,
        "metadata_index_dir": str(args.metadata_index_dir.resolve()),
        "image_category_bonus": args.image_category_bonus,
        "combined_category_bonus": args.combined_category_bonus,
        "category_top_per_class": args.category_top_per_class,
        "category_confidence_floor": args.category_confidence_floor,
        "category_confidence_threshold": args.category_confidence_threshold,
        "category_prototype_weight": args.category_prototype_weight,
        "category_relation_neighbors": args.category_relation_neighbors,
        "related_top1_bonus_factor": args.related_top1_bonus_factor,
        "related_top2_bonus_factor": args.related_top2_bonus_factor,
        "unrelated_top1_bonus_factor": args.unrelated_top1_bonus_factor,
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

    text_clip_scores = (
        args.text_image_weight * text_to_image_scores
        + (1.0 - args.text_image_weight) * text_to_text_scores
    )
    query_texts = queries.evaluation_text.astype(str).tolist()
    catalog_categories = sorted(
        gallery.category.fillna("").astype(str).unique().tolist()
    )
    explicit_text_categories = [
        explicit_categories_from_text(text, catalog_categories)
        for text in query_texts
    ]
    metadata_queries = [
        metadata_refinement_text(text, categories)
        for text, categories in zip(query_texts, explicit_text_categories)
    ]
    metadata_retriever = MetadataRetriever(
        args.metadata_index_dir,
        args.catalog_csv,
        len(gallery),
    )
    metadata_scores, metadata_usable = metadata_retriever.score_many(
        metadata_queries
    )

    text_scores = text_clip_scores.copy()
    if args.text_metadata_weight > 0.0 and np.any(metadata_usable):
        text_scores[metadata_usable] = (
            (1.0 - args.text_metadata_weight)
            * text_clip_scores[metadata_usable]
            + args.text_metadata_weight * metadata_scores[metadata_usable]
        )

    combined_base_scores = (
        args.image_weight * visual_scores
        + (1.0 - args.image_weight) * text_clip_scores
    )
    combined_metadata_scores = combined_base_scores.copy()

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

    category_prototypes, category_prototype_metadata = load_category_prototypes(
        catalog_categories,
        checkpoint,
        checkpoint_identity,
        args.query_text_cache_dir,
        args.device,
        args.batch_size,
        not args.no_cache,
    )
    category_evidence, relation_graph = build_category_evidence(
        query_image_vectors,
        visual_scores,
        catalog_categories,
        category_indices,
        category_prototypes,
        _policy_settings(args),
    )

    predicted_categories = np.asarray(
        [item["predicted"] for item in category_evidence], dtype=str
    )
    category_confidences = np.asarray(
        [item["confidence"] for item in category_evidence], dtype=np.float32
    )
    category_strengths = np.asarray(
        [item["strength"] for item in category_evidence], dtype=np.float32
    )
    text_categories = []
    routed_explicit_text_categories = []
    applied_categories = []
    category_sources = []
    for query_index, predicted in enumerate(predicted_categories):
        strength = category_strengths[query_index]
        explicit = explicit_text_categories[query_index]
        text_category = explicit[0] if len(explicit) == 1 else None
        applied_category, category_source = _combined_category_route(
            explicit,
            predicted,
            strength,
        )

        text_categories.append(text_category)
        routed_explicit_text_categories.append(explicit)
        applied_categories.append(applied_category)
        category_sources.append(category_source)

    # Match website serving: TF-IDF can refine only categories supported by
    # the image evidence (or one category explicitly written in text). Other
    # products retain their raw CLIP fusion score and are not hard-filtered.
    combined_metadata_support = np.zeros_like(
        combined_base_scores,
        dtype=np.float32,
    )
    for query_index, category_source in enumerate(category_sources):
        if not metadata_usable[query_index]:
            continue
        if category_source == "text" and applied_categories[query_index]:
            combined_metadata_support[query_index] = (
                gallery_categories == str(applied_categories[query_index])
            ).astype(np.float32)
        elif category_source == "image":
            for category, factor in category_evidence[query_index]["factors"].items():
                combined_metadata_support[query_index] += (
                    float(factor)
                    * (gallery_categories == str(category)).astype(np.float32)
                )

    if (
        args.combined_metadata_weight > 0.0
        and np.any(combined_metadata_support > 0.0)
    ):
        combined_metadata_scores = apply_category_gated_metadata_refinement(
            combined_base_scores,
            metadata_scores,
            combined_metadata_support,
            args.combined_metadata_weight,
        )

    image_deployed_scores = apply_dynamic_category_bonus(
        visual_scores,
        category_evidence,
        gallery_categories,
        args.image_category_bonus,
    )

    text_candidate_groups = [
        category_indices[value] if value is not None else None
        for value in text_categories
    ]
    text_deployed_rankings = _rank_rows(
        text_scores,
        query_positions,
        text_candidate_groups,
    )

    combined_base_rankings = _rank_rows(combined_base_scores, query_positions)
    combined_metadata_rankings = _rank_rows(
        combined_metadata_scores, query_positions
    )
    combined_category_rankings = _category_policy_rankings(
        combined_base_scores,
        category_sources,
        applied_categories,
        category_evidence,
        gallery_categories,
        category_indices,
        query_positions,
        args.combined_category_bonus,
    )
    combined_full_rankings = _category_policy_rankings(
        combined_metadata_scores,
        category_sources,
        applied_categories,
        category_evidence,
        gallery_categories,
        category_indices,
        query_positions,
        args.combined_category_bonus,
    )

    query_mode_rankings = {
        "image_only_deployed": _rank_rows(image_deployed_scores, query_positions),
        "text_only_deployed": text_deployed_rankings,
        "combined_full_deployed": combined_full_rankings,
    }
    combined_rankings = {
        "combined_base": combined_base_rankings,
        "combined_metadata": combined_metadata_rankings,
        "combined_category": combined_category_rankings,
        "combined_full": combined_full_rankings,
    }
    query_mode_specs = {
        "image_only_deployed": {
            "comparison_group": "query_mode",
            "role": "deployed image query",
            "deployed": True,
            "input": "image",
            "formula": "S_II + lambda_I*g(h)*w(c_i)",
        },
        "text_only_deployed": {
            "comparison_group": "query_mode",
            "role": "deployed text query",
            "deployed": True,
            "input": "text",
            "formula": (
                "CLIP text fusion + TF-IDF metadata; "
                "exact category => hard candidate pool"
            ),
        },
        "combined_full_deployed": {
            "comparison_group": "query_mode",
            "role": "deployed combined query",
            "deployed": True,
            "input": "image + text",
            "formula": (
                "raw cosine fusion + category-gated TF-IDF + explicit-text hard pool "
                "or dynamic image category bonus"
            ),
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
        "combined_metadata": {
            "comparison_group": "combined_ablation",
            "role": "TF-IDF metadata contribution",
            "deployed": False,
            "input": "image + text",
            "formula": "raw cosine fusion + category-gated TF-IDF refinement",
        },
        "combined_category": {
            "comparison_group": "combined_ablation",
            "role": "category contribution",
            "deployed": False,
            "input": "image + text",
            "formula": (
                "raw fusion + explicit-text hard pool or dynamic image "
                "category bonus"
            ),
        },
        "combined_full": {
            "comparison_group": "combined_ablation",
            "role": "full deployed ranking",
            "deployed": True,
            "input": "image + text",
            "formula": (
                "raw cosine fusion + category-gated TF-IDF + explicit-text hard pool "
                "or dynamic image category bonus"
            ),
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
    metadata_rows = _metadata_metrics(
        combined_rankings, metadata_scores, metadata_usable, ks
    )

    diagnostics = pd.DataFrame(
        {
            "product_id": query_ids,
            "actual_category": query_categories,
            **{
                f"image_{key}": value
                for key, value in evidence_columns(category_evidence).items()
            },
            "text_category": [value or "" for value in text_categories],
            "explicit_text_categories": [
                ";".join(values) for values in routed_explicit_text_categories
            ],
            "applied_category": [value or "" for value in applied_categories],
            "category_source": [value or "" for value in category_sources],
            "metadata_query": metadata_queries,
            "metadata_usable": metadata_usable,
        }
    )
    settings["diagnostics"] = {
        "queries_with_metadata": int(np.sum(metadata_usable)),
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
    pd.DataFrame(metadata_rows).to_csv(
        output / "metadata_comparison.csv", index=False
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
            "category_prototypes": category_prototype_metadata,
            "metadata_index": metadata_retriever.metadata,
        },
        "category_relation_graph": relation_graph,
        "query_mode_metrics": query_mode_metrics,
        "combined_ablation_metrics": combined_metrics,
        "metadata_metrics": metadata_rows,
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
        "## TF-IDF metadata diagnostics",
        "",
        "| System | K | Metadata queries | Mean metadata match | Metadata hit rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in metadata_rows:
        mean_match = row["mean_metadata_match"]
        hit_rate = row["metadata_hit_rate"]
        summary.append(
            f"| {row['system']} | {row['k']} | {row['metadata_queries']} | "
            f"{mean_match:.4f} | {hit_rate:.4f} |"
            if mean_match is not None and hit_rate is not None
            else (
                f"| {row['system']} | {row['k']} | "
                f"{row['metadata_queries']} | N/A | N/A |"
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
