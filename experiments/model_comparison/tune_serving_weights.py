"""Select serving weights on validation data and persist runtime defaults.

The test split is deliberately unavailable in this command.  The selected
weights can be evaluated later with ``compare_system_pipeline`` on test data.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from .compare_full_gallery import (
    DEFAULT_CACHE,
    DEFAULT_CATALOG,
    DEFAULT_FINETUNED_IMAGES,
    _load_and_validate_data,
)
from .compare_modalities import DEFAULT_RUN, PROJECT, _cache_signature, _cached_embeddings, _prepare_output
from .compare_system_pipeline import (
    DEFAULT_FINETUNED_TEXT,
    _attribute_masks,
    _attribute_score_matrix,
    _load_runtime_embeddings,
    _normalized,
)
from .encoders import encode_clip_text
from .evaluation import evaluate_query_gallery_category_rankings


DEFAULT_RESULTS = Path(__file__).resolve().parent / "results"


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--catalog-csv", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--image-embeddings", type=Path, default=DEFAULT_FINETUNED_IMAGES)
    parser.add_argument("--text-embeddings", type=Path, default=DEFAULT_FINETUNED_TEXT)
    parser.add_argument("--query-text-cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--expected-gallery-size", type=int, default=5000)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--weights-output",
        type=Path,
        default=PROJECT / "retrieval_weights.json",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--objective-k", type=int, default=10)
    parser.add_argument("--max-macro-f1-drop", type=float, default=0.02)
    parser.add_argument("--min-combined-image-weight", type=float, default=0.20)
    parser.add_argument("--max-combined-image-weight", type=float, default=0.80)
    parser.add_argument("--category-top-per-class", type=int, default=3)
    parser.add_argument("--category-confidence-floor", type=float, default=0.50)
    parser.add_argument("--category-confidence-threshold", type=float, default=0.60)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def _validate_arguments(args):
    if args.expected_gallery_size < 2 or args.batch_size < 1:
        raise ValueError("Gallery size and batch size must be positive.")
    if args.objective_k < 1 or args.category_top_per_class < 1:
        raise ValueError("objective-k and category-top-per-class must be positive.")
    if not 0.0 <= args.max_macro_f1_drop <= 1.0:
        raise ValueError("max-macro-f1-drop must be between 0 and 1.")
    if not (
        0.0
        <= args.min_combined_image_weight
        < args.max_combined_image_weight
        <= 1.0
    ):
        raise ValueError(
            "Combined image-weight bounds must satisfy 0 <= minimum < maximum <= 1."
        )
    if not (
        0.0
        <= args.category_confidence_floor
        < args.category_confidence_threshold
        <= 1.0
    ):
        raise ValueError("Category thresholds must satisfy 0 <= floor < threshold <= 1.")


def _top_rank_rows(scores, excluded_positions, top_k, candidate_groups=None):
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
        count = min(int(top_k), len(candidates))
        if count == 0:
            rankings.append(np.asarray([], dtype=np.int64))
            continue
        candidate_scores = row[candidates]
        if count < len(candidates):
            selected = np.argpartition(-candidate_scores, count - 1)[:count]
        else:
            selected = np.arange(len(candidates), dtype=np.int64)
        selected = selected[
            np.argsort(-candidate_scores[selected], kind="stable")
        ]
        rankings.append(candidates[selected])
    return rankings


def _category_rankings(
    scores,
    category_sources,
    applied_categories,
    predicted_categories,
    category_strengths,
    gallery_categories,
    category_indices,
    excluded_positions,
    soft_bonus,
    top_k,
):
    values = np.asarray(scores, dtype=np.float32)
    adjusted = values.copy()
    candidate_groups = []
    for query_index, source in enumerate(category_sources):
        if source == "image":
            adjusted[query_index] += (
                soft_bonus
                * category_strengths[query_index]
                * (gallery_categories == predicted_categories[query_index])
            )
            candidate_groups.append(None)
        elif source in {"text", "text_image"}:
            candidate_groups.append(category_indices[applied_categories[query_index]])
        else:
            candidate_groups.append(None)
    return _top_rank_rows(adjusted, excluded_positions, top_k, candidate_groups)


def _metrics(rankings, query_ids, query_categories, gallery_ids, gallery_categories, k):
    result = evaluate_query_gallery_category_rankings(
        rankings,
        query_ids,
        query_categories,
        gallery_ids,
        gallery_categories,
        (k,),
    )
    return {
        "precision": result[f"precision@{k}"],
        "recall": result[f"recall@{k}"],
        "f1": result[f"f1@{k}"],
        "category_macro_precision": result["category_macro"][f"precision@{k}"],
        "category_macro_recall": result["category_macro"][f"recall@{k}"],
        "category_macro_f1": result["category_macro"][f"f1@{k}"],
        "evaluated_queries": result["evaluated_queries"],
        "skipped_queries": result["skipped_queries"],
    }


def _attribute_metrics(rankings, attribute_scores, query_attributes, k):
    eligible = [index for index, values in enumerate(query_attributes) if values]
    mean_matches = []
    all_matches = []
    for query_index in eligible:
        top = np.asarray(rankings[query_index], dtype=np.int64)[:k]
        if len(top):
            selected = attribute_scores[query_index, top]
            mean_matches.append(float(np.mean(selected)))
            all_matches.append(float(np.mean(selected >= 1.0)))
    return {
        "attribute_queries": len(eligible),
        "mean_attribute_match": float(np.mean(mean_matches)) if mean_matches else None,
        "all_attributes_match_rate": float(np.mean(all_matches)) if all_matches else None,
    }


def _best_retrieval_row(rows):
    return max(
        rows,
        key=lambda row: (
            row["category_macro_f1"],
            row["f1"],
            row["category_macro_precision"],
        ),
    )


def _best_combined_row(rows, maximum_drop):
    selectable = [row for row in rows if row.get("eligible_for_selection", True)]
    best_macro_f1 = max(row["category_macro_f1"] for row in selectable)
    eligible = [
        row
        for row in selectable
        if row["category_macro_f1"] >= best_macro_f1 - maximum_drop
    ]
    return max(
        eligible,
        key=lambda row: (
            row["all_attributes_match_rate"],
            row["mean_attribute_match"],
            row["category_macro_f1"],
            row["f1"],
        ),
    )


def _values_around(value, radius, step, lower=0.0, upper=1.0):
    start = max(lower, value - radius)
    stop = min(upper, value + radius)
    count = int(round((stop - start) / step))
    return sorted({round(start + index * step, 6) for index in range(count + 1)} | {round(value, 6)})


def self_test():
    rankings = _top_rank_rows(
        np.asarray([[1.0, 0.2, 0.9], [0.1, 1.0, 0.8]], dtype=np.float32),
        np.asarray([0, 1]),
        1,
    )
    assert rankings[0].tolist() == [2]
    assert rankings[1].tolist() == [2]
    rows = [
        {"category_macro_f1": 0.50, "f1": 0.40, "category_macro_precision": 0.6,
         "all_attributes_match_rate": 0.40, "mean_attribute_match": 0.50},
        {"category_macro_f1": 0.49, "f1": 0.39, "category_macro_precision": 0.6,
         "all_attributes_match_rate": 0.70, "mean_attribute_match": 0.80},
    ]
    assert _best_combined_row(rows, 0.02) is rows[1]
    bounded = _values_around(0.20, 0.20, 0.05, lower=0.20, upper=0.80)
    assert min(bounded) >= 0.20 and max(bounded) <= 0.80
    print("PASS: top-K ranking and constrained combined-weight selection")


def main():
    args = arguments()
    if args.self_test:
        self_test()
        return
    _validate_arguments(args)
    args.query_split = "validation"

    sys.path.insert(0, str(PROJECT))
    from src.category_ranking import category_confidence_strength, rank_category_candidates
    from src.clip_data import file_sha256
    from src.clip_runtime import model_identity, same_model_identity
    from src.query_signals import (
        ATTRIBUTE_RULES,
        extract_query_attributes,
    )

    _, queries, gallery, query_positions, experiment, training, training_path = (
        _load_and_validate_data(args, file_sha256)
    )
    checkpoint = args.run_dir / "best"
    checkpoint_identity = model_identity(checkpoint)
    if not same_model_identity(training["checkpoint_identity"], checkpoint_identity):
        raise ValueError("The fine-tuned checkpoint differs from training.json.")

    image_vectors, _ = _load_runtime_embeddings(
        args.image_embeddings,
        checkpoint_identity,
        args.catalog_csv,
        len(gallery),
        "runtime image embeddings",
    )
    runtime_text_vectors, _ = _load_runtime_embeddings(
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
    query_text_vectors, _ = _cached_embeddings(
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

    visual_scores = image_vectors[query_positions] @ image_vectors.T
    text_to_image_scores = query_text_vectors[query_positions] @ image_vectors.T
    text_to_text_scores = query_text_vectors[query_positions] @ runtime_text_vectors.T
    for scores in (visual_scores, text_to_image_scores, text_to_text_scores):
        scores[np.arange(len(queries)), query_positions] = -np.inf

    query_ids = queries.product_id.astype(str).to_numpy()
    query_categories = queries.category.fillna("").astype(str).to_numpy()
    gallery_ids = gallery.product_id.astype(str).to_numpy()
    gallery_categories = gallery.category.fillna("").astype(str).to_numpy()
    category_indices = {
        str(category): np.asarray(indices, dtype=np.int64)
        for category, indices in gallery.groupby(
            gallery["category"].astype(str), sort=True
        ).groups.items()
    }
    query_texts = queries.evaluation_text.astype(str).tolist()
    query_attributes = [extract_query_attributes(text) for text in query_texts]
    attribute_scores = _attribute_score_matrix(
        query_attributes,
        _attribute_masks(gallery, ATTRIBUTE_RULES),
        len(gallery),
    )

    predicted_categories = []
    category_strengths = []
    for scores in visual_scores:
        ranked_categories, confidence = rank_category_candidates(
            scores, category_indices, args.category_top_per_class
        )
        predicted = ranked_categories[0][0]
        strength = category_confidence_strength(
            confidence,
            args.category_confidence_floor,
            args.category_confidence_threshold,
        )
        predicted_categories.append(predicted)
        category_strengths.append(strength)

    predicted_categories = np.asarray(predicted_categories, dtype=str)
    category_strengths = np.asarray(category_strengths, dtype=np.float32)
    k = args.objective_k

    text_trials = []
    for text_image_weight in np.linspace(0.0, 1.0, 21):
        scores = (
            text_image_weight * text_to_image_scores
            + (1.0 - text_image_weight) * text_to_text_scores
        )
        rankings = _top_rank_rows(scores, query_positions, k)
        text_trials.append(
            {
                "text_image_weight": float(text_image_weight),
                **_metrics(
                    rankings,
                    query_ids,
                    query_categories,
                    gallery_ids,
                    gallery_categories,
                    k,
                ),
            }
        )
    selected_text = _best_retrieval_row(text_trials)
    text_image_weight = selected_text["text_image_weight"]

    image_trials = []
    for image_category_bonus in np.linspace(0.0, 0.20, 9):
        adjusted = visual_scores.copy()
        for query_index, predicted in enumerate(predicted_categories):
            adjusted[query_index] += (
                image_category_bonus
                * category_strengths[query_index]
                * (gallery_categories == predicted)
            )
        rankings = _top_rank_rows(adjusted, query_positions, k)
        image_trials.append(
            {
                "image_category_bonus": float(image_category_bonus),
                **_metrics(
                    rankings,
                    query_ids,
                    query_categories,
                    gallery_ids,
                    gallery_categories,
                    k,
                ),
            }
        )
    selected_image = _best_retrieval_row(image_trials)

    combined_query_indices = np.flatnonzero(
        np.asarray([bool(values) for values in query_attributes])
    )
    if not len(combined_query_indices):
        raise ValueError(
            "No validation queries contain supported attributes for combined tuning."
        )

    # A realistic combined query uses the image for the product identity and a
    # short text prompt for refinements such as colour, pattern, or style. It
    # must not reuse the complete product name/description because that gives
    # the text branch enough identity information to ignore the image.
    combined_query_attributes = [
        query_attributes[index] for index in combined_query_indices
    ]
    combined_query_texts = [
        " ".join(values) for values in combined_query_attributes
    ]
    combined_query_frame = queries.iloc[combined_query_indices][
        ["product_id", "image_sha256"]
    ].copy().reset_index(drop=True)
    combined_query_frame["evaluation_text"] = combined_query_texts
    combined_signature = _cache_signature(
        combined_query_frame,
        {
            "architecture": "CLIP text encoder",
            "identity": checkpoint_identity,
            "query_policy": "attribute-only validation text",
        },
    )
    combined_text_vectors, combined_text_metadata = _cached_embeddings(
        args.query_text_cache_dir,
        "clip_finetuned_validation_attribute_queries",
        combined_signature,
        lambda: encode_clip_text(
            combined_query_texts,
            checkpoint,
            args.device,
            args.batch_size,
        ),
        not args.no_cache,
    )
    combined_text_vectors = _normalized(
        combined_text_vectors, "combined attribute-text embeddings"
    )

    combined_positions = query_positions[combined_query_indices]
    combined_visual_scores = visual_scores[combined_query_indices]
    combined_text_to_image_scores = combined_text_vectors @ image_vectors.T
    combined_text_to_text_scores = combined_text_vectors @ runtime_text_vectors.T
    for scores in (combined_text_to_image_scores, combined_text_to_text_scores):
        scores[np.arange(len(combined_query_indices)), combined_positions] = -np.inf
    combined_text_scores = (
        text_image_weight * combined_text_to_image_scores
        + (1.0 - text_image_weight) * combined_text_to_text_scores
    )
    combined_attribute_scores = attribute_scores[combined_query_indices]
    combined_query_ids = query_ids[combined_query_indices]
    combined_query_categories = query_categories[combined_query_indices]
    combined_predicted_categories = predicted_categories[combined_query_indices]
    combined_category_strengths = category_strengths[combined_query_indices]
    combined_category_sources = np.asarray(
        ["image" if strength > 0.0 else None for strength in combined_category_strengths],
        dtype=object,
    )
    combined_applied_categories = np.asarray(
        [
            predicted if source == "image" else None
            for predicted, source in zip(
                combined_predicted_categories, combined_category_sources
            )
        ],
        dtype=object,
    )

    combined_trials = []
    attempted = set()

    def evaluate_combined(
        image_weight,
        attribute_weight,
        category_bonus,
        stage,
        eligible_for_selection=True,
    ):
        key = (round(image_weight, 6), round(attribute_weight, 6), round(category_bonus, 6))
        if key in attempted:
            return
        attempted.add(key)
        scores = (
            image_weight * combined_visual_scores
            + (1.0 - image_weight) * combined_text_scores
        )
        if attribute_weight > 0.0:
            scores = (
                (1.0 - attribute_weight) * scores
                + attribute_weight * combined_attribute_scores
            )
        rankings = _category_rankings(
            scores,
            combined_category_sources,
            combined_applied_categories,
            combined_predicted_categories,
            combined_category_strengths,
            gallery_categories,
            category_indices,
            combined_positions,
            category_bonus,
            k,
        )
        combined_trials.append(
            {
                "stage": stage,
                "image_weight": float(image_weight),
                "attribute_weight": float(attribute_weight),
                "combined_category_weight": float(category_bonus),
                "eligible_for_selection": bool(eligible_for_selection),
                **_metrics(
                    rankings,
                    combined_query_ids,
                    combined_query_categories,
                    gallery_ids,
                    gallery_categories,
                    k,
                ),
                **_attribute_metrics(
                    rankings,
                    combined_attribute_scores,
                    combined_query_attributes,
                    k,
                ),
            }
        )

    evaluate_combined(
        0.50,
        0.25,
        0.10,
        stage="previous_default",
        eligible_for_selection=(
            args.min_combined_image_weight
            <= 0.50
            <= args.max_combined_image_weight
        ),
    )
    for values in product(
        np.linspace(
            args.min_combined_image_weight,
            args.max_combined_image_weight,
            7,
        ),
        np.linspace(0.0, 0.5, 6),
        np.linspace(0.0, 0.20, 5),
    ):
        evaluate_combined(*values, stage="coarse")
    coarse_best = _best_combined_row(combined_trials, args.max_macro_f1_drop)
    for values in product(
        _values_around(
            coarse_best["image_weight"],
            0.20,
            0.05,
            lower=args.min_combined_image_weight,
            upper=args.max_combined_image_weight,
        ),
        _values_around(coarse_best["attribute_weight"], 0.10, 0.05, upper=0.60),
        _values_around(coarse_best["combined_category_weight"], 0.05, 0.025, upper=0.30),
    ):
        evaluate_combined(*values, stage="refined")
    selected_combined = _best_combined_row(
        combined_trials, args.max_macro_f1_drop
    )

    for rows, selected in (
        (text_trials, selected_text),
        (image_trials, selected_image),
        (combined_trials, selected_combined),
    ):
        for row in rows:
            row["selected"] = row is selected

    weights = {
        "TEXT_RETRIEVAL_IMAGE_WEIGHT": round(text_image_weight, 6),
        "COMBINED_IMAGE_WEIGHT": round(selected_combined["image_weight"], 6),
        "COMBINED_ATTRIBUTE_WEIGHT": round(selected_combined["attribute_weight"], 6),
        "COMBINED_CATEGORY_WEIGHT": round(
            selected_combined["combined_category_weight"], 6
        ),
        "CATEGORY_BONUS": round(selected_image["image_category_bonus"], 6),
    }
    report = {
        "version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_split": "validation",
        "objective_k": k,
        "selection_policy": {
            "text_and_image_only": "maximize category-macro F1, then query-macro F1",
            "combined": (
                "use attribute-only validation text, keep the image weight within "
                f"[{args.min_combined_image_weight:.2f}, "
                f"{args.max_combined_image_weight:.2f}], and maximize attribute "
                "match among candidates within "
                f"{args.max_macro_f1_drop:.4f} absolute category-macro F1 of the best"
            ),
        },
        "combined_query_policy": (
            "attribute-only text; the image supplies product and category context"
        ),
        "combined_image_weight_bounds": {
            "minimum": args.min_combined_image_weight,
            "maximum": args.max_combined_image_weight,
        },
        "weights": weights,
        "selected_metrics": {
            "text_only": selected_text,
            "image_only": selected_image,
            "combined": selected_combined,
        },
        "provenance": {
            "run_dir": str(args.run_dir.resolve()),
            "catalog_csv": str(args.catalog_csv.resolve()),
            "dataset_sha256": experiment["dataset_sha256"],
            "training_json_sha256": file_sha256(training_path),
            "catalog_sha256": file_sha256(args.catalog_csv),
            "validation_queries": len(queries),
            "combined_validation_queries": len(combined_query_indices),
            "gallery_size": len(gallery),
            "combined_query_text_embeddings": combined_text_metadata,
        },
    }

    output = _prepare_output(args.output_dir)
    pd.DataFrame(text_trials).to_csv(output / "text_weight_trials.csv", index=False)
    pd.DataFrame(image_trials).to_csv(output / "image_weight_trials.csv", index=False)
    pd.DataFrame(combined_trials).to_csv(
        output / "combined_weight_trials.csv", index=False
    )
    (output / "selected_weights.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    args.weights_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.weights_output.with_suffix(args.weights_output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(args.weights_output)

    print(json.dumps(weights, indent=2))
    print(f"Validation trials: {output}")
    print(f"Runtime defaults: {args.weights_output.resolve()}")
    print("The test split was not used. Run compare_system_pipeline next for final evaluation.")


if __name__ == "__main__":
    main()
