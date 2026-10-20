"""Compare category-free, thresholded and confidence-weighted image ranking."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .compare_full_gallery import (
    DEFAULT_CATALOG,
    DEFAULT_FINETUNED_IMAGES,
    _annotated_category_rows,
    _annotated_rows,
    _load_and_validate_data,
    _load_verified_finetuned_images,
    _markdown_table,
)
from .compare_modalities import DEFAULT_RUN, PROJECT, _prepare_output
from .evaluation import evaluate_query_gallery_category_rankings


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--catalog-csv", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument(
        "--image-embeddings",
        type=Path,
        default=DEFAULT_FINETUNED_IMAGES,
    )
    parser.add_argument("--expected-gallery-size", type=int, default=5000)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument("--category-top-per-class", type=int, default=3)
    parser.add_argument("--category-confidence-floor", type=float, default=0.50)
    parser.add_argument("--category-confidence-threshold", type=float, default=0.60)
    parser.add_argument("--category-bonus", type=float, default=0.05)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def _validate_arguments(args):
    if args.expected_gallery_size < 2 or args.category_top_per_class < 1:
        raise ValueError("Gallery size and top-per-class must be positive.")
    if any(k < 1 for k in args.ks):
        raise ValueError("Every K value must be positive.")
    if not (
        0.0
        <= args.category_confidence_floor
        < args.category_confidence_threshold
        <= 1.0
    ):
        raise ValueError(
            "Category thresholds must satisfy 0 <= floor < threshold <= 1."
        )
    if args.category_bonus < 0.0:
        raise ValueError("Category bonus cannot be negative.")


def _normalized(matrix):
    values = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 0.0) or not np.isfinite(norms).all():
        raise ValueError("Embeddings must contain finite non-zero rows.")
    return values / norms


def _descending_rankings(scores):
    return np.argsort(-scores, axis=1, kind="stable")


def _classification_diagnostics(actual, predicted):
    actual = np.asarray(actual, dtype=str)
    predicted = np.asarray(predicted, dtype=str)
    labels = sorted(set(actual) | set(predicted))
    rows = []
    for label in labels:
        true_positive = int(np.sum((actual == label) & (predicted == label)))
        false_positive = int(np.sum((actual != label) & (predicted == label)))
        false_negative = int(np.sum((actual == label) & (predicted != label)))
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        rows.append(
            {
                "category": label,
                "support": int(np.sum(actual == label)),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return rows, {
        "accuracy": float(np.mean(actual == predicted)),
        "macro_precision": float(np.mean([row["precision"] for row in rows])),
        "macro_recall": float(np.mean([row["recall"] for row in rows])),
        "macro_f1": float(np.mean([row["f1"] for row in rows])),
    }


def self_test():
    sys.path.insert(0, str(PROJECT))
    from src.category_ranking import (
        apply_confidence_weighted_category_bonus,
        category_confidence_strength,
    )

    assert category_confidence_strength(0.50, 0.50, 0.60) == 0.0
    assert np.isclose(category_confidence_strength(0.55, 0.50, 0.60), 0.5)
    adjusted, strength, effective = apply_confidence_weighted_category_bonus(
        np.array([0.70, 0.69]),
        np.array([1.0, 0.0]),
        0.55,
        0.05,
        0.50,
        0.60,
    )
    assert np.allclose(adjusted, [0.725, 0.69])
    assert np.isclose(strength, 0.5) and np.isclose(effective, 0.025)
    print("PASS: confidence-weighted category ranking")


def main():
    args = arguments()
    if args.self_test:
        self_test()
        return
    _validate_arguments(args)

    sys.path.insert(0, str(PROJECT))
    from src.category_ranking import (
        apply_confidence_weighted_category_bonus,
        category_confidence_strength,
        rank_category_candidates,
    )
    from src.clip_data import file_sha256
    from src.clip_runtime import model_identity, same_model_identity

    _, queries, gallery, query_positions, experiment, training, training_path = (
        _load_and_validate_data(args, file_sha256)
    )
    checkpoint = args.run_dir / "best"
    checkpoint_identity = model_identity(checkpoint)
    if not same_model_identity(training["checkpoint_identity"], checkpoint_identity):
        raise ValueError("The fine-tuned checkpoint differs from the training record.")

    settings = {
        "protocol": "held-out test queries against full catalog gallery",
        "run_dir": str(args.run_dir.resolve()),
        "catalog_csv": str(args.catalog_csv.resolve()),
        "image_embeddings": str(args.image_embeddings.resolve()),
        "dataset_sha256": experiment["dataset_sha256"],
        "training_json_sha256": file_sha256(training_path),
        "catalog_sha256": file_sha256(args.catalog_csv),
        "query_count": len(queries),
        "gallery_size": len(gallery),
        "ks": sorted(set(args.ks)),
        "category_top_per_class": args.category_top_per_class,
        "category_confidence_floor": args.category_confidence_floor,
        "category_confidence_threshold": args.category_confidence_threshold,
        "category_bonus": args.category_bonus,
        "self_match": "excluded before category prediction and ranking",
        "relevance": "same ground-truth category after ranking",
    }
    if args.dry_run:
        print(json.dumps(settings, indent=2, ensure_ascii=False))
        print("DRY RUN: category-aware 473-query x 5,000-gallery configuration is valid.")
        return

    embeddings, embedding_metadata = _load_verified_finetuned_images(
        args.image_embeddings,
        checkpoint_identity,
        args.catalog_csv,
        len(gallery),
    )
    gallery_vectors = _normalized(embeddings)
    query_vectors = gallery_vectors[query_positions]
    visual_scores = query_vectors @ gallery_vectors.T

    query_ids = queries.product_id.astype(str).to_numpy()
    query_categories = queries.category.fillna("").astype(str).to_numpy()
    gallery_ids = gallery.product_id.astype(str).to_numpy()
    gallery_categories = gallery.category.fillna("").astype(str).to_numpy()
    visual_scores[np.arange(len(queries)), query_positions] = -np.inf

    category_indices = {
        str(category): np.asarray(indices, dtype=np.int64)
        for category, indices in gallery.groupby(
            gallery["category"].astype(str), sort=True
        ).groups.items()
    }
    predicted_categories = []
    confidences = []
    strengths = []
    threshold_scores = visual_scores.copy()
    weighted_scores = visual_scores.copy()

    for query_index, scores in enumerate(visual_scores):
        ranked_categories, confidence = rank_category_candidates(
            scores,
            category_indices,
            args.category_top_per_class,
        )
        predicted = ranked_categories[0][0]
        matches = (gallery_categories == predicted).astype(np.float32)
        if confidence >= args.category_confidence_threshold:
            threshold_scores[query_index] = (
                scores + args.category_bonus * matches
            )
        adjusted, strength, _ = apply_confidence_weighted_category_bonus(
            scores,
            matches,
            confidence,
            args.category_bonus,
            args.category_confidence_floor,
            args.category_confidence_threshold,
        )
        weighted_scores[query_index] = adjusted
        predicted_categories.append(predicted)
        confidences.append(confidence)
        strengths.append(strength)

    predicted_categories = np.asarray(predicted_categories, dtype=str)
    confidences = np.asarray(confidences, dtype=np.float32)
    strengths = np.asarray(strengths, dtype=np.float32)
    rankings = {
        "no_category": _descending_rankings(visual_scores),
        "threshold_soft_category": _descending_rankings(threshold_scores),
        "confidence_weighted_category": _descending_rankings(weighted_scores),
    }
    specs = {
        "no_category": {
            "comparison_group": "category_aware_ablation",
            "role": "category-free baseline",
            "deployed": False,
            "input": "image",
            "formula": "S_II",
        },
        "threshold_soft_category": {
            "comparison_group": "category_aware_ablation",
            "role": "previous threshold strategy",
            "deployed": False,
            "input": "image",
            "formula": "S_II + lambda*I[h>=tau]*I[c_i=c_hat]",
        },
        "confidence_weighted_category": {
            "comparison_group": "category_aware_ablation",
            "role": "confidence-weighted serving",
            "deployed": True,
            "input": "image",
            "formula": "S_II + lambda*g(h)*I[c_i=c_hat]",
        },
    }
    ks = settings["ks"]
    metrics = {
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
    rows = _annotated_rows(
        specs,
        metrics,
        ks,
        "same-category product retrieval",
    )
    category_rows = _annotated_category_rows(
        specs,
        metrics,
        ks,
        "same-category product retrieval",
    )
    classification_rows, classification = _classification_diagnostics(
        query_categories,
        predicted_categories,
    )
    classification.update(
        {
            "threshold_coverage": float(
                np.mean(confidences >= args.category_confidence_threshold)
            ),
            "weighted_coverage": float(np.mean(strengths > 0.0)),
            "mean_category_strength": float(np.mean(strengths)),
        }
    )

    output = _prepare_output(args.output_dir)
    predictions = pd.DataFrame(
        {
            "product_id": query_ids,
            "actual_category": query_categories,
            "predicted_category": predicted_categories,
            "confidence": confidences,
            "category_strength": strengths,
            "threshold_applied": confidences
            >= args.category_confidence_threshold,
            "weighted_applied": strengths > 0.0,
            "correct": predicted_categories == query_categories,
        }
    )
    predictions.to_csv(output / "category_predictions.csv", index=False)
    pd.DataFrame(rows).to_csv(output / "comparison.csv", index=False)
    pd.DataFrame(category_rows).to_csv(
        output / "comparison_by_category.csv", index=False
    )
    pd.DataFrame(classification_rows).to_csv(
        output / "category_prediction_by_category.csv", index=False
    )
    (output / "experiment.json").write_text(
        json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "settings": settings,
        "embedding": embedding_metadata,
        "category_prediction": classification,
        "metrics": metrics,
    }
    (output / "comparison.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summary = [
        "# Category-aware ranking comparison",
        "",
        f"The benchmark used {len(queries)} held-out test queries against "
        f"{len(gallery)} catalog products.",
        "The query product was excluded before category prediction and ranking.",
        "",
        *_markdown_table("Ranking results", rows),
        "",
        "## Category prediction diagnostics",
        "",
        f"- Accuracy: {classification['accuracy']:.4f}",
        f"- Macro-F1: {classification['macro_f1']:.4f}",
        f"- Previous threshold coverage: {classification['threshold_coverage']:.4f}",
        f"- Confidence-weighted coverage: {classification['weighted_coverage']:.4f}",
        f"- Mean category strength: {classification['mean_category_strength']:.4f}",
    ]
    (output / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")

    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nCategory prediction diagnostics:")
    print(json.dumps(classification, indent=2))
    print(f"Saved category-aware comparison to: {output}")


if __name__ == "__main__":
    main()
