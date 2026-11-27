"""Compare category-free, hard-filtered and soft category-aware image ranking."""

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
    _load_verified_finetuned_images,
    _markdown_table,
)
from .compare_modalities import DEFAULT_RUN, PROJECT, _prepare_output
from .evaluation import evaluate_query_gallery_category_rankings
from .category_runtime import (
    CategoryPolicySettings,
    apply_dynamic_category_bonus,
    build_category_evidence,
    evidence_columns,
    load_category_prototypes,
    validate_policy_settings,
)


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
    parser.add_argument("--category-cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--category-prototype-weight", type=float, default=1.0)
    parser.add_argument("--category-relation-neighbors", type=int, default=5)
    parser.add_argument("--related-top1-bonus-factor", type=float, default=0.50)
    parser.add_argument("--related-top2-bonus-factor", type=float, default=0.20)
    parser.add_argument("--unrelated-top1-bonus-factor", type=float, default=0.10)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def _validate_arguments(args):
    if (
        args.expected_gallery_size < 2
        or args.category_top_per_class < 1
        or args.batch_size < 1
    ):
        raise ValueError("Gallery size, batch size and top-per-class must be positive.")
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


def _normalized(matrix):
    values = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 0.0) or not np.isfinite(norms).all():
        raise ValueError("Embeddings must contain finite non-zero rows.")
    return values / norms


def _descending_rankings(scores):
    return np.argsort(-scores, axis=1, kind="stable")


def _hard_category_rankings(
    scores,
    predicted_categories,
    category_indices,
    excluded_positions,
):
    """Rank only products belonging to each query's predicted category."""
    values = np.asarray(scores, dtype=np.float32)
    predicted = np.asarray(predicted_categories, dtype=str)
    excluded = np.asarray(excluded_positions, dtype=np.int64)
    if values.ndim != 2 or len(values) != len(predicted) or len(values) != len(excluded):
        raise ValueError("Scores, predicted categories and exclusions must align.")

    rankings = []
    for query_index, category in enumerate(predicted):
        candidates = np.asarray(
            category_indices.get(str(category), np.empty(0, dtype=np.int64)),
            dtype=np.int64,
        )
        candidates = candidates[candidates != excluded[query_index]]
        order = np.argsort(-values[query_index, candidates], kind="stable")
        rankings.append(candidates[order])
    return rankings


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


def _hard_dynamic_recovery(
    hard_rankings,
    dynamic_rankings,
    predicted_categories,
    query_ids,
    query_categories,
    gallery_ids,
    gallery_categories,
    ks,
):
    """Paired Top-K diagnosis under the same-category relevance definition.

    Recovery@K is conditional on a wrong category prediction. The companion
    loss rate shows how often Dynamic loses a hit that Hard had, preventing a
    one-sided claim based only on rescued queries.
    """
    predicted = np.asarray(predicted_categories, dtype=str)
    query_ids = np.asarray(query_ids, dtype=str)
    actual = np.asarray(query_categories, dtype=str)
    gallery_ids = np.asarray(gallery_ids, dtype=str)
    gallery_categories = np.asarray(gallery_categories, dtype=str)
    if not (
        len(hard_rankings) == len(dynamic_rankings) == len(predicted)
        == len(query_ids) == len(actual)
    ):
        raise ValueError("Hard, Dynamic and query arrays must align.")
    if len(gallery_ids) != len(gallery_categories):
        raise ValueError("Gallery IDs and categories must align.")

    per_query = []
    for query_index, (hard, dynamic) in enumerate(
        zip(hard_rankings, dynamic_rankings)
    ):
        relevant = (
            (gallery_ids != query_ids[query_index])
            & (gallery_categories == actual[query_index])
        )
        if not actual[query_index].strip() or not np.any(relevant):
            continue
        hard = np.asarray(hard, dtype=np.int64)
        dynamic = np.asarray(dynamic, dtype=np.int64)
        if (np.any(hard < 0) or np.any(hard >= len(gallery_ids))
                or np.any(dynamic < 0) or np.any(dynamic >= len(gallery_ids))):
            raise ValueError("Ranking contains an invalid gallery index.")
        record = {
            "product_id": query_ids[query_index],
            "actual_category": actual[query_index],
            "predicted_category": predicted[query_index],
            "prediction_correct": bool(predicted[query_index] == actual[query_index]),
        }
        for k in ks:
            hard_hit = bool(np.any(relevant[hard[:k]]))
            dynamic_hit = bool(np.any(relevant[dynamic[:k]]))
            record[f"hard_hit_at_{k}"] = hard_hit
            record[f"dynamic_hit_at_{k}"] = dynamic_hit
            record[f"dynamic_rescue_at_{k}"] = not hard_hit and dynamic_hit
            record[f"dynamic_loss_at_{k}"] = hard_hit and not dynamic_hit
        per_query.append(record)

    details = pd.DataFrame(per_query)
    if details.empty:
        raise ValueError("No eligible same-category queries for recovery diagnostics.")
    rows = []
    for k in ks:
        wrong = details.loc[~details["prediction_correct"]]
        right = details.loc[details["prediction_correct"]]
        hard_hits = details[f"hard_hit_at_{k}"]
        dynamic_hits = details[f"dynamic_hit_at_{k}"]
        rescued = int(wrong[f"dynamic_rescue_at_{k}"].sum())
        lost = int(right[f"dynamic_loss_at_{k}"].sum())
        rows.append({
            "k": k,
            "eligible_queries": len(details),
            "wrong_prediction_queries": len(wrong),
            "right_prediction_queries": len(right),
            "hard_hits": int(hard_hits.sum()),
            "dynamic_hits": int(dynamic_hits.sum()),
            "rescued_queries": rescued,
            "recovery_at_k": rescued / len(wrong) if len(wrong) else 0.0,
            "lost_queries": lost,
            "loss_at_k": lost / len(right) if len(right) else 0.0,
            "net_hit_rate_change": (
                float(dynamic_hits.mean() - hard_hits.mean()) if len(details) else 0.0
            ),
        })
    return rows, details


def _recovery_markdown(rows):
    lines = [
        "| K | Eligible | Wrong prediction | Rescued | Recovery@K | Lost | Loss@K | Net Hit@K change |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['k']} | {row['eligible_queries']} | "
            f"{row['wrong_prediction_queries']} | {row['rescued_queries']} | "
            f"{row['recovery_at_k']:.4f} | {row['lost_queries']} | "
            f"{row['loss_at_k']:.4f} | {row['net_hit_rate_change']:.4f} |"
        )
    return lines


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
    hard = _hard_category_rankings(
        np.asarray([[0.95, 0.80, 0.70], [0.60, 0.90, 0.75]], dtype=np.float32),
        np.asarray(["shirt", "shoe"]),
        {
            "shirt": np.asarray([0, 1], dtype=np.int64),
            "shoe": np.asarray([2], dtype=np.int64),
        },
        np.asarray([0, 1], dtype=np.int64),
    )
    assert hard[0].tolist() == [1]
    assert hard[1].tolist() == [2]
    synthetic = [
        {
            "strength": 0.5,
            "factors": {"shirt": 0.5, "top": 0.2},
        }
    ]
    dynamic = apply_dynamic_category_bonus(
        np.asarray([[0.7, 0.6, 0.5]], dtype=np.float32),
        synthetic,
        np.asarray(["shirt", "top", "shoe"]),
        0.2,
    )
    assert np.allclose(dynamic, [[0.75, 0.62, 0.5]])
    recovery, cases = _hard_dynamic_recovery(
        [np.array([1]), np.array([2])],
        [np.array([0, 1, 2]), np.array([1, 2, 0])],
        ["shirt", "shoe"], ["query-a", "query-b"],
        ["shorts", "shoe"], ["a", "b", "c"],
        ["shorts", "shirt", "shoe"], [1, 2],
    )
    assert len(cases) == 2
    assert recovery[0]["rescued_queries"] == 1
    assert recovery[0]["lost_queries"] == 1
    assert recovery[0]["recovery_at_k"] == 1.0
    assert recovery[0]["loss_at_k"] == 1.0
    print("PASS: hard, legacy and dynamic category ranking")


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
        "category_prototype_weight": args.category_prototype_weight,
        "category_relation_neighbors": args.category_relation_neighbors,
        "related_top1_bonus_factor": args.related_top1_bonus_factor,
        "related_top2_bonus_factor": args.related_top2_bonus_factor,
        "unrelated_top1_bonus_factor": args.unrelated_top1_bonus_factor,
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
    categories = sorted(category_indices)
    category_prototypes, prototype_metadata = load_category_prototypes(
        categories,
        checkpoint,
        checkpoint_identity,
        args.category_cache_dir,
        args.device,
        args.batch_size,
        not args.no_cache,
    )
    evidence, relation_graph = build_category_evidence(
        query_vectors,
        visual_scores,
        categories,
        category_indices,
        category_prototypes,
        _policy_settings(args),
    )
    predicted_categories = np.asarray(
        [item["predicted"] for item in evidence], dtype=str
    )
    confidences = np.asarray(
        [item["confidence"] for item in evidence], dtype=np.float32
    )
    strengths = np.asarray(
        [item["strength"] for item in evidence], dtype=np.float32
    )
    threshold_scores = visual_scores.copy()
    weighted_scores = visual_scores.copy()

    for query_index, (scores, predicted, confidence) in enumerate(
        zip(visual_scores, predicted_categories, confidences)
    ):
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
    dynamic_scores = apply_dynamic_category_bonus(
        visual_scores,
        evidence,
        gallery_categories,
        args.category_bonus,
    )
    rankings = {
        "no_category": _descending_rankings(visual_scores),
        "hard_category_filter": _hard_category_rankings(
            visual_scores,
            predicted_categories,
            category_indices,
            query_positions,
        ),
        "threshold_soft_category": _descending_rankings(threshold_scores),
        "confidence_weighted_category": _descending_rankings(weighted_scores),
        "dynamic_relation_category": _descending_rankings(dynamic_scores),
    }
    specs = {
        "no_category": {
            "comparison_group": "category_aware_ablation",
            "role": "category-free baseline",
            "deployed": False,
            "input": "image",
            "formula": "S_II",
        },
        "hard_category_filter": {
            "comparison_group": "category_aware_ablation",
            "role": "default image-only hard predicted-category filter",
            "deployed": True,
            "input": "image",
            "formula": "rank only c_i=c_hat",
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
            "deployed": False,
            "input": "image",
            "formula": "S_II + lambda*g(h)*I[c_i=c_hat]",
        },
        "dynamic_relation_category": {
            "comparison_group": "category_aware_ablation",
            "role": "historical flexible ablation, not served by the website",
            "deployed": False,
            "input": "image",
            "formula": "S_II + lambda*g(h)*w(c_i)",
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
            "route_counts": {
                str(name): int(count)
                for name, count in pd.Series(
                    [item["route"] for item in evidence]
                ).value_counts().items()
            },
        }
    )

    recovery_rows, recovery_cases = _hard_dynamic_recovery(
        rankings["hard_category_filter"],
        rankings["dynamic_relation_category"],
        predicted_categories,
        query_ids,
        query_categories,
        gallery_ids,
        gallery_categories,
        ks,
    )

    output = _prepare_output(args.output_dir)
    predictions = pd.DataFrame(
        {
            "product_id": query_ids,
            "actual_category": query_categories,
            **evidence_columns(evidence),
            "threshold_applied": confidences
            >= args.category_confidence_threshold,
            "weighted_applied": strengths > 0.0,
            "correct": predicted_categories == query_categories,
        }
    )
    predictions.to_csv(output / "category_predictions.csv", index=False)
    pd.DataFrame(recovery_rows).to_csv(output / "hard_dynamic_recovery.csv", index=False)
    recovery_cases.to_csv(output / "hard_dynamic_queries.csv", index=False)
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
        "category_prototypes": prototype_metadata,
        "category_relation_graph": relation_graph,
        "category_prediction": classification,
        "hard_dynamic_recovery": recovery_rows,
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
        f"- Dynamic route counts: {classification['route_counts']}",
        "",
        "## Hard versus Dynamic: paired category recovery",
        "",
        "Recovery@K = wrong-category queries rescued by Dynamic / eligible wrong-category queries.",
        "Loss@K = correct-category queries where Hard hits but Dynamic misses / eligible correct-category queries.",
        "A hit means at least one other product of the query's true category appears in Top-K; it does not verify style or attributes.",
        "",
        *_recovery_markdown(recovery_rows),
    ]
    (output / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")

    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nCategory prediction diagnostics:")
    print(json.dumps(classification, indent=2))
    print(f"Saved category-aware comparison to: {output}")


if __name__ == "__main__":
    main()
