"""Project-local helpers for selecting serving weights on validation data.

This module deliberately depends only on ``project/src``.  Research scripts in
the repository-level ``experiments`` directory may evaluate the resulting
artifacts, but the operational pipeline never imports them.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from src.category_ranking import (
    build_mutual_category_neighbors,
    categories_are_mutual_neighbors,
    category_confidence_strength,
    combine_category_evidence,
    rank_category_candidates,
    rank_category_score_mapping,
)
from src.clip_data import file_sha256, load_experiment_data
from src.clip_runtime import model_identity, same_model_identity, validate_embedding_metadata
from src.config import BASE_DIR


DEFAULT_RUN = BASE_DIR / "runs" / "clip_finetune"
DEFAULT_CATALOG = BASE_DIR / "data" / "processed" / "products.csv"
DEFAULT_IMAGE_EMBEDDINGS = BASE_DIR / "embeddings" / "image_embeddings.npy"
DEFAULT_TEXT_EMBEDDINGS = BASE_DIR / "embeddings" / "text_embeddings.npy"
DEFAULT_CACHE = BASE_DIR / "validation" / "cache"
DEFAULT_RESULTS = BASE_DIR / "validation"


def build_evaluation_text(row):
    """Build a query without inserting the target category label."""
    parts = []
    for label, column in (("Product", "product_name"), ("Description", "description")):
        value = row.get(column, "")
        if pd.notna(value) and str(value).strip():
            parts.append(f"{label}: {str(value).strip()}")
    return ". ".join(parts) or "Fashion product"


def normalized(matrix, label):
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError(f"{label} must be a finite two-dimensional matrix.")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 0.0):
        raise ValueError(f"{label} contains a zero vector.")
    return values / norms


def _hash_payload(payload):
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def cache_signature(products, model_identity_value):
    rows = [
        [str(row.product_id), str(row.image_sha256), str(row.evaluation_text)]
        for row in products.itertuples()
    ]
    return _hash_payload({"model": model_identity_value, "rows": rows})


def cached_embeddings(cache_dir, name, signature, producer, use_cache=True):
    cache_dir = Path(cache_dir)
    array_path = cache_dir / f"{name}.npy"
    metadata_path = cache_dir / f"{name}.json"
    if use_cache and array_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("signature") == signature:
            values = np.load(array_path, allow_pickle=False)
            if values.ndim == 2 and len(values) == metadata.get("rows"):
                print(f"Using cached validation embeddings: {array_path}", flush=True)
                return values, metadata.get("model", {})

    started = time.perf_counter()
    values, metadata = producer()
    metadata = {**metadata, "encoding_seconds": time.perf_counter() - started}
    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(array_path, values, allow_pickle=False)
        metadata_path.write_text(
            json.dumps(
                {
                    "signature": signature,
                    "rows": len(values),
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "model": metadata,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    return values, metadata


def prepare_output(requested):
    output = Path(requested) if requested else (
        DEFAULT_RESULTS / f"weight_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def portable_path(path):
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(BASE_DIR.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def encode_texts(model, processor, texts, device, batch_size):
    batches = []
    maximum = model.config.text_config.max_position_embeddings
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            inputs = processor(
                text=list(texts[start:start + batch_size]),
                padding=True,
                truncation=True,
                max_length=maximum,
                return_tensors="pt",
            ).to(device)
            output = model.text_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                return_dict=True,
            )
            values = model.text_projection(output.pooler_output)
            values = torch.nn.functional.normalize(values, dim=-1)
            batches.append(values.cpu().numpy())
    embeddings = np.concatenate(batches).astype(np.float32)
    return embeddings, {
        "architecture": "CLIP text encoder",
        "embedding_dimension": int(embeddings.shape[1]),
    }


def load_and_validate_data(args):
    products, experiment = load_experiment_data(args.run_dir)
    training_path = Path(args.run_dir) / "training.json"
    training = json.loads(training_path.read_text(encoding="utf-8"))
    if training.get("status") != "complete":
        raise ValueError("The selected CLIP training run is not complete.")
    if training.get("dataset_sha256") != experiment.get("dataset_sha256"):
        raise ValueError("training.json and experiment.json refer to different datasets.")

    gallery = pd.read_csv(args.catalog_csv, dtype={"product_id": str})
    required = {
        "product_id", "image_reference", "product_name", "description",
        "category", "image_path", "image_sha256",
    }
    if not required.issubset(gallery.columns):
        raise ValueError(f"Catalog must contain: {sorted(required)}")
    if args.expected_gallery_size is not None and len(gallery) != args.expected_gallery_size:
        raise ValueError(
            f"Expected {args.expected_gallery_size} gallery products, found {len(gallery)}."
        )
    if len(gallery) < 2 or gallery.product_id.duplicated().any():
        raise ValueError("The catalog needs at least two unique product IDs.")
    mandatory = [
        "product_id", "image_reference", "product_name", "category",
        "image_path", "image_sha256",
    ]
    if gallery[mandatory].isna().any().any():
        raise ValueError("Catalog identity, category and image fields cannot be missing.")

    gallery["evaluation_text"] = gallery.apply(build_evaluation_text, axis=1)
    saved = products.set_index(products.product_id.astype(str), drop=False)
    catalog = gallery.set_index(gallery.product_id.astype(str), drop=False)
    if set(saved.index) != set(catalog.index):
        raise ValueError("The catalog is not the exact catalog used by this training run.")
    for product_id, row in saved.iterrows():
        candidate = catalog.loc[product_id]
        if (
            str(row.image_sha256) != str(candidate.image_sha256)
            or str(row.category) != str(candidate.category)
            or build_evaluation_text(row) != str(candidate.evaluation_text)
        ):
            raise ValueError(f"Catalog content changed for product: {product_id}")

    queries = products.loc[products.split == "validation"].copy().reset_index(drop=True)
    if queries.empty:
        raise ValueError("No products were assigned to the validation split.")
    queries["evaluation_text"] = queries.apply(build_evaluation_text, axis=1)
    positions = {value: index for index, value in enumerate(gallery.product_id.astype(str))}
    query_positions = np.asarray(
        [positions[str(product_id)] for product_id in queries.product_id],
        dtype=np.int64,
    )
    return products, queries, gallery, query_positions, experiment, training, training_path


def load_runtime_embeddings(path, identity, catalog_csv, rows, label):
    path = Path(path)
    validate_embedding_metadata(path, identity, catalog_csv)
    values = np.load(path, allow_pickle=False)
    if values.ndim != 2 or len(values) != rows or not np.isfinite(values).all():
        raise ValueError(f"Invalid {label}: {path}")
    return normalized(values, label), {
        "source": portable_path(path),
        "rows": int(values.shape[0]),
        "dimension": int(values.shape[1]),
    }


def rank_rows(scores, excluded_positions, top_k, candidate_groups=None):
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
        selected = selected[np.argsort(-candidate_scores[selected], kind="stable")]
        rankings.append(candidates[selected])
    return rankings


def _f1(precision, recall):
    return 0.0 if precision + recall == 0.0 else (
        2.0 * precision * recall / (precision + recall)
    )


def evaluate_category_rankings(
    rankings,
    query_ids,
    query_categories,
    gallery_ids,
    gallery_categories,
    k,
):
    query_ids = np.asarray(query_ids, dtype=str)
    query_categories = np.asarray(query_categories, dtype=str)
    gallery_ids = np.asarray(gallery_ids, dtype=str)
    gallery_categories = np.asarray(gallery_categories, dtype=str)
    totals = {name: 0.0 for name in ("precision", "recall", "f1")}
    category_totals = defaultdict(lambda: {name: 0.0 for name in totals})
    category_counts = defaultdict(int)
    evaluated = 0
    for index, ranking in enumerate(rankings):
        category = query_categories[index].strip()
        relevant = np.flatnonzero(
            (gallery_ids != query_ids[index]) & (gallery_categories == category)
        )
        if not category or not len(relevant):
            continue
        relevant_set = set(int(value) for value in relevant)
        top = np.asarray(ranking, dtype=np.int64)[:k]
        hits = sum(int(value) in relevant_set for value in top)
        precision = hits / len(top) if len(top) else 0.0
        recall = hits / len(relevant)
        values = {"precision": precision, "recall": recall, "f1": _f1(precision, recall)}
        for name, value in values.items():
            totals[name] += value
            category_totals[category][name] += value
        category_counts[category] += 1
        evaluated += 1

    result = {
        name: totals[name] / evaluated if evaluated else 0.0
        for name in totals
    }
    for name in totals:
        values = [
            category_totals[category][name] / count
            for category, count in category_counts.items()
            if count
        ]
        result[f"category_macro_{name}"] = float(np.mean(values)) if values else 0.0
    result["evaluated_queries"] = evaluated
    result["skipped_queries"] = len(query_ids) - evaluated
    return result


def attribute_masks(products, rules):
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
            matched |= search_text.str.contains(pattern, regex=True, na=False).to_numpy()
        masks[name] = matched.astype(np.float32)
    return masks


def attribute_score_matrix(query_attributes, masks, gallery_size):
    scores = np.zeros((len(query_attributes), gallery_size), dtype=np.float32)
    for query_index, attributes in enumerate(query_attributes):
        if attributes:
            scores[query_index] = np.mean(
                np.stack([masks[name] for name in attributes]),
                axis=0,
            )
    return scores


@dataclass(frozen=True)
class CategoryPolicySettings:
    top_per_class: int
    confidence_floor: float
    confidence_threshold: float
    prototype_weight: float = 1.0
    relation_neighbors: int = 5
    related_top1_factor: float = 0.50
    related_top2_factor: float = 0.20
    unrelated_top1_factor: float = 0.10


def validate_policy_settings(settings):
    if settings.top_per_class < 1 or settings.relation_neighbors < 1:
        raise ValueError("Category neighbour counts must be positive.")
    if not 0.0 <= settings.confidence_floor < settings.confidence_threshold <= 1.0:
        raise ValueError("Category thresholds must satisfy 0 <= floor < threshold <= 1.")
    if settings.prototype_weight < 0.0:
        raise ValueError("Category prototype weight cannot be negative.")
    factors = (
        settings.related_top1_factor,
        settings.related_top2_factor,
        settings.unrelated_top1_factor,
    )
    if any(value < 0.0 or value > 1.0 for value in factors):
        raise ValueError("Category bonus factors must be between 0 and 1.")
    if settings.related_top2_factor > settings.related_top1_factor:
        raise ValueError("The related Top-2 factor cannot exceed the Top-1 factor.")


def load_category_prototypes(
    categories,
    model,
    processor,
    identity,
    cache_dir,
    device,
    batch_size,
    use_cache=True,
):
    labels = [str(value) for value in categories]
    prompts = [f"a catalog product photo of {category}" for category in labels]
    frame = pd.DataFrame(
        {
            "product_id": [f"category::{value}" for value in labels],
            "image_sha256": [""] * len(labels),
            "evaluation_text": prompts,
        }
    )
    signature = cache_signature(
        frame,
        {
            "architecture": "CLIP category text prototype",
            "identity": identity,
            "prompt": "a catalog product photo of {category}",
        },
    )
    values, metadata = cached_embeddings(
        cache_dir,
        "clip_category_prototypes",
        signature,
        lambda: encode_texts(model, processor, prompts, device, batch_size),
        use_cache,
    )
    return normalized(values, "category prototype embeddings"), metadata


def build_category_evidence(
    query_vectors,
    visual_scores,
    categories,
    category_indices,
    category_prototypes,
    settings,
):
    validate_policy_settings(settings)
    labels = [str(value) for value in categories]
    relations = build_mutual_category_neighbors(
        labels,
        category_prototypes,
        settings.relation_neighbors,
    )
    evidence = []
    for query_vector, visual_row in zip(query_vectors, visual_scores):
        visual_ranked, visual_confidence = rank_category_candidates(
            visual_row,
            category_indices,
            settings.top_per_class,
        )
        prototype_scores = dict(
            zip(labels, np.asarray(category_prototypes @ query_vector, dtype=np.float32))
        )
        prototype_ranked, prototype_confidence = rank_category_score_mapping(
            prototype_scores
        )
        ranked, combined_confidence = combine_category_evidence(
            visual_ranked,
            prototype_scores,
            settings.prototype_weight,
        )
        predicted = ranked[0][0]
        if predicted == prototype_ranked[0][0]:
            confidence, source = prototype_confidence, "prototype"
        elif predicted == visual_ranked[0][0]:
            confidence, source = visual_confidence, "visual_neighbors"
        else:
            confidence, source = combined_confidence, "combined"
        strength = category_confidence_strength(
            confidence,
            settings.confidence_floor,
            settings.confidence_threshold,
        )
        leading = [category for category, _ in ranked[:2]]
        related = (
            len(leading) == 2
            and categories_are_mutual_neighbors(leading[0], leading[1], relations)
        )
        if strength <= 0.0:
            route, factors = "no_category", {}
        elif confidence < settings.confidence_threshold and related:
            route = "ambiguous_related_top2"
            factors = {
                leading[0]: settings.related_top1_factor,
                leading[1]: settings.related_top2_factor,
            }
        elif confidence < settings.confidence_threshold:
            route = "ambiguous_unrelated_top1"
            factors = {leading[0]: settings.unrelated_top1_factor}
        else:
            route = "confident_top1"
            factors = {leading[0]: 1.0}
        evidence.append(
            {
                "predicted": predicted,
                "second_category": leading[1] if len(leading) > 1 else None,
                "confidence": float(confidence),
                "confidence_source": source,
                "strength": float(strength),
                "route": route,
                "related_top2": bool(related),
                "factors": factors,
            }
        )
    return evidence, relations


def apply_dynamic_category_bonus(scores, evidence, gallery_categories, bonus):
    adjusted = np.asarray(scores, dtype=np.float32).copy()
    labels = np.asarray(gallery_categories, dtype=str)
    for query_index, item in enumerate(evidence):
        for category, factor in item["factors"].items():
            adjusted[query_index] += (
                float(bonus)
                * float(item["strength"])
                * float(factor)
                * (labels == str(category))
            )
    return adjusted
