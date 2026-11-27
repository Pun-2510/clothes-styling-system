"""Evaluate frozen OpenCLIP or SigLIP on the existing three-seed protocol."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .compare_full_gallery import (
    _annotated_category_rows, _annotated_rows, _load_and_validate_data,
)
from .compare_modalities import DEFAULT_RUN, _cache_signature, _cached_embeddings, _prepare_output
from .encoders import _normalize
from .evaluation import (
    evaluate_query_gallery_category_rankings,
    evaluate_query_gallery_identity_rankings,
    rank_queries_by_cosine,
)
from .fusion import clip_query_gallery_score_fusion


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-family", choices=("openclip", "siglip"), required=True)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--catalog-csv", type=Path, required=True)
    parser.add_argument("--expected-gallery-size", type=int, default=5000)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--image-weight", type=float, default=0.5)
    parser.add_argument("--text-image-weight", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _encode_openclip(gallery, device, batch_size):
    import open_clip
    import torch
    from PIL import Image

    architecture, weights = "ViT-B-32", "laion2b_s34b_b79k"
    model, _, preprocess = open_clip.create_model_and_transforms(
        architecture, pretrained=weights
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(architecture)
    text_batches, image_batches = [], []
    texts = gallery.evaluation_text.astype(str).tolist()
    paths = gallery.image_path.astype(str).tolist()
    with torch.inference_mode():
        for start in range(0, len(gallery), batch_size):
            tokens = tokenizer(texts[start:start + batch_size]).to(device)
            text_batches.append(model.encode_text(tokens).float().cpu().numpy())
            images = []
            for path in paths[start:start + batch_size]:
                with Image.open(path) as image:
                    images.append(preprocess(image.convert("RGB")))
            image_batches.append(
                model.encode_image(torch.stack(images).to(device)).float().cpu().numpy()
            )
    import numpy as np
    image_vectors = _normalize(np.concatenate(image_batches))
    text_vectors = _normalize(np.concatenate(text_batches))
    metadata = {
        "architecture": architecture,
        "weights": weights,
        "source": "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
        "parameters": sum(p.numel() for p in model.parameters()),
        "embedding_dimension": int(image_vectors.shape[1]),
        "category_in_evaluation_text": False,
    }
    return image_vectors, text_vectors, metadata


def _encode_siglip(gallery, device, batch_size, revision):
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoProcessor, SiglipModel

    source = "google/siglip-base-patch16-224"
    processor = AutoProcessor.from_pretrained(source, revision=revision)
    model = SiglipModel.from_pretrained(source, revision=revision).to(device).eval()
    max_length = model.config.text_config.max_position_embeddings
    text_batches, image_batches = [], []
    texts = gallery.evaluation_text.astype(str).tolist()
    paths = gallery.image_path.astype(str).tolist()
    with torch.inference_mode():
        for start in range(0, len(gallery), batch_size):
            text_inputs = processor(
                text=texts[start:start + batch_size],
                padding="max_length", truncation=True, max_length=max_length,
                return_tensors="pt",
            ).to(device)
            text_output = model.get_text_features(**text_inputs)
            text_features = getattr(text_output, "pooler_output", text_output)
            text_batches.append(text_features.float().cpu().numpy())
            images = []
            for path in paths[start:start + batch_size]:
                with Image.open(path) as image:
                    images.append(image.convert("RGB").copy())
            image_inputs = processor(images=images, return_tensors="pt").to(device)
            image_output = model.get_image_features(**image_inputs)
            image_features = getattr(image_output, "pooler_output", image_output)
            image_batches.append(image_features.float().cpu().numpy())
    image_vectors = _normalize(np.concatenate(image_batches))
    text_vectors = _normalize(np.concatenate(text_batches))
    metadata = {
        "architecture": "SigLIP ViT-B/16",
        "source": source,
        "revision": revision,
        "parameters": sum(p.numel() for p in model.parameters()),
        "embedding_dimension": int(image_vectors.shape[1]),
        "category_in_evaluation_text": False,
    }
    return image_vectors, text_vectors, metadata


def main():
    args = arguments()
    if args.expected_gallery_size < 2 or args.batch_size < 1 or not args.ks or min(args.ks) < 1:
        raise ValueError("Gallery size, batch size, and K must be positive.")
    if not 0 <= args.image_weight <= 1 or not 0 <= args.text_image_weight <= 1:
        raise ValueError("Fusion weights must be between zero and one.")
    import sys
    from .compare_modalities import PROJECT
    sys.path.insert(0, str(PROJECT))
    from src.clip_data import file_sha256

    _, queries, gallery, positions, experiment, _, training_path = (
        _load_and_validate_data(args, file_sha256)
    )
    source = (
        "laion/CLIP-ViT-B-32-laion2B-s34B-b79K" if args.model_family == "openclip"
        else "google/siglip-base-patch16-224"
    )
    settings = {
        "model_family": args.model_family,
        "model_source": source,
        "run_dir": str(args.run_dir.resolve()),
        "catalog_csv": str(args.catalog_csv.resolve()),
        "catalog_sha256": file_sha256(args.catalog_csv),
        "dataset_sha256": experiment["dataset_sha256"],
        "training_json_sha256": file_sha256(training_path),
        "gallery_size": len(gallery), "query_count": len(queries),
        "query_split": "test", "ks": sorted(set(args.ks)),
        "image_weight": args.image_weight,
        "text_image_weight": args.text_image_weight,
        "relevance": "same category for modality rows; exact product ID for alignment rows",
        "category_in_encoder_text": False,
        "category_policy": "none (frozen raw baseline)",
    }
    if args.dry_run:
        print(json.dumps(settings, indent=2))
        return
    from huggingface_hub import HfApi
    revision = HfApi().model_info(source).sha
    settings["revision_resolved"] = revision
    if args.model_family == "openclip":
        settings["pretrained_tag"] = "laion2b_s34b_b79k"
        settings["revision_note"] = "OpenCLIP loads the named pretrained tag; the Hugging Face revision is recorded for provenance but is not pinned by the tag."
    identity = {"source": source, "revision": revision, "family": args.model_family,
                "category_in_text": False}
    signature = _cache_signature(gallery, identity)
    computed = {}

    def get_vectors():
        if not computed:
            vectors = (
                _encode_openclip(gallery, args.device, args.batch_size)
                if args.model_family == "openclip" else
                _encode_siglip(gallery, args.device, args.batch_size, revision)
            )
            computed["image"], computed["text"], computed["meta"] = vectors
        return computed

    image_vectors, image_meta = _cached_embeddings(
        args.cache_dir, args.model_family + "_image", signature,
        lambda: (get_vectors()["image"], get_vectors()["meta"]), True,
    )
    text_vectors, text_meta = _cached_embeddings(
        args.cache_dir, args.model_family + "_text", signature,
        lambda: (get_vectors()["text"], get_vectors()["meta"]), True,
    )
    qids = queries.product_id.astype(str).to_numpy()
    gids = gallery.product_id.astype(str).to_numpy()
    qc = queries.category.astype(str).to_numpy()
    gc = gallery.category.astype(str).to_numpy()
    family = args.model_family
    rankings = {
        family + "_image_only": rank_queries_by_cosine(image_vectors[positions], image_vectors, qids, gids),
        family + "_text_only": rank_queries_by_cosine(text_vectors[positions], text_vectors, qids, gids),
        family + "_combined_raw": clip_query_gallery_score_fusion(
            image_vectors[positions], text_vectors[positions], image_vectors,
            text_vectors, qids, gids,
            image_weight=args.image_weight,
            text_image_weight=args.text_image_weight,
        ),
    }
    paired = {
        family + "_image_to_text": rank_queries_by_cosine(image_vectors[positions], text_vectors, qids, gids, False),
        family + "_text_to_image": rank_queries_by_cosine(text_vectors[positions], image_vectors, qids, gids, False),
    }
    ks = settings["ks"]
    metrics = {name: evaluate_query_gallery_category_rankings(ranks, qids, qc, gids, gc, ks)
               for name, ranks in rankings.items()}
    metrics.update({name: evaluate_query_gallery_identity_rankings(ranks, qids, qc, gids, ks)
                    for name, ranks in paired.items()})
    baseline = family + "_baseline"
    alignment = family + "_alignment"
    specs = {
        family + "_image_only": dict(comparison_group=baseline, role="frozen image encoder", deployed=False, input="image", formula=family + " S_II"),
        family + "_text_only": dict(comparison_group=baseline, role="frozen text encoder", deployed=False, input="text", formula=family + " S_TT"),
        family + "_combined_raw": dict(comparison_group=baseline, role="frozen multimodal fusion", deployed=False, input="image + text", formula="raw S_II + S_TI + S_TT fusion"),
        family + "_image_to_text": dict(comparison_group=alignment, role="frozen image to text", deployed=False, input="image", formula=family + " S_IT"),
        family + "_text_to_image": dict(comparison_group=alignment, role="frozen text to image", deployed=False, input="text", formula=family + " S_TI"),
    }
    rows, category_rows = [], []
    for group, task in ((baseline, "same-category product retrieval"),
                        (alignment, "exact paired-product retrieval")):
        selected = {name: spec for name, spec in specs.items() if spec["comparison_group"] == group}
        rows.extend(_annotated_rows(selected, metrics, ks, task))
        category_rows.extend(_annotated_category_rows(selected, metrics, ks, task))
    output = _prepare_output(args.output_dir)
    pd.DataFrame(rows).to_csv(output / "comparison.csv", index=False)
    pd.DataFrame(category_rows).to_csv(output / "comparison_by_category.csv", index=False)
    (output / "experiment.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    (output / "comparison.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "settings": settings, "models": {"image": image_meta, "text": text_meta},
        "metrics": metrics,
    }, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))
    print("Saved:", output)


if __name__ == "__main__":
    main()
