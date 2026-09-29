"""Benchmark ResNet18, SBERT and CLIP on one fashion catalog.

Run from any directory (Python environment needs torch, torchvision, transformers,
sentence-transformers, numpy, pandas, Pillow):
  python model/compare_models.py --help
  python model/compare_models.py --image-dir project/data/data
  python model/compare_models.py --models resnet18 sbert clip --device cpu
  python model/compare_models.py --models clip_finetuned --image-dir project/data/data

Default input: the saved CLIP experiment dataset, restricted to its test split.
Use --csv for another catalog with product_id, category, image_path,
product_name and optional description. No model training is performed.
Category relevance measures similar-category retrieval, NOT outfit compatibility.
No automatic overall winner: compare identical tasks and consider required inputs.
"""

import argparse
import csv
import gc
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "project"
MODEL_NAMES = ("resnet18", "resnet18_trained", "sbert", "clip", "clip_finetuned")


def arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=PROJECT / "runs/clip_finetune_3epochs_local/dataset.csv")
    parser.add_argument("--split", default="test", help="Filter split column if present; use 'all' for all rows")
    parser.add_argument("--image-dir", type=Path, help="Relocate images using the basename in image_path")
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=["resnet18", "sbert", "clip"])
    parser.add_argument("--resnet-checkpoint", type=Path, default=ROOT / "model/Web_Test/models/resnet_outfit.pth")
    parser.add_argument("--clip-checkpoint", type=Path, default=PROJECT / "runs/clip_finetune_3epochs_local/best")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument("--latency-queries", type=int, default=30)
    parser.add_argument("--output", type=Path, help="New output directory; existing reports are never overwritten")
    parser.add_argument("--self-test", action="store_true", help="Test ranking metrics without loading models/data")
    return parser.parse_args()


def retrieval(queries, gallery, ids, categories, ks, relevance):
    """Macro metrics; exclude every image/text of the query product for category tasks."""
    import numpy as np
    totals = {f"{metric}@{k}": 0.0 for k in ks for metric in ("precision", "recall", "hit_rate", "mrr")}
    valid = 0
    for index, query in enumerate(queries):
        eligible = ids != ids[index] if relevance == "category" else np.ones(len(ids), dtype=bool)
        positive = (categories == categories[index]) if relevance == "category" else (ids == ids[index])
        positive = positive & eligible
        if not positive.any():
            continue
        candidates = np.flatnonzero(eligible)
        scores = gallery[candidates] @ query
        ranked = candidates[np.argsort(-scores, kind="stable")[:max(ks)]]
        valid += 1
        for k in ks:
            hits = positive[ranked[:k]]
            positions = np.flatnonzero(hits)
            totals[f"precision@{k}"] += float(hits.sum()) / len(hits)
            totals[f"recall@{k}"] += float(hits.sum()) / int(positive.sum())
            totals[f"hit_rate@{k}"] += float(hits.any())
            totals[f"mrr@{k}"] += 1 / (int(positions[0]) + 1) if len(positions) else 0
    return {**{key: value / valid if valid else None for key, value in totals.items()},
            "evaluated_queries": valid, "skipped_queries": len(ids) - valid, "gallery_size": len(ids)}


def self_test():
    import numpy as np
    ids = np.array(["a", "a", "b", "c"])
    categories = np.array(["shoe", "shoe", "shoe", "hat"])
    features = np.eye(4, dtype=np.float32)
    result = retrieval(features, features, ids, categories, [1, 10], "category")
    assert result["evaluated_queries"] == 3 and result["skipped_queries"] == 1
    assert abs(result["recall@1"] - 5 / 6) < 1e-9
    assert result["recall@10"] == 1
    pairs = retrieval(features, features, ids, categories, [1, 10], "product")
    assert pairs["hit_rate@1"] == 1 and pairs["recall@1"] == 0.75
    assert pairs["recall@10"] == 1
    print("PASS: macro recall, multiple positives, self-product exclusion, singleton category, K > gallery")


def sync(device):
    import torch
    if device == "cuda":
        torch.cuda.synchronize()


def load_encoder(name, args):
    import numpy as np
    import torch
    from PIL import Image

    def images(rows):
        result = []
        for path in rows.image_path:
            with Image.open(path) as image:
                result.append(image.convert("RGB"))
        return result

    if name.startswith("resnet18"):
        from torchvision import models, transforms
        if name == "resnet18_trained":
            checkpoint = torch.load(args.resnet_checkpoint, map_location="cpu", weights_only=True)
            state = checkpoint.get("model_state_dict", checkpoint)
            state = {key.removeprefix("model."): value for key, value in state.items()}
            model = models.resnet18(weights=None)
            model.fc = torch.nn.Linear(model.fc.in_features, state["fc.weight"].shape[0])
            model.load_state_dict(state)
            preprocess = transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop(224),
                transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
            source = str(args.resnet_checkpoint.resolve())
        else:
            weights = models.ResNet18_Weights.DEFAULT
            model = models.resnet18(weights=weights)
            preprocess = weights.transforms()
            source = str(weights)
        model = torch.nn.Sequential(*list(model.children())[:-1]).to(args.device).eval()

        def encode(rows, modality):
            tensor = torch.stack([preprocess(image) for image in images(rows)]).to(args.device)
            with torch.inference_mode():
                return model(tensor).flatten(1).cpu().numpy()
        modalities = ["image"]
    elif name == "sbert":
        from sentence_transformers import SentenceTransformer
        source = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        model = SentenceTransformer(source, device=args.device)

        def encode(rows, modality):
            return model.encode(rows.product_text.tolist(), batch_size=args.batch_size,
                                convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        modalities = ["text"]
    else:
        from transformers import AutoProcessor, CLIPModel
        source = str(args.clip_checkpoint.resolve()) if name == "clip_finetuned" else "openai/clip-vit-base-patch32"
        processor = AutoProcessor.from_pretrained(source)
        model = CLIPModel.from_pretrained(source).to(args.device).eval()

        def encode(rows, modality):
            with torch.inference_mode():
                if modality == "image":
                    inputs = processor(images=images(rows), return_tensors="pt").to(args.device)
                    output = model.vision_model(pixel_values=inputs["pixel_values"], return_dict=True)
                    features = model.visual_projection(output.pooler_output)
                else:
                    inputs = processor(text=rows.product_text.tolist(), padding=True, truncation=True,
                        max_length=model.config.text_config.max_position_embeddings, return_tensors="pt").to(args.device)
                    output = model.text_model(**inputs, return_dict=True)
                    features = model.text_projection(output.pooler_output)
                return features.cpu().numpy()
        modalities = ["image", "text"]

    def normalized(rows, modality):
        matrix = np.asarray(encode(rows, modality), dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if not np.isfinite(matrix).all() or (norms == 0).any():
            raise ValueError("Nonfinite or zero embeddings; benchmark stopped")
        return matrix / norms

    return model, normalized, modalities, source


def main():
    args = arguments()
    if args.self_test:
        self_test()
        return
    if min(args.batch_size, args.latency_queries, *args.ks) < 1:
        raise ValueError("batch-size, latency-queries and ks must be positive")
    import importlib.metadata
    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image
    sys.path.insert(0, str(PROJECT))
    from src.clip_data import build_product_text, file_sha256

    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    torch.manual_seed(42)
    products = pd.read_csv(args.csv, dtype={"product_id": str})
    required = ["product_id", "category", "product_name", "image_path"]
    if not set(required).issubset(products.columns):
        raise ValueError(f"CSV needs columns: {required}")
    if args.split != "all" and "split" in products:
        products = products.loc[products.split == args.split].copy()
    if len(products) < 2 or products[required].isna().any().any():
        raise ValueError("Need at least two rows with complete required fields")
    if any(products[column].astype(str).str.strip().eq("").any() for column in required):
        raise ValueError("Required fields cannot be blank")
    def image_path(value):
        if args.image_dir:
            return args.image_dir.resolve() / str(value).replace("\\", "/").split("/")[-1]
        path = Path(value)
        return path if path.is_absolute() else PROJECT / path
    products["image_path"] = products.image_path.map(image_path)
    # Validate before loading any model, and never silently drop different rows per model.
    hashes = []
    for path in products.image_path:
        with Image.open(path) as image:
            image.verify()
        hashes.append(file_sha256(path))
    if len(set(hashes)) != len(hashes):
        raise ValueError("Exact duplicate images found; deduplicate the evaluation CSV first")
    products["product_text"] = products.apply(build_product_text, axis=1)
    products = products.reset_index(drop=True)
    output = args.output or ROOT / "model/comparison_results" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    products.assign(image_sha256=hashes).to_csv(output / "evaluation_catalog.csv", index=False)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
        "settings": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "source_csv_sha256": file_sha256(args.csv), "rows": len(products),
        "environment": {"python": sys.version, "platform": platform.platform(),
            "device": torch.cuda.get_device_name() if args.device == "cuda" else platform.processor(),
            "torch_threads": torch.get_num_threads(),
            "packages": {package: importlib.metadata.version(package) for package in ("torch", "numpy", "pandas")}},
        "limitations": [
            "Category appears in catalog text. Category retrieval is a metadata-assisted proxy, not free-form query quality.",
            "Compare image/category tasks across ResNet and CLIP; text/category across SBERT and CLIP. Different tasks are not a common leaderboard.",
            "Cross-modal product retrieval is supported only by CLIP in this benchmark.",
            "No Vietnamese translation is measured. No user relevance or outfit compatibility labels are available.",
            "Training overlap is NOT verified for any checkpoint. A saved test split alone does not prove independence from ResNet training.",
            "Single-query latency includes preprocessing and image file IO, excludes model loading, translation, HTTP and UI.",
            "Model load time can include network downloads; CUDA peak is allocated tensor memory, not total GPU or CPU RAM.",
            "Precision denominator is min(K, eligible gallery size); queries without positives are skipped and counted."],
        "models": {}, "errors": {}}
    rows = []

    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        if rows:
            with (output / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        summary = ["# So sánh mô hình tìm kiếm thời trang", "",
            f"Trạng thái: `{report['status']}`. Catalog: {len(products)} sản phẩm.", "",
            "Các điểm chất lượng nằm trong [0, 1]; nhân 100 để biểu diễn phần trăm.", "",
            "| Model | Tác vụ | K | Precision | Recall | Hit rate | MRR |",
            "|---|---|---:|---:|---:|---:|---:|"]
        for row in rows:
            values = ["N/A" if row[key] is None else f"{row[key]:.4f}"
                      for key in ("precision", "recall", "hit_rate", "mrr")]
            summary.append(f"| {row['model']} | {row['task']} | {row['k']} | " + " | ".join(values) + " |")
        summary += ["", "## Hiệu năng", "",
            "P50/P95 đo truy vấn từng sản phẩm sau warm-up, gồm mã hóa và xếp hạng cùng phương thức.", "",
            "| Model | Đầu vào | Tham số (triệu) | Embedding | Tạo catalog (s) | P50 (ms) | P95 (ms) |",
            "|---|---|---:|---:|---:|---:|---:|"]
        for name, metadata in report["models"].items():
            for modality, perf in metadata["modalities"].items():
                summary.append(f"| {name} | {modality} | {metadata['parameters']/1e6:.2f} | "
                    f"{perf['embedding_dim']} | {perf['catalog_encode_seconds']:.3f} | "
                    f"{perf['query_p50_ms']:.3f} | {perf['query_p95_ms']:.3f} |")
        summary += ["", "## Cách dùng kết quả để chọn mô hình", "",
            "- Tìm bằng ảnh: đối chiếu ResNet và CLIP ở image_to_image_category.",
            "- Tìm bằng mô tả: đối chiếu SBERT và CLIP ở text_to_text_category.",
            "- Tìm mô tả sang ảnh: xem text_to_image_product của CLIP. ResNet/SBERT riêng lẻ không hỗ trợ tác vụ này.",
            "- Cân nhắc đồng thời chất lượng, P95, dung lượng và đầu vào cần hỗ trợ; không cộng điểm của các tác vụ khác nhau.",
            "- Chưa benchmark hệ kết hợp SBERT + ResNet + RRF hoặc toàn bộ API của project.",
            "", "## Giới hạn phép đo", ""]
        summary.extend(f"- {item}" for item in report["limitations"])
        if report["errors"]:
            summary += ["", "## Các mô hình chưa chạy thành công", ""]
            summary.extend(f"- {name}: {error}" for name, error in report["errors"].items())
        (output / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")

    save()
    for name in dict.fromkeys(args.models):
        model = encoder = None
        try:
            print(f"Benchmarking {name} on {len(products)} rows...", flush=True)
            if args.device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            model, encoder, modalities, source = load_encoder(name, args)
            sync(args.device)
            metadata = {"source": source, "load_seconds": time.perf_counter() - start,
                "parameters": sum(p.numel() for p in model.parameters()),
                "parameter_mib": sum(p.numel() * p.element_size() for p in model.parameters()) / 2**20,
                "modalities": {}}
            features = {}
            for modality in modalities:
                encoder(products.iloc[:1], modality)  # Warm up before timed measurements.
                sync(args.device)
                start = time.perf_counter()
                features[modality] = np.concatenate([encoder(products.iloc[i:i + args.batch_size], modality)
                    for i in range(0, len(products), args.batch_size)])
                sync(args.device)
                elapsed = time.perf_counter() - start
                timings = []
                indices = np.random.default_rng(42).permutation(len(products))[:args.latency_queries]
                for index in indices:
                    sync(args.device)
                    start = time.perf_counter()
                    query = encoder(products.iloc[index:index + 1], modality)[0]
                    scores = features[modality] @ query
                    scores[products.product_id.to_numpy() == products.product_id.iloc[index]] = -np.inf
                    np.argsort(-scores, kind="stable")[:max(args.ks)]
                    sync(args.device)
                    timings.append((time.perf_counter() - start) * 1000)
                metadata["modalities"][modality] = {"embedding_dim": features[modality].shape[1],
                    "embedding_mib": features[modality].nbytes / 2**20, "catalog_encode_seconds": elapsed,
                    "catalog_items_per_second": len(products) / elapsed, "latency_samples": len(timings),
                    "query_p50_ms": float(np.percentile(timings, 50)), "query_p95_ms": float(np.percentile(timings, 95))}
            tasks = [(m, m, "category") for m in modalities]
            if len(modalities) == 2:
                tasks += [("text", "image", "product"), ("image", "text", "product")]
            metadata["metrics"] = {}
            for query_modality, gallery_modality, relevance in tasks:
                task = f"{query_modality}_to_{gallery_modality}_{relevance}"
                metrics = retrieval(features[query_modality], features[gallery_modality],
                    products.product_id.to_numpy(), products.category.to_numpy(), args.ks, relevance)
                metadata["metrics"][task] = metrics
                # Latency applies to same-modality search only, never label it cross-modal latency.
                performance = metadata["modalities"][query_modality]
                for k in sorted(set(args.ks)):
                    rows.append({"model": name, "task": task, "k": k,
                        **{metric: metrics[f"{metric}@{k}"] for metric in ("precision", "recall", "hit_rate", "mrr")},
                        "evaluated_queries": metrics["evaluated_queries"], "skipped_queries": metrics["skipped_queries"],
                        "parameters": metadata["parameters"], "parameter_mib": metadata["parameter_mib"],
                        "embedding_dim": performance["embedding_dim"],
                        "query_p50_ms": performance["query_p50_ms"] if relevance == "category" else None,
                        "query_p95_ms": performance["query_p95_ms"] if relevance == "category" else None})
            metadata["cuda_peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20 if args.device == "cuda" else None
            report["models"][name] = metadata
        except Exception as error:
            report["errors"][name] = f"{type(error).__name__}: {error}"
            print(f"FAILED {name}: {error}", file=sys.stderr)
        finally:
            del model, encoder
            gc.collect()
            if args.device == "cuda":
                torch.cuda.empty_cache()
            save()
    report["status"] = "partial_failure" if report["errors"] else "complete"
    save()
    print(f"Reports: {output.resolve()}")
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
