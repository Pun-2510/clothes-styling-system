import time
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from config import MAX_IMAGES, TOP_K, RESULTS_DIR
from image_search import (
    load_resnet_model,
    load_fashion_dataset,
    get_transform,
    create_image_embeddings,
    extract_embedding,
    classify_image,
)

# ============================================================
# COMBINE TEST
# ResNet18 + SBERT
#
# Baseline   = SBERT + ResNet18 random weights
# Pretrained = SBERT + ResNet18 ImageNet weights
# Trained    = SBERT + trained ResNet18
#
# Multimodal ranking uses Reciprocal Rank Fusion (RRF)
# between Image Search and Text Search.
# ============================================================

MODEL_TYPES = ["baseline", "pretrained", "trained"]
RRF_K = 60
RESULTS_DIR = Path(RESULTS_DIR)


# ============================================================
# LOAD SBERT
# ============================================================

def load_sbert():
    import sys
    import importlib.util

    current_dir = Path(__file__).resolve().parent
    sbert_dir = current_dir.parent / "SBERT_Model"

    config_path = sbert_dir / "config.py"
    model_path = sbert_dir / "model.py"

    if not config_path.exists():
        raise FileNotFoundError(f"Không tìm thấy SBERT config: {config_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Không tìm thấy SBERT model: {model_path}")

    config_spec = importlib.util.spec_from_file_location(
        "combine_sbert_config", config_path
    )
    sbert_config = importlib.util.module_from_spec(config_spec)

    old_config = sys.modules.get("config")
    sys.modules["config"] = sbert_config

    try:
        config_spec.loader.exec_module(sbert_config)

        model_spec = importlib.util.spec_from_file_location(
            "combine_sbert_model", model_path
        )
        sbert_model = importlib.util.module_from_spec(model_spec)
        model_spec.loader.exec_module(sbert_model)
    finally:
        if old_config is not None:
            sys.modules["config"] = old_config
        else:
            sys.modules.pop("config", None)

    return sbert_model


def sbert_search(sbert, query, top_k):
    """Dùng đúng search_products() của SBERT_Model."""
    return sbert.search_products(query, top_k=top_k)


# ============================================================
# HELPERS
# ============================================================

def product_name(product):
    if not isinstance(product, dict):
        return ""

    return (
        product.get("productDisplayName")
        or product.get("product")
        or product.get("name")
        or ""
    ).strip()

def get_result_score(result):
    """
    Lấy điểm từ kết quả SBERT, ResNet hoặc RRF.
    """
    if not isinstance(result, dict):
        return 0.0

    value = result.get("score")
    if value is not None:
        return float(value)

    value = result.get("similarity")
    if value is not None:
        return float(value)

    value = result.get("confidence")
    if value is not None:
        return float(value)

    return 0.0

def result_name(result):
    """Hỗ trợ nhiều kiểu kết quả từ SBERT."""
    if isinstance(result, dict):
        for key in ["productDisplayName", "product", "name"]:
            if result.get(key):
                return str(result[key]).strip()
    return str(result).strip()


def true_match(query_product, result):
    return product_name(query_product) != "" and \
           product_name(query_product) == result_name(result)


def topk_hit(query_product, ranked_results, k):
    return any(
        true_match(query_product, r)
        for r in ranked_results[:k]
    )


def get_sbert_ranked(sbert, text, top_k):
    result = sbert_search(sbert, text, top_k)
    if result is None:
        return []
    return list(result)


def rrf_combine(image_ranked, text_ranked, top_k):
    """
    Combine hai ranking bằng Reciprocal Rank Fusion.
    Score = 1 / (RRF_K + rank)
    """
    scores = {}
    objects = {}

    for rank, item in enumerate(image_ranked, start=1):
        name = result_name(item)
        if not name:
            continue

        scores[name] = scores.get(name, 0.0) + 1.0 / (RRF_K + rank)
        objects[name] = item

    for rank, item in enumerate(text_ranked, start=1):
        name = result_name(item)
        if not name:
            continue

        scores[name] = scores.get(name, 0.0) + 1.0 / (RRF_K + rank)
        objects[name] = item

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        {
            "name": name,
            "score": score,
            "source": objects[name],
        }
        for name, score in ranked[:top_k]
    ]


# ============================================================
# RESNET IMAGE RANKING
# ============================================================

def image_search_ranked(
    model,
    products,
    embeddings,
    transform,
    classes,
    query_image,
    query_index,
):
    query_embedding = extract_embedding(
        model,
        query_image,
        transform
    )

    similarities = embeddings @ query_embedding

    ranked = []

    for i, score in enumerate(similarities):
        if i == query_index:
            continue

        product = products[i]

        ranked.append({
            "product": product_name(product),
            "category": product.get("articleType", ""),
            "brand": product.get(
                "brandName",
                product.get("brand", "Unknown")
            ),
            "similarity": float(score),
            "image": product.get("image"),
            "index": i,
        })

    ranked.sort(
        key=lambda x: x["similarity"],
        reverse=True
    )

    return ranked


# ============================================================
# EVALUATE ONE COMBINE VERSION
# ============================================================

def evaluate_model(
    model_type,
    products,
    resnet_config,
    sbert,
):
    print("\n")
    print("=" * 90)
    print(f"TESTING COMBINE MODEL: {model_type.upper()}")
    print("=" * 90)

    print("SBERT     : sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    print(f"ResNet18  : {model_type}")
    print("Fusion    : Reciprocal Rank Fusion (RRF)")

    model, _ = load_resnet_model(model_type=model_type)
    transform = get_transform(resnet_config)
    classes = resnet_config.SELECTED_CLASSES

    print("\nCreating ResNet image embeddings...")

    start = time.perf_counter()

    image_embeddings = create_image_embeddings(
        model,
        products,
        transform
    )

    embedding_time = time.perf_counter() - start

    print(f"Image embedding time: {embedding_time:.4f}s")

    total = 0
    classification_correct = 0
    confidences = []

    metrics = {
        "image": {"top1": 0, "top5": 0, "similarity": [], "time": []},
        "text": {"top1": 0, "top5": 0, "similarity": [], "time": []},
        "multimodal": {"top1": 0, "top5": 0, "score": [], "time": []},
    }

    category_stats = {}

    details = []

    print("\nTesting queries...")

    for query_index, query_product in enumerate(products):

        image = query_product.get("image")

        if image is None:
            continue

        if not isinstance(image, Image.Image):
            continue

        image = image.convert("RGB")

        true_name = product_name(query_product)
        true_category = query_product.get("articleType", "")
        text_query = true_name

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        predicted_category, confidence = classify_image(
            model,
            image,
            transform,
            classes
        )

        confidences.append(confidence)

        cls_correct = predicted_category == true_category

        if cls_correct:
            classification_correct += 1

        # ----------------------------------------------------
        # IMAGE SEARCH
        # ----------------------------------------------------

        start = time.perf_counter()

        image_ranked = image_search_ranked(
            model,
            products,
            image_embeddings,
            transform,
            classes,
            image,
            query_index
        )

        image_time = time.perf_counter() - start

        image_top1 = topk_hit(query_product, image_ranked, 1)
        image_top5 = topk_hit(query_product, image_ranked, 5)

        image_similarity = 0.0

        for item in image_ranked[:5]:
            if true_match(query_product, item):
                image_similarity = item["similarity"]
                break

        if image_top1:
            metrics["image"]["top1"] += 1
        if image_top5:
            metrics["image"]["top5"] += 1
            metrics["image"]["similarity"].append(image_similarity)

        metrics["image"]["time"].append(image_time)

        # ----------------------------------------------------
        # TEXT SEARCH - SBERT
        # ----------------------------------------------------

        start = time.perf_counter()

        text_ranked = get_sbert_ranked(
            sbert,
            text_query,
            TOP_K
        )

        text_time = time.perf_counter() - start

        text_top1 = topk_hit(query_product, text_ranked, 1)
        text_top5 = topk_hit(query_product, text_ranked, 5)

        text_similarity = 0.0

        for item in text_ranked[:5]:
            if true_match(query_product, item):
                if isinstance(item, dict):
                    text_similarity = float(
                        item.get(
                            "similarity",
                            item.get("score", 0.0)
                        ) or 0.0
                    )
                break

        if text_top1:
            metrics["text"]["top1"] += 1
        if text_top5:
            metrics["text"]["top5"] += 1
            metrics["text"]["similarity"].append(text_similarity)

        metrics["text"]["time"].append(text_time)

        # ----------------------------------------------------
        # MULTIMODAL COMBINE
        # ----------------------------------------------------

        start = time.perf_counter()

        combined = rrf_combine(
            image_ranked[:TOP_K],
            text_ranked[:TOP_K],
            TOP_K
        )

        multimodal_time = time.perf_counter() - start

        multimodal_names = [
            item["name"]
            for item in combined
        ]

        multimodal_top1 = true_name in multimodal_names[:1]
        multimodal_top5 = true_name in multimodal_names[:5]

        combined_score = 0.0

        for item in combined[:5]:
            if item["name"] == true_name:
                combined_score = item["score"]
                break

        if multimodal_top1:
            metrics["multimodal"]["top1"] += 1
        if multimodal_top5:
            metrics["multimodal"]["top5"] += 1
            metrics["multimodal"]["score"].append(combined_score)

        metrics["multimodal"]["time"].append(
            image_time + text_time + multimodal_time
        )

        # ----------------------------------------------------
        # CATEGORY
        # ----------------------------------------------------

        if true_category not in category_stats:
            category_stats[true_category] = {
                "total": 0,
                "classification": 0,
                "image_top1": 0,
                "image_top5": 0,
                "text_top1": 0,
                "text_top5": 0,
                "multimodal_top1": 0,
                "multimodal_top5": 0,
            }

        s = category_stats[true_category]
        s["total"] += 1
        s["classification"] += int(cls_correct)
        s["image_top1"] += int(image_top1)
        s["image_top5"] += int(image_top5)
        s["text_top1"] += int(text_top1)
        s["text_top5"] += int(text_top5)
        s["multimodal_top1"] += int(multimodal_top1)
        s["multimodal_top5"] += int(multimodal_top5)

        details.append({
            "model": model_type,
            "query_index": query_index,
            "true_product": true_name,
            "true_category": true_category,
            "predicted_category": predicted_category,
            "confidence": confidence,
            "image_top1": int(image_top1),
            "image_top5": int(image_top5),
            "image_similarity": image_similarity,
            "text_top1": int(text_top1),
            "text_top5": int(text_top5),
            "text_similarity": text_similarity,
            "multimodal_top1": int(multimodal_top1),
            "multimodal_top5": int(multimodal_top5),
            "multimodal_score": combined_score,
            "query_time": image_time + text_time + multimodal_time,
        })

        total += 1

        if total % 50 == 0:
            print(f"[{model_type}] Tested {total}/{len(products)}")

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    output = []

    for mode in ["image", "text", "multimodal"]:

        m = metrics[mode]

        # Image/Text dùng similarity.
        if mode in ["image", "text"]:
            mean_similarity = (
                float(np.mean(m["similarity"]))
                if m["similarity"]
                else 0.0
            )
        else:
            mean_similarity = 0.0

        # Multimodal dùng RRF score.
        if mode == "multimodal":
            mean_combine_score = (
                float(np.mean(m["score"]))
                if m["score"]
                else 0.0
            )
        else:
            mean_combine_score = 0.0

        output.append({
            "model": model_type,
            "mode": mode,
            "test_images": total,

            "classification_accuracy": (
                classification_correct / total * 100
                if total else 0.0
            ),

            "top1_accuracy": (
                m["top1"] / total * 100
                if total else 0.0
            ),

            "top5_accuracy": (
                m["top5"] / total * 100
                if total else 0.0
            ),

            "mean_similarity": mean_similarity,

            "mean_combine_score": mean_combine_score,

            "average_confidence": (
                float(np.mean(confidences))
                if confidences
                else 0.0
            ),

            "embedding_time": embedding_time,

            "average_query_time": (
                float(np.mean(m["time"]))
                if m["time"]
                else 0.0
            ),
        })

    # --------------------------------------------------------
    # CONSOLE RESULT
    # --------------------------------------------------------

    print("\n")
    print("-" * 90)
    print(f"RESULT: COMBINE {model_type.upper()}")
    print("-" * 90)
    print(f"Test images             : {total}")
    print(
        f"Classification Accuracy : "
        f"{classification_correct / total * 100:.2f}%"
        if total else "Classification Accuracy : 0.00%"
    )

    for r in output:
        print(f"\n{r['mode'].upper()} SEARCH")
        print(f"Top-1 Accuracy          : {r['top1_accuracy']:.2f}%")
        print(f"Top-5 Accuracy          : {r['top5_accuracy']:.2f}%")

        if r["mode"] == "multimodal":
            print(f"Mean Combine Score      : {r['mean_combine_score']:.6f}")
        else:
            print(f"Mean Similarity         : {r['mean_similarity']:.4f}")

        print(f"Average Query Time      : {r['average_query_time']:.4f}s")

    return output, category_stats, details


# ============================================================
# SAVE
# ============================================================

def save_csv(results, details):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = RESULTS_DIR / "combine_model_comparison.csv"

    headers = [
        "Model", "Mode", "Test Images",
        "Classification Accuracy (%)",
        "Top-1 Accuracy (%)",
        "Top-5 Accuracy (%)",
        "Mean Similarity",
        "Mean Combine Score",
        "Average Confidence",
        "Embedding Time (s)",
        "Average Query Time (s)"
    ]

    with open(model_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for r in results:
            writer.writerow([
                r["model"],
                r["mode"],
                r["test_images"],
                f"{r['classification_accuracy']:.2f}",
                f"{r['top1_accuracy']:.2f}",
                f"{r['top5_accuracy']:.2f}",
                f"{r['mean_similarity']:.4f}",
                f"{r['mean_combine_score']:.6f}",
                f"{r['average_confidence']:.4f}",
                f"{r['embedding_time']:.4f}",
                f"{r['average_query_time']:.4f}",
            ])

    detail_path = RESULTS_DIR / "combine_detailed_results.csv"

    if details:
        with open(detail_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=details[0].keys())
            writer.writeheader()
            writer.writerows(details)

    return model_path, detail_path


def save_category_csv(all_categories):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    path = RESULTS_DIR / "combine_category_comparison.csv"

    headers = [
        "Model", "Category", "Images",
        "Classification Accuracy (%)",
        "Image Top-1 (%)", "Image Top-5 (%)",
        "Text Top-1 (%)", "Text Top-5 (%)",
        "Multimodal Top-1 (%)", "Multimodal Top-5 (%)"
    ]

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for model, categories in all_categories.items():
            for category, s in sorted(categories.items()):
                n = s["total"]
                writer.writerow([
                    model,
                    category,
                    n,
                    f"{s['classification']/n*100:.2f}",
                    f"{s['image_top1']/n*100:.2f}",
                    f"{s['image_top5']/n*100:.2f}",
                    f"{s['text_top1']/n*100:.2f}",
                    f"{s['text_top5']/n*100:.2f}",
                    f"{s['multimodal_top1']/n*100:.2f}",
                    f"{s['multimodal_top5']/n*100:.2f}",
                ])

    return path


# ============================================================
# FINAL COMPARISON
# ============================================================

def print_final_comparison(results):

    print("\n\n")
    print("=" * 120)
    print("FINAL COMBINE MODEL COMPARISON")
    print("=" * 120)

    print(
        f"{'Model':<15}"
        f"{'Mode':<15}"
        f"{'Images':<10}"
        f"{'Top-1':<12}"
        f"{'Top-5':<12}"
        f"{'Similarity/Score':<20}"
        f"{'Time':<12}"
    )

    print("-" * 120)

    for r in results:

        score = (
            r["mean_combine_score"]
            if r["mode"] == "multimodal"
            else r["mean_similarity"]
        )

        print(
            f"{r['model']:<15}"
            f"{r['mode']:<15}"
            f"{r['test_images']:<10}"
            f"{r['top1_accuracy']:.2f}%"
            f"{'':<5}"
            f"{r['top5_accuracy']:.2f}%"
            f"{'':<5}"
            f"{score:<20.6f}"
            f"{r['average_query_time']:.4f}s"
        )

    print("=" * 120)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 90)
    print("AUTOMATIC COMBINE MODEL EVALUATION")
    print("=" * 90)

    print("\nComponents:")
    print("  ResNet18 -> Image Search")
    print("  SBERT    -> Text Search")
    print("  RRF      -> Multimodal Combine")

    print(f"\nMAX_IMAGES = {MAX_IMAGES}")
    print(f"TOP_K     = {TOP_K}")
    print(f"RRF_K     = {RRF_K}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # ResNet config + dataset
    # --------------------------------------------------------

    print("\nLoading ResNet configuration...")

    _, resnet_config = load_resnet_model(
        model_type="baseline"
    )

    if not hasattr(resnet_config, "SELECTED_CLASSES"):
        raise RuntimeError(
            "Không lấy được ResNet config. "
            "load_resnet_model() phải trả về (model, config)."
        )

    print("\nLoading fashion dataset...")

    products = load_fashion_dataset(
        max_images=MAX_IMAGES
    )

    print(f"Dataset loaded: {len(products)} products")

    # --------------------------------------------------------
    # SBERT
    # --------------------------------------------------------

    print("\nLoading SBERT...")

    sbert = load_sbert()

    print("SBERT loaded successfully.")

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    all_results = []
    all_categories = {}
    all_details = []

    for model_type in MODEL_TYPES:

        results, categories, details = evaluate_model(
            model_type,
            products,
            resnet_config,
            sbert
        )

        all_results.extend(results)
        all_categories[model_type] = categories
        all_details.extend(details)

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print_final_comparison(all_results)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    print("\n")
    print("=" * 90)
    print("SAVING RESULTS")
    print("=" * 90)

    model_csv, detail_csv = save_csv(
        all_results,
        all_details
    )

    category_csv = save_category_csv(
        all_categories
    )

    print(f"\nModel comparison : {model_csv}")
    print(f"Category results : {category_csv}")
    print(f"Detailed results : {detail_csv}")

    print("\n")
    print("=" * 90)
    print("COMBINE MODEL TEST COMPLETED")
    print("=" * 90)


if __name__ == "__main__":
    main()
