import sys
import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms
from datasets import load_dataset

from config import (
    RESNET_DIR,
    RESNET_MODEL_PATH,
    MAX_IMAGES,
    FILTER_BY_CATEGORY,
    ENABLE_PATTERN_SEARCH,
    GENDER_MODEL_PATH,
    COLOR_WEIGHT,
)
from background_removal import remove_image_background
from person_clothing import extract_clothing_region


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_PATTERN_EMBEDDING_CACHE = {}
_COLOR_EMBEDDING_CACHE = {}


# =========================================================
# Load ResNet config/model mà không sửa ResNet_Model
# =========================================================

def load_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(
        module_name,
        file_path
    )

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


def load_resnet_modules():
    config_path = RESNET_DIR / "config.py"
    model_path = RESNET_DIR / "model.py"
    classification_path = RESNET_DIR / "classification.py"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy ResNet config: {config_path}"
        )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy ResNet model: {model_path}"
        )

    if not classification_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy ResNet classification: {classification_path}"
        )

    resnet_config = load_module(
        "resnet_config",
        config_path
    )

    resnet_model = load_module(
        "resnet_model",
        model_path
    )

    resnet_classification = load_module(
        "resnet_classification",
        classification_path
    )

    return resnet_config, resnet_model, resnet_classification


# =========================================================
# Transform
# =========================================================

def get_transform(resnet_config):
    return transforms.Compose([
        transforms.Resize(
            (resnet_config.RESIZE_SIZE, resnet_config.RESIZE_SIZE)
        ),
        transforms.CenterCrop(
            resnet_config.IMAGE_SIZE
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=resnet_config.NORMALIZE_MEAN,
            std=resnet_config.NORMALIZE_STD
        )
    ])


# =========================================================
# Load model
# =========================================================

def load_resnet_model(model_type="trained"):
    """
    model_type:
        baseline  -> ResNet18 random weights
        pretrained -> ImageNet pretrained
        trained   -> model đã train fashion
    """

    resnet_config, resnet_module, _ = load_resnet_modules()

    checkpoint = None
    classes = list(resnet_config.SELECTED_CLASSES)

    if model_type == "trained":
        if not RESNET_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Không tìm thấy model đã train:\n{RESNET_MODEL_PATH}"
            )
        checkpoint = torch.load(
            RESNET_MODEL_PATH, map_location=DEVICE, weights_only=True
        )
        if isinstance(checkpoint, dict) and checkpoint.get("classes"):
            classes = list(checkpoint["classes"])

    resnet_config.SELECTED_CLASSES = classes
    resnet_config.NUM_CLASSES = len(classes)
    model = resnet_module.FashionResNet(
        num_classes=len(classes),
        pretrained=(model_type != "baseline")
    )

    if model_type == "trained":

        if isinstance(checkpoint, dict) and \
                "model_state_dict" in checkpoint:

            model.load_state_dict(
                checkpoint["model_state_dict"]
            )

        else:
            model.load_state_dict(checkpoint)

    model = model.to(DEVICE)
    model.eval()

    gender_path = RESNET_DIR / "gender.py"
    if GENDER_MODEL_PATH.is_file() and gender_path.is_file():
        gender_module = load_module("resnet_gender", gender_path)
        gender_checkpoint = torch.load(
            GENDER_MODEL_PATH, map_location=DEVICE, weights_only=True
        )
        model.gender_head = gender_module.GenderHead(
            num_classes=len(gender_checkpoint["classes"])
        ).to(DEVICE)
        model.gender_head.load_state_dict(gender_checkpoint["model_state_dict"])
        model.gender_head.eval()
        model.gender_classes = list(gender_checkpoint["classes"])
        model.gender_module = gender_module

    return model, resnet_config


# =========================================================
# Load dataset
# =========================================================

def load_fashion_dataset(max_images=MAX_IMAGES):

    resnet_config, _, _ = load_resnet_modules()

    print("Loading fashion dataset...")

    dataset = load_dataset(
        resnet_config.DATASET_NAME,
        split=resnet_config.DATASET_SPLIT
    )

    selected_classes = set(
        resnet_config.SELECTED_CLASSES
    )

    products = []

    for item in dataset:

        category = item.get("articleType")

        if category not in selected_classes:
            continue

        products.append(item)

        if len(products) >= max_images:
            break

    print(
        f"Loaded {len(products)} products"
    )

    return products


# =========================================================
# Convert image
# =========================================================

def get_pil_image(image_data):

    if isinstance(image_data, Image.Image):
        source = image_data.copy()
    else:
        source = Image.open(image_data)

    source = ImageOps.exif_transpose(source)
    if source.width < 32 or source.height < 32:
        raise ValueError("Image dimensions must be at least 32x32 pixels")
    if source.mode in {"RGBA", "LA"} or "transparency" in source.info:
        rgba = source.convert("RGBA")
        canvas = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        source = Image.alpha_composite(canvas, rgba)
    return source.convert("RGB")


# =========================================================
# Extract ResNet image embedding
# =========================================================

@torch.no_grad()
def extract_embedding(
    model,
    image,
    transform
):
    """
    Lấy feature trước FC của ResNet18.
    ResNet18 -> 512 dimensions.
    """

    if not ENABLE_PATTERN_SEARCH:
        _, _, classification = load_resnet_modules()
        return classification.extract_resnet_embedding(
            model, image, transform, DEVICE
        )

    image = get_pil_image(image)

    width, height = image.size
    # Global shape plus overlapping local garment regions. Local views retain
    # logos, prints and fabric texture that global average pooling can dilute.
    views = [image]
    if ENABLE_PATTERN_SEARCH and width >= 32 and height >= 32:
        views.extend([
            image.crop((0, 0, width, int(height * 0.62))),
            image.crop((0, int(height * 0.38), width, height)),
            image.crop((int(width * 0.15), int(height * 0.15),
                        int(width * 0.85), int(height * 0.85))),
        ])
    tensor = torch.stack([transform(view) for view in views]).to(DEVICE)

    # ResNet backbone
    x = model.model.conv1(tensor)
    x = model.model.bn1(x)
    x = model.model.relu(x)
    x = model.model.maxpool(x)

    x = model.model.layer1(x)
    x = model.model.layer2(x)
    x = model.model.layer3(x)
    x = model.model.layer4(x)

    x = model.model.avgpool(x)

    x = torch.flatten(
        x,
        1
    )

    # Normalize
    x = F.normalize(x, p=2, dim=1)
    if len(views) > 1:
        # Shape remains dominant; local regions provide pattern/logo detail.
        x = 0.55 * x[:1] + 0.45 * x[1:].mean(dim=0, keepdim=True)
    x = F.normalize(x, p=2, dim=1)
    return x.squeeze(0).cpu().numpy()


# =========================================================
# Classify image
# =========================================================

def classify_image(
    model,
    image,
    transform,
    classes
):
    result = classify_image_details(model, image, transform, classes)
    return result.category, result.confidence


def classify_image_details(
    model,
    image,
    transform,
    classes
):
    _, _, classification = load_resnet_modules()
    return classification.classify_garment(
        model, image, transform, classes, DEVICE
    )


# =========================================================
# Create catalog embeddings
# =========================================================

def create_image_embeddings(
    model,
    products,
    transform
):

    embeddings = []

    print("Creating image embeddings...")

    for index, product in enumerate(products):

        try:

            image = product["image"]

            embedding = extract_embedding(
                model,
                image,
                transform
            )

            embeddings.append(
                embedding
            )

        except Exception as e:

            print(
                f"Skip product {index}: {e}"
            )

            embeddings.append(
                np.zeros(512, dtype=np.float32)
            )

    embeddings = np.array(
        embeddings,
        dtype=np.float32
    )

    print(
        f"Embedding shape: {embeddings.shape}"
    )

    return embeddings


def create_pattern_embeddings(products, classification_module):
    """Build lightweight colour/texture descriptors once per product list."""
    cache_key = (id(products), len(products))
    cached = _PATTERN_EMBEDDING_CACHE.get(cache_key)
    if cached is not None:
        return cached
    rows = []
    for product in products:
        try:
            descriptor = classification_module.analyze_pattern(
                product["image"]
            ).descriptor
            rows.append(np.asarray(descriptor, dtype=np.float32))
        except Exception:
            rows.append(np.zeros(48, dtype=np.float32))
    embeddings = np.asarray(rows, dtype=np.float32)
    _PATTERN_EMBEDDING_CACHE.clear()
    _PATTERN_EMBEDDING_CACHE[cache_key] = embeddings
    return embeddings


def create_color_embeddings(products, classification_module):
    cache_key = (id(products), len(products))
    cached = _COLOR_EMBEDDING_CACHE.get(cache_key)
    if cached is not None:
        return cached
    rows = []
    for product in products:
        try:
            rows.append(classification_module.extract_color_descriptor(product["image"]))
        except Exception:
            rows.append(np.zeros(23, dtype=np.float32))
    embeddings = np.asarray(rows, dtype=np.float32)
    _COLOR_EMBEDDING_CACHE.clear()
    _COLOR_EMBEDDING_CACHE[cache_key] = embeddings
    return embeddings


# =========================================================
# Search
# =========================================================

def search_by_image(
    query_image,
    model,
    products,
    image_embeddings,
    transform,
    classes,
    top_k=10,
    filter_category=True,
    remove_background=False,
    category_weight=0.15,
    pattern_weight=0.20,
    detect_person=False,
    gender_weight=0.05,
):

    # -----------------------------------------------------
    # 1. Classify query
    # -----------------------------------------------------

    person_image, person_detection = (
        extract_clothing_region(query_image)
        if detect_person
        else (get_pil_image(query_image), {
            "person_detected": False,
            "person_confidence": 0.0,
            "crop_box": None,
        })
    )
    processed_image = (
        remove_image_background(person_image)
        if remove_background
        else person_image
    )

    original_classification = classify_image_details(
        model, person_image, transform, classes
    )
    classification = original_classification
    if remove_background:
        foreground_classification = classify_image_details(
            model, processed_image, transform, classes
        )
        # Prefer the original unless foreground isolation gives a materially
        # stronger prediction; this limits damage when rembg removes garment edges.
        if foreground_classification.confidence > original_classification.confidence + 0.10:
            classification = foreground_classification
    query_category = classification.category
    confidence = classification.confidence
    _, _, classification_module = load_resnet_modules()
    pattern = (
        classification_module.analyze_pattern(processed_image)
        if ENABLE_PATTERN_SEARCH else None
    )

    print(
        f"\nPredicted category: {query_category}"
    )

    print(
        f"Confidence: {confidence:.4f}"
    )

    # -----------------------------------------------------
    # 2. Query embedding
    # -----------------------------------------------------

    original_embedding = extract_embedding(model, person_image, transform)
    if remove_background:
        foreground_embedding = extract_embedding(model, processed_image, transform)
        query_embedding = 0.75 * original_embedding + 0.25 * foreground_embedding
        query_embedding /= max(float(np.linalg.norm(query_embedding)), 1e-12)
    else:
        query_embedding = original_embedding

    # -----------------------------------------------------
    # 3. Cosine similarity
    #
    # Vì embedding đã L2 normalize:
    # cosine = dot product
    # -----------------------------------------------------

    similarities = classification_module.cosine_similarities(
        query_embedding, image_embeddings
    )
    color_query = classification_module.extract_color_descriptor(processed_image)
    color_embeddings = create_color_embeddings(products, classification_module)
    color_similarities = classification_module.cosine_similarities(
        color_query, color_embeddings
    )
    gender_result = None
    if hasattr(model, "gender_head"):
        gender_result = model.gender_module.classify_gender(
            model.gender_head, query_embedding, model.gender_classes
        )
    if ENABLE_PATTERN_SEARCH:
        pattern_embeddings = create_pattern_embeddings(
            products, classification_module
        )
        pattern_query = np.asarray(pattern.descriptor, dtype=np.float32)
        pattern_similarities = pattern_embeddings @ pattern_query
    else:
        pattern_similarities = np.zeros(len(products), dtype=np.float32)

    # Use classification as a soft prior instead of a hard category gate.
    # Novel garments can therefore map to the nearest known class (for
    # example ao dai -> Dresses) while visual cosine similarity remains the
    # main ranking signal.
    category_scores = np.array([
        classification.probabilities.get(
            product.get("articleType", ""), 0.0
        )
        for product in products
    ], dtype=np.float32)
    ranking_scores = (
        (1.0 - pattern_weight) * similarities
        + pattern_weight * pattern_similarities
        if ENABLE_PATTERN_SEARCH else (
            (1.0 - COLOR_WEIGHT) * similarities
            + COLOR_WEIGHT * color_similarities
        )
    )
    if filter_category:
        ranking_scores = ranking_scores + category_weight * category_scores
    gender_scores = np.zeros(len(products), dtype=np.float32)
    if gender_result is not None and not gender_result.is_uncertain:
        for index, product in enumerate(products):
            value = product.get("gender", "")
            normalized = (
                "Men" if value in {"Men", "Boys"}
                else "Women" if value in {"Women", "Girls"}
                else "Unisex"
            )
            if normalized in {gender_result.gender, "Unisex"}:
                gender_scores[index] = gender_result.confidence
        ranking_scores = ranking_scores + gender_weight * gender_scores

    # -----------------------------------------------------
    # 4. Filter category
    # -----------------------------------------------------

    candidate_indices = []

    for index, product in enumerate(products):

        category = product.get(
            "articleType",
            ""
        )

        candidate_indices.append(index)

    # -----------------------------------------------------
    # 5. Sort similarity
    # -----------------------------------------------------

    candidate_indices.sort(
        key=lambda i: ranking_scores[i],
        reverse=True
    )

    # -----------------------------------------------------
    # 6. Top K
    # -----------------------------------------------------

    results = []

    used_products = set()

    for index in candidate_indices:

        product = products[index]

        product_name = product.get(
            "productDisplayName",
            "Unknown"
        )

        # tránh duplicate product
        if product_name in used_products:
            continue

        used_products.add(product_name)

        result = {
            "rank": len(results) + 1,
            "product": product_name,
            "category": product.get(
                "articleType",
                ""
            ),
            "brand": product.get(
                "brandName",
                product.get("brand", "Unknown")
            ),
            "similarity": float(
                similarities[index]
            ),
            "category_score": float(category_scores[index]),
            "ranking_score": float(ranking_scores[index]),
            "pattern_similarity": float(pattern_similarities[index]),
            "color_similarity": float(color_similarities[index]),
            "gender_score": float(gender_scores[index]),
            "image": product.get("image"),
            "index": index
        }

        results.append(result)

        if len(results) >= top_k:
            break

    return {
        "query_category": query_category,
        "confidence": confidence,
        "is_uncertain": classification.is_unknown,
        "normalized_entropy": classification.normalized_entropy,
        "gender": ({
            "label": gender_result.gender,
            "confidence": gender_result.confidence,
            "probabilities": gender_result.probabilities,
            "is_uncertain": gender_result.is_uncertain,
        } if gender_result is not None else None),
        "pattern": ({
            "enabled": True,
            "type": pattern.pattern,
            "complexity": pattern.complexity,
            "dominant_colors": list(pattern.dominant_colors),
        } if pattern is not None else {"enabled": False}),
        "color_weight": COLOR_WEIGHT,
        "class_probabilities": classification.probabilities,
        "background_removed": remove_background,
        "person_detection": person_detection,
        "results": results
    }
