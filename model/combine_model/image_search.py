import sys
import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from datasets import load_dataset

from config import (
    RESNET_DIR,
    RESNET_MODEL_PATH,
    MAX_IMAGES,
    FILTER_BY_CATEGORY,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# Load ResNet config/model mà không sửa ResNet_Model
# =========================================================

def load_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(
        module_name,
        file_path
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def load_resnet_modules():
    config_path = RESNET_DIR / "config.py"
    model_path = RESNET_DIR / "model.py"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy ResNet config: {config_path}"
        )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy ResNet model: {model_path}"
        )

    resnet_config = load_module(
        "resnet_config",
        config_path
    )

    resnet_model = load_module(
        "resnet_model",
        model_path
    )

    return resnet_config, resnet_model


# =========================================================
# Transform
# =========================================================

def get_transform(resnet_config):
    return transforms.Compose([
        transforms.Resize(
            resnet_config.RESIZE_SIZE
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

    resnet_config, resnet_module = load_resnet_modules()

    model = resnet_module.FashionResNet(
        num_classes=resnet_config.NUM_CLASSES,
        pretrained=(model_type != "baseline")
    )

    if model_type == "trained":

        if not RESNET_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Không tìm thấy model đã train:\n"
                f"{RESNET_MODEL_PATH}"
            )

        checkpoint = torch.load(
            RESNET_MODEL_PATH,
            map_location=DEVICE
        )

        if isinstance(checkpoint, dict) and \
                "model_state_dict" in checkpoint:

            model.load_state_dict(
                checkpoint["model_state_dict"]
            )

        else:
            model.load_state_dict(checkpoint)

    model = model.to(DEVICE)
    model.eval()

    return model, resnet_config


# =========================================================
# Load dataset
# =========================================================

def load_fashion_dataset(max_images=MAX_IMAGES):

    resnet_config, _ = load_resnet_modules()

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
        return image_data.convert("RGB")

    return Image.open(image_data).convert("RGB")


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

    image = get_pil_image(image)

    tensor = transform(image)
    tensor = tensor.unsqueeze(0)
    tensor = tensor.to(DEVICE)

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
    x = F.normalize(
        x,
        p=2,
        dim=1
    )

    return x.squeeze(0).cpu().numpy()


# =========================================================
# Classify image
# =========================================================

@torch.no_grad()
def classify_image(
    model,
    image,
    transform,
    classes
):

    image = get_pil_image(image)

    tensor = transform(image)
    tensor = tensor.unsqueeze(0)
    tensor = tensor.to(DEVICE)

    output = model(tensor)

    probabilities = torch.softmax(
        output,
        dim=1
    )

    confidence, prediction = torch.max(
        probabilities,
        dim=1
    )

    category = classes[
        prediction.item()
    ]

    return (
        category,
        confidence.item()
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
    filter_category=True
):

    # -----------------------------------------------------
    # 1. Classify query
    # -----------------------------------------------------

    query_category, confidence = classify_image(
        model,
        query_image,
        transform,
        classes
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

    query_embedding = extract_embedding(
        model,
        query_image,
        transform
    )

    # -----------------------------------------------------
    # 3. Cosine similarity
    #
    # Vì embedding đã L2 normalize:
    # cosine = dot product
    # -----------------------------------------------------

    similarities = (
        image_embeddings @ query_embedding
    )

    # -----------------------------------------------------
    # 4. Filter category
    # -----------------------------------------------------

    candidate_indices = []

    for index, product in enumerate(products):

        category = product.get(
            "articleType",
            ""
        )

        if (
            not filter_category
            or category == query_category
        ):
            candidate_indices.append(index)

    # -----------------------------------------------------
    # 5. Sort similarity
    # -----------------------------------------------------

    candidate_indices.sort(
        key=lambda i: similarities[i],
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
            "image": product.get("image"),
            "index": index
        }

        results.append(result)

        if len(results) >= top_k:
            break

    return {
        "query_category": query_category,
        "confidence": confidence,
        "results": results
    }   