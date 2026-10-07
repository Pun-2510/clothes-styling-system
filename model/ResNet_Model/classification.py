"""Garment category classification for the trained ResNet18 model."""

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import torch
import numpy as np
from PIL import Image, ImageOps


@dataclass(frozen=True)
class ClassificationResult:
    category: str
    confidence: float
    probabilities: dict[str, float]

    # Thông tin bổ sung để debug / xử lý unknown
    second_category: str | None = None
    second_confidence: float = 0.0
    margin: float = 0.0
    is_unknown: bool = False
    normalized_entropy: float = 0.0


@dataclass(frozen=True)
class PatternResult:
    pattern: str
    complexity: float
    dominant_colors: tuple[str, ...]
    descriptor: tuple[float, ...]


def extract_color_descriptor(image, hue_bins=12, saturation_bins=4, value_bins=4):
    """Return a compact normalized HSV histogram for colour similarity."""
    rgb = _as_rgb(image).resize((96, 96), Image.Resampling.BILINEAR)
    hsv = np.asarray(rgb.convert("HSV"), dtype=np.float32) / 255.0
    parts = []
    for channel, bins in enumerate((hue_bins, saturation_bins, value_bins)):
        histogram, _ = np.histogram(hsv[:, :, channel], bins=bins, range=(0.0, 1.0))
        histogram = histogram.astype(np.float32)
        parts.append(histogram / max(float(histogram.sum()), 1.0))
    mean_rgb = np.asarray(rgb, dtype=np.float32).reshape(-1, 3).mean(axis=0) / 255.0
    mean_rgb /= max(float(np.linalg.norm(mean_rgb)), 1e-12)
    descriptor = np.concatenate([*parts, 3.0 * mean_rgb])
    descriptor /= max(float(np.linalg.norm(descriptor)), 1e-12)
    return descriptor


@torch.inference_mode()
def extract_resnet_embedding(model, image, transform, device):
    """Return one finite, L2-normalized 512D pre-FC ResNet embedding."""
    rgb_image = _as_rgb(image)
    tensor = transform(rgb_image)
    if tensor.ndim != 3:
        raise ValueError("transform must return [C, H, W]")
    tensor = tensor.unsqueeze(0).to(device)
    backbone = model.model
    features = backbone.conv1(tensor)
    features = backbone.bn1(features)
    features = backbone.relu(features)
    features = backbone.maxpool(features)
    features = backbone.layer1(features)
    features = backbone.layer2(features)
    features = backbone.layer3(features)
    features = backbone.layer4(features)
    features = backbone.avgpool(features).flatten(1)
    features = torch.nn.functional.normalize(features, p=2, dim=1)
    if not torch.isfinite(features).all():
        raise ValueError("ResNet produced a non-finite embedding")
    return features[0].cpu().numpy().astype(np.float32, copy=False)


def cosine_similarities(query_embedding, catalog_embeddings):
    """Safely compute cosine scores even when callers pass unnormalized data."""
    query = np.asarray(query_embedding, dtype=np.float32)
    catalog = np.asarray(catalog_embeddings, dtype=np.float32)
    if query.ndim != 1 or catalog.ndim != 2 or catalog.shape[1] != query.shape[0]:
        raise ValueError("Expected query [D] and catalog [N, D] with equal D")
    if not np.isfinite(query).all() or not np.isfinite(catalog).all():
        raise ValueError("Embeddings must contain only finite values")
    query = query / max(float(np.linalg.norm(query)), 1e-12)
    norms = np.linalg.norm(catalog, axis=1, keepdims=True)
    normalized_catalog = catalog / np.maximum(norms, 1e-12)
    return np.clip(normalized_catalog @ query, -1.0, 1.0)


def _as_rgb(image):
    """Convert a path, bytes, file object, or PIL image to safe RGB."""
    if isinstance(image, Image.Image):
        source = image.copy()
    elif isinstance(image, (bytes, bytearray, memoryview)):
        source = Image.open(BytesIO(bytes(image)))
    elif isinstance(image, (str, Path)) or hasattr(image, "read"):
        source = Image.open(image)
    else:
        raise TypeError(
            "image must be a PIL image, path, encoded bytes, or binary file object"
        )

    source = ImageOps.exif_transpose(source)
    # Background-removal output is often RGBA. Composite it on white so the
    # transparent region does not silently become black after RGB conversion.
    if source.mode in {"RGBA", "LA"} or "transparency" in source.info:
        rgba = source.convert("RGBA")
        canvas = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        source = Image.alpha_composite(canvas, rgba)
    return source.convert("RGB")


def analyze_pattern(image, histogram_bins=12):
    """Describe garment colour/texture for pattern-aware catalog matching.

    The normalized descriptor can be compared with cosine similarity. Brand
    is intentionally resolved from the matched catalog product instead of
    guessed directly from pixels.
    """
    rgb = _as_rgb(image).resize((160, 160), Image.Resampling.LANCZOS)
    array = np.asarray(rgb, dtype=np.float32) / 255.0
    gray = array.mean(axis=2)
    dx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    dy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    edge = np.sqrt(dx * dx + dy * dy)
    complexity = float(np.clip(edge.mean() * 8.0, 0.0, 1.0))

    # Directional energy separates common stripe/check patterns without
    # requiring another learned model.
    horizontal = float(dy.mean())
    vertical = float(dx.mean())
    balance = min(horizontal, vertical) / max(horizontal, vertical, 1e-8)
    if complexity < 0.12:
        pattern = "solid"
    elif balance < 0.55:
        pattern = "striped"
    elif complexity > 0.42 and balance > 0.75:
        pattern = "checkered_or_complex"
    else:
        pattern = "printed_or_textured"

    descriptors = []
    for channel in range(3):
        hist, _ = np.histogram(
            array[:, :, channel], bins=histogram_bins, range=(0.0, 1.0)
        )
        descriptors.extend(hist.astype(np.float32))
    edge_hist, _ = np.histogram(edge, bins=histogram_bins, range=(0.0, 1.0))
    descriptors.extend(edge_hist.astype(np.float32))
    vector = np.asarray(descriptors, dtype=np.float32)
    vector /= max(float(np.linalg.norm(vector)), 1e-8)

    palette = {
        "black": np.array([0.08, 0.08, 0.08]),
        "white": np.array([0.92, 0.92, 0.92]),
        "gray": np.array([0.50, 0.50, 0.50]),
        "red": np.array([0.80, 0.18, 0.16]),
        "green": np.array([0.18, 0.65, 0.25]),
        "blue": np.array([0.16, 0.32, 0.78]),
        "yellow": np.array([0.88, 0.78, 0.18]),
        "pink": np.array([0.90, 0.48, 0.62]),
        "brown": np.array([0.45, 0.28, 0.16]),
    }
    pixels = array.reshape(-1, 3)
    samples = pixels[::max(1, len(pixels) // 2000)]
    counts = {name: 0 for name in palette}
    for pixel in samples:
        nearest = min(palette, key=lambda name: np.linalg.norm(pixel - palette[name]))
        counts[nearest] += 1
    dominant = tuple(sorted(counts, key=counts.get, reverse=True)[:3])
    return PatternResult(pattern, complexity, dominant, tuple(map(float, vector)))


@torch.inference_mode()
def classify_garment(
    model,
    image,
    transform,
    classes,
    device,
    confidence_threshold: float = 0.60,
    margin_threshold: float = 0.15,
    entropy_threshold: float = 0.85,
    use_tta: bool = True,
    temperature: float = 1.0,
):
    """
    Classify one garment image.

    Always returns the nearest known garment category. ``is_unknown`` marks
    low-confidence/out-of-distribution inputs without discarding that nearest
    category. This lets garments such as ao dai or cheongsam map to Dresses
    and still participate in similarity search.

    Important:
    This only detects uncertain predictions. A closed-set ResNet18
    still cannot identify unseen categories such as "ao dai" or
    "qipao" by name unless those classes were included during training
    or a zero-shot model (CLIP/SigLIP) is added.
    """

    if not classes:
        raise ValueError(
            "classes must contain at least one garment category"
        )

    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError(
            "confidence_threshold must be between 0 and 1"
        )

    if not 0.0 <= margin_threshold <= 1.0:
        raise ValueError(
            "margin_threshold must be between 0 and 1"
        )

    if not 0.0 <= entropy_threshold <= 1.0:
        raise ValueError("entropy_threshold must be between 0 and 1")

    if temperature <= 0.0:
        raise ValueError("temperature must be greater than 0")

    # --------------------------------------------------
    # 1. Prepare image
    # --------------------------------------------------

    rgb_image = _as_rgb(image)

    views = [transform(rgb_image)]
    if use_tta:
        views.append(transform(ImageOps.mirror(rgb_image)))

    if any(view.ndim != 3 for view in views):
        raise ValueError("transform must return tensors with shape [C, H, W]")
    tensor = torch.stack(views).to(device)

    # --------------------------------------------------
    # 2. Inference
    # --------------------------------------------------

    model.eval()

    logits = model(tensor)

    # Some models/wrappers may return tuple/list
    if isinstance(logits, (tuple, list)):
        logits = logits[0]

    if logits.ndim != 2 or logits.shape[0] != len(views):
        raise ValueError(
            f"Expected classifier output shape [{len(views)}, num_classes], "
            f"got {tuple(logits.shape)}"
        )

    # Average logits across original/flipped views before softmax. This is
    # generally more stable than averaging already-normalized probabilities.
    logits = logits.mean(dim=0, keepdim=True) / temperature
    probabilities = torch.softmax(
        logits,
        dim=1,
    ).squeeze(0)

    # --------------------------------------------------
    # 3. Validate classifier output
    # --------------------------------------------------

    if probabilities.numel() != len(classes):
        raise ValueError(
            "The classifier output size does not match "
            f"the checkpoint classes: "
            f"model={probabilities.numel()}, "
            f"classes={len(classes)}"
        )

    # --------------------------------------------------
    # 4. Convert all scores
    # --------------------------------------------------

    probability_values = probabilities.detach().cpu().tolist()

    scores = {
        category: float(score)
        for category, score in zip(
            classes,
            probability_values,
        )
    }

    # --------------------------------------------------
    # 5. Top-K
    # --------------------------------------------------

    top_k = min(2, len(classes))

    values, indices = torch.topk(
        probabilities,
        k=top_k,
    )

    top1_index = int(indices[0].item())
    top1_confidence = float(values[0].item())

    top1_category = classes[top1_index]

    if top_k > 1:
        top2_index = int(indices[1].item())
        top2_confidence = float(values[1].item())
        top2_category = classes[top2_index]
    else:
        top2_confidence = 0.0
        top2_category = None

    # Difference between first and second prediction
    margin = top1_confidence - top2_confidence

    entropy = -torch.sum(
        probabilities * torch.log(probabilities.clamp_min(1e-12))
    )
    max_entropy = torch.log(torch.tensor(float(len(classes))))
    normalized_entropy = float((entropy / max_entropy).item()) if len(classes) > 1 else 0.0

    # --------------------------------------------------
    # 6. Unknown / OOD rejection
    # --------------------------------------------------

    low_confidence = (
        top1_confidence < confidence_threshold
    )

    ambiguous_prediction = (
        top_k > 1
        and margin < margin_threshold
    )

    high_entropy = normalized_entropy > entropy_threshold

    is_unknown = (
        low_confidence
        or ambiguous_prediction
        or high_entropy
    )

    # --------------------------------------------------
    # 7. Result
    # --------------------------------------------------

    return ClassificationResult(
        category=top1_category,
        confidence=top1_confidence,
        probabilities=scores,
        second_category=top2_category,
        second_confidence=top2_confidence,
        margin=margin,
        is_unknown=is_unknown,
        normalized_entropy=normalized_entropy,
    )
