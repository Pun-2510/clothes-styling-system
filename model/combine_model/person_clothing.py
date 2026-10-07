"""Pose-aware person detection and clothing-region extraction."""

from functools import lru_cache
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torchvision.models.detection import (
    KeypointRCNN_ResNet50_FPN_Weights,
    keypointrcnn_resnet50_fpn,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _rgb(image):
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
    if source.mode in {"RGBA", "LA"} or "transparency" in source.info:
        rgba = source.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        source = Image.alpha_composite(background, rgba)
    return source.convert("RGB")


@lru_cache(maxsize=1)
def load_person_detector():
    """Load the pretrained COCO person/keypoint detector once per process."""
    weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
    model = keypointrcnn_resnet50_fpn(weights=weights).to(DEVICE).eval()
    return model, weights.transforms()


@torch.inference_mode()
def extract_clothing_region(image, confidence_threshold=0.75):
    """Return a pose-aware body crop while excluding the head and feet.

    Keypoints make the crop useful for standing and sitting people. If no
    reliable person is detected, the original image is returned unchanged.
    """
    rgb = _rgb(image)
    model, preprocess = load_person_detector()
    output = model([preprocess(rgb).to(DEVICE)])[0]
    scores = output["scores"].detach().cpu()
    valid = torch.where(scores >= confidence_threshold)[0]
    if len(valid) == 0:
        return rgb, {
            "person_detected": False,
            "person_confidence": 0.0,
            "person_box": None,
            "person_area_ratio": 0.0,
            "clothing_area_ratio": 1.0,
            "crop_box": (0, 0, *rgb.size),
        }

    # Prefer the largest confident person, useful when a catalog image has
    # small people in its background.
    boxes = output["boxes"].detach().cpu()
    areas = (boxes[valid, 2] - boxes[valid, 0]) * (
        boxes[valid, 3] - boxes[valid, 1]
    )
    index = int(valid[int(torch.argmax(areas))])
    x1, y1, x2, y2 = boxes[index].tolist()
    width, height = x2 - x1, y2 - y1
    image_area = float(rgb.width * rgb.height)
    person_area_ratio = min(1.0, max(0.0, width * height / image_area))

    keypoints = output["keypoints"][index].detach().cpu()
    keypoint_scores = output.get("keypoints_scores")
    if keypoint_scores is not None:
        keypoint_scores = keypoint_scores[index].detach().cpu()

    def reliable(indices):
        visibility = keypoints[indices, 2] > 0
        if keypoint_scores is not None:
            visibility &= keypoint_scores[indices] > 2.0
        return bool(torch.all(visibility))

    # COCO keypoints: shoulders 5/6, knees 13/14, ankles 15/16.
    if reliable([5, 6]):
        top = float(keypoints[[5, 6], 1].min()) - 0.08 * height
    else:
        top = y1 + 0.12 * height

    if reliable([13, 14, 15, 16]):
        knees = keypoints[[13, 14], 1]
        ankles = keypoints[[15, 16], 1]
        # Stop above the ankles. This follows bent legs better than taking a
        # fixed lower portion of the person's box and avoids most footwear.
        bottom = float(torch.max(knees + 0.78 * (ankles - knees)))
    else:
        bottom = y1 + 0.88 * height

    left = max(0, int(x1 - 0.04 * width))
    right = min(rgb.width, int(x2 + 0.04 * width))
    top = max(0, int(top))
    bottom = min(rgb.height, int(bottom))
    if right - left < 16 or bottom - top < 16:
        return rgb, {
            "person_detected": False,
            "person_confidence": float(scores[index]),
            "person_box": tuple(int(value) for value in (x1, y1, x2, y2)),
            "person_area_ratio": person_area_ratio,
            "clothing_area_ratio": 1.0,
            "crop_box": (0, 0, *rgb.size),
        }

    crop_box = (left, top, right, bottom)
    return rgb.crop(crop_box), {
        "person_detected": True,
        "person_confidence": float(scores[index]),
        "person_box": tuple(int(value) for value in (x1, y1, x2, y2)),
        "person_area_ratio": person_area_ratio,
        "clothing_area_ratio": ((right - left) * (bottom - top)) / image_area,
        "crop_box": crop_box,
    }
