"""ResNet image-search test website. Run: python app.py --warmup image."""

# 1. Thu vien va cau hinh
import argparse
import hashlib
import importlib.util
import tempfile
from collections import OrderedDict
import base64
import binascii
import io
import json
import logging
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SBERT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RESNET_CLASSES = ["Tshirts", "Shirts", "Jeans", "Trousers", "Dresses", "Jackets"]
_BACKGROUND_SESSION = None
_PERSON_CLOTHING_MODULE = None


# 2. Kien truc ResNet18: giu ten model.* de doc checkpoint da huan luyen
def create_resnet(num_classes, pretrained=False):
    import torch.nn as nn
    from torchvision.models import ResNet18_Weights, resnet18

    class FashionResNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = resnet18(
                weights=ResNet18_Weights.DEFAULT if pretrained else None
            )
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        def forward(self, x):
            return self.model(x)

    return FashionResNet()


# 3. Dữ liệu, cache embedding và suy luận
ROOT = Path(__file__).resolve().parent
CACHE_VERSION = "fashion-web-v2"


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     default=str).encode("utf-8")).hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def top_indices(scores, candidates, limit):
    """Exact top-k; deterministic index order for equal scores."""
    import numpy as np
    candidates = np.asarray(candidates, dtype=np.int64)
    if not len(candidates):
        return candidates
    values = scores[candidates]
    count = min(limit, len(candidates))
    if count < len(candidates):
        threshold = np.partition(values, len(values) - count)[len(values) - count]
        keep = values >= threshold
        candidates, values = candidates[keep], values[keep]
    return candidates[np.lexsort((candidates, -values))[:count]]


def fused_rank_scores(rankings, weights, size):
    """RRF over complete candidate rankings; no approximate truncation."""
    import numpy as np
    scores = np.zeros(size, dtype=np.float64)
    for ranking, weight in zip(rankings, weights):
        if weight:
            scores[ranking] += weight / (60 + np.arange(1, len(ranking) + 1))
    return scores


def reciprocal_rank_fusion(rankings, weights, limit):
    scores = {}
    for ranking, weight in zip(rankings, weights):
        if weight == 0:
            continue
        for rank, index in enumerate(ranking, 1):
            scores[index] = scores.get(index, 0.0) + weight / (60 + rank)
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))[:limit]


def garment_views(image):
    """Global and local views used to preserve garment pattern/logo detail."""
    image = image.convert("RGB")
    width, height = image.size
    return [
        image,
        image.crop((0, 0, width, int(height * 0.62))),
        image.crop((0, int(height * 0.38), width, height)),
        image.crop((int(width * 0.15), int(height * 0.15),
                    int(width * 0.85), int(height * 0.85))),
    ]


class RecommendationEngine:
    def __init__(self):
        self.lock = threading.RLock()
        self.products = []
        self.dataset = None
        self.text_model = self.image_model = self.bert_model = None
        self.gender_head = None
        self.gender_classes = []
        self.color_embeddings = None
        self.status = "Sẵn sàng. Model sẽ tải khi bạn tìm lần đầu."
        self.device = None
        self.cache_dir = Path(os.getenv("WEB_CACHE_DIR", str(ROOT / ".cache")))
        self.disk_cache = os.getenv("WEB_DISK_CACHE", "1") != "0"
        self.cache_hits = {}
        self.query_cache = OrderedDict()
        self.bert_cache = OrderedDict()
        self.resnet_benchmark = None
        self.batch_size = max(1, min(int(os.getenv("WEB_BATCH_SIZE", "32")), 256))

    @staticmethod
    def remove_background(image):
        """Remove upload background before ResNet classification/retrieval."""
        global _BACKGROUND_SESSION
        if os.getenv("WEB_REMOVE_BACKGROUND", "0") == "0":
            return image.convert("RGB"), False
        try:
            from rembg import new_session, remove
        except ImportError as error:
            raise RuntimeError(
                "Thiếu rembg. Hãy cài lại model/Web_Test/requirements.txt."
            ) from error
        if _BACKGROUND_SESSION is None:
            model_name = os.getenv("WEB_BACKGROUND_MODEL", "u2netp")
            _BACKGROUND_SESSION = new_session(model_name)
        foreground = remove(
            image.convert("RGBA"),
            session=_BACKGROUND_SESSION,
        )
        if not hasattr(foreground, "convert"):
            from PIL import Image
            foreground = Image.open(io.BytesIO(foreground)).convert("RGBA")
        else:
            foreground = foreground.convert("RGBA")
        from PIL import Image
        canvas = Image.new("RGBA", foreground.size, (255, 255, 255, 255))
        return Image.alpha_composite(canvas, foreground).convert("RGB"), True

    @staticmethod
    def prepare_uploaded_image(image):
        """Normalize orientation/transparency and reject unusably small images."""
        from PIL import Image, ImageOps
        source = ImageOps.exif_transpose(image)
        if source.width < 32 or source.height < 32:
            raise ValueError("Ảnh quá nhỏ; mỗi chiều phải từ 32 pixel trở lên.")
        if source.mode in {"RGBA", "LA"} or "transparency" in source.info:
            rgba = source.convert("RGBA")
            canvas = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            source = Image.alpha_composite(canvas, rgba)
        return source.convert("RGB")

    @staticmethod
    def detect_clothing_region(image):
        """Crop a detected person's clothing using pretrained pose keypoints."""
        global _PERSON_CLOTHING_MODULE
        if os.getenv("WEB_DETECT_PERSON", "0") == "0":
            return image.convert("RGB"), {
                "person_detected": False,
                "person_confidence": 0.0,
                "crop_box": None,
            }
        if _PERSON_CLOTHING_MODULE is None:
            path = ROOT.parent / "combine_model" / "person_clothing.py"
            spec = importlib.util.spec_from_file_location("person_clothing", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            _PERSON_CLOTHING_MODULE = module
        return _PERSON_CLOTHING_MODULE.extract_clothing_region(image)

    @staticmethod
    def color_descriptor(image):
        """Normalized 20D HSV histogram used only for colour similarity."""
        import numpy as np
        from PIL import Image
        rgb = image.convert("RGB").resize((96, 96), Image.Resampling.BILINEAR)
        hsv = np.asarray(rgb.convert("HSV"), dtype=np.float32) / 255.0
        parts = []
        for channel, bins in enumerate((12, 4, 4)):
            histogram, _ = np.histogram(
                hsv[:, :, channel], bins=bins, range=(0.0, 1.0)
            )
            histogram = histogram.astype(np.float32)
            parts.append(histogram / max(float(histogram.sum()), 1.0))
        mean_rgb = np.asarray(rgb, dtype=np.float32).reshape(-1, 3).mean(axis=0) / 255.0
        mean_rgb /= max(float(np.linalg.norm(mean_rgb)), 1e-12)
        descriptor = np.concatenate([*parts, 3.0 * mean_rgb])
        descriptor /= max(float(np.linalg.norm(descriptor)), 1e-12)
        return descriptor

    def catalog(self):
        if self.products:
            return
        from datasets import load_dataset, Image as DatasetImage
        self.status = "Đang tải danh mục sản phẩm..."
        size = max(1, min(int(os.getenv("WEB_MAX_PRODUCTS", "3000")), 3000))
        dataset = load_dataset("ashraq/fashion-product-images-small", split="train")
        dataset = dataset.cast_column("image", DatasetImage(decode=False))
        # Read only metadata. Image bytes stay in the Arrow dataset until needed.
        metadata = dataset.remove_columns("image")
        products, seen = [], set()
        for index, row in enumerate(metadata):
            name = str(row.get("productDisplayName") or "").strip()
            if not name or name in seen:
                continue
            seen.add(name)
            row["_dataset_index"] = index
            products.append(row)
            if len(products) >= size:
                break
        if not products:
            raise ValueError("Danh mục không có sản phẩm.")
        self.dataset = dataset
        self.catalog_key = fingerprint([dataset._fingerprint, products])
        self.products = products
        self.categories = sorted({str(p.get("articleType", "")) for p in products})

    def product_image(self, index):
        from PIL import Image
        row = self.products[index]
        value = self.dataset[row["_dataset_index"]]["image"]
        source = io.BytesIO(value["bytes"]) if value.get("bytes") is not None else value["path"]
        with Image.open(source) as image:
            return image.convert("RGB")

    def setup_device(self):
        import torch
        if self.device is None:
            requested = os.getenv("WEB_DEVICE", "auto")
            if requested not in ("auto", "cpu", "cuda"):
                raise ValueError("WEB_DEVICE phải là auto, cpu hoặc cuda.")
            if requested == "cuda" and not torch.cuda.is_available():
                raise ValueError("CUDA không khả dụng trên máy này.")
            self.device = ("cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else requested
            if os.getenv("WEB_CPU_THREADS"):
                torch.set_num_threads(max(1, int(os.environ["WEB_CPU_THREADS"])))

    def cached_embeddings(self, kind, identity, dimensions, build):
        import numpy as np
        key = fingerprint([CACHE_VERSION, self.catalog_key, kind, identity])
        target = self.cache_dir / (kind + "-" + key + ".npy")
        self.cache_hits[kind] = False
        if self.disk_cache:
            try:
                array = np.load(target, allow_pickle=False, mmap_mode="r")
                if (array.shape == (len(self.products), dimensions)
                        and array.dtype == np.float32 and np.isfinite(array).all()
                        and np.allclose(np.linalg.norm(array, axis=1), 1, atol=1e-3)):
                    self.cache_hits[kind] = True
                    return array
                logging.warning("Invalid embedding cache; rebuilding %s", target.name)
            except (OSError, ValueError, EOFError):
                pass
        array = np.asarray(build(), dtype=np.float32)
        if array.shape != (len(self.products), dimensions) or not np.isfinite(array).all():
            raise ValueError("Embedding có kích thước hoặc giá trị không hợp lệ.")
        if self.disk_cache:
            temporary = None
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=self.cache_dir, suffix=".npy", delete=False) as stream:
                    temporary = Path(stream.name)
                    np.save(stream, array, allow_pickle=False)
                os.replace(temporary, target)
            except OSError:
                logging.warning("Không ghi được cache; tiếp tục dùng RAM.", exc_info=True)
            finally:
                if temporary is not None and temporary.exists():
                    temporary.unlink()
        return array

    def load_text(self):
        if self.text_model is not None:
            return
        from sentence_transformers import SentenceTransformer
        import sentence_transformers
        self.setup_device()
        self.status = "Đang tải SBERT..."
        name = os.getenv("WEB_SBERT_MODEL", SBERT_MODEL)
        model = SentenceTransformer(name, device="cpu")
        # Hash actual parameters once: cache remains correct even when a local
        # model is overwritten under the same name or no Hub commit is exposed.
        digest = hashlib.sha256()
        for key, tensor in model.state_dict().items():
            digest.update(key.encode())
            digest.update(memoryview(tensor.detach().cpu().contiguous().numpy()).cast("B"))
        model.to(self.device)
        tokenizer = model.tokenizer.backend_tokenizer.to_str()
        identity = [name, digest.hexdigest(), tokenizer, model.max_seq_length,
                    str(model), sentence_transformers.__version__, "name-type-colour-gender-brand-v1"]
        texts = [" ".join(str(p.get(key) or "") for key in
                 ("productDisplayName", "articleType", "baseColour", "gender", "brandName"))
                 for p in self.products]
        def build():
            self.status = "Đang mã hóa mô tả sản phẩm (chỉ lần đầu)..."
            return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True,
                                batch_size=self.batch_size, show_progress_bar=True)
        embeddings = self.cached_embeddings("text", identity, model.get_embedding_dimension(), build)
        self.text_embeddings, self.text_model = embeddings, model

    def load_image(self):
        if self.image_model is not None:
            return
        import numpy as np
        import torch
        import torchvision
        from torchvision import transforms
        self.setup_device()
        self.status = "Đang tải ResNet18..."
        path = Path(os.getenv("WEB_RESNET_MODEL", str(ROOT / "models/resnet_outfit.pth")))
        if not path.is_file():
            raise FileNotFoundError(f"Thiếu trọng số ResNet: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        classes = checkpoint.get("classes", RESNET_CLASSES)
        model = create_resnet(len(classes))
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        model.to(self.device).eval()
        transform = transforms.Compose([
            transforms.Resize((256, 256)), transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        backbone = torch.nn.Sequential(*list(model.model.children())[:-1])
        def build():
            features = []
            with torch.inference_mode():
                for offset in range(0, len(self.products), self.batch_size):
                    self.status = f"Đang mã hóa ảnh: {offset}/{len(self.products)}..."
                    product_images = [self.product_image(i) for i in range(
                        offset, min(offset + self.batch_size, len(self.products))
                    )]
                    pattern_enabled = os.getenv("WEB_PATTERN_SEARCH", "0") == "1"
                    batch = torch.stack([
                        transform(view) for image in product_images
                        for view in (garment_views(image) if pattern_enabled else [image])
                    ])
                    feature = backbone(batch.to(self.device)).flatten(1)
                    feature = torch.nn.functional.normalize(feature, dim=1)
                    if pattern_enabled:
                        feature = feature.reshape(len(product_images), 4, -1)
                        feature = 0.55 * feature[:, 0] + 0.45 * feature[:, 1:].mean(dim=1)
                    features.append(torch.nn.functional.normalize(feature, dim=1).cpu())
            return torch.cat(features).numpy()
        identity = [file_digest(path), classes, str(transform), torchvision.__version__,
                    torch.__version__, "pattern=" + os.getenv("WEB_PATTERN_SEARCH", "0")]
        embeddings = self.cached_embeddings("image", identity, 512, build)
        self.image_embeddings = embeddings
        def build_colors():
            self.status = "Đang mã hóa màu sắc sản phẩm..."
            return np.stack([
                self.color_descriptor(self.product_image(index))
                for index in range(len(self.products))
            ]).astype(np.float32)
        color_identity = [self.catalog_key, "hsv-12-4-4-rgbmean-v2"]
        self.color_embeddings = self.cached_embeddings(
            "color", color_identity, 23, build_colors
        )
        self.transform, self.backbone, self.classes = transform, backbone, classes
        self.image_model = model
        gender_path = ROOT / "models" / "resnet_gender.pth"
        if gender_path.is_file():
            module_path = ROOT.parent / "ResNet_Model" / "gender.py"
            spec = importlib.util.spec_from_file_location("resnet_gender", module_path)
            gender_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(gender_module)
            gender_checkpoint = torch.load(
                gender_path, map_location=self.device, weights_only=True
            )
            self.gender_head = gender_module.GenderHead(
                num_classes=len(gender_checkpoint["classes"])
            ).to(self.device)
            self.gender_head.load_state_dict(gender_checkpoint["model_state_dict"])
            self.gender_head.eval()
            self.gender_classes = list(gender_checkpoint["classes"])

    def benchmark_resnet_models(self, force=False):
        """Return the leakage-free ResNet audit when available."""

        with self.lock:
            audit_path = ROOT.parent / "ResNet_Model" / "resnet_audit.json"
            if audit_path.is_file():
                audit = json.loads(audit_path.read_text(encoding="utf-8"))
                classification = audit["classification"]
                cosine = audit["cosine"]
                self.resnet_benchmark = {
                    "rows": [{
                        "model": "fashion fine-tuned · unseen audit",
                        "samples": audit["evaluation_set"]["samples"],
                        "classification_accuracy": classification["accuracy"],
                        "retrieval_top1": cosine["retrieval_top1"],
                        "retrieval_top5": cosine["retrieval_top5"],
                        "average_confidence": classification["average_confidence"],
                        "milliseconds_per_image": None,
                        "status": "ok",
                    }],
                    "classes": audit["classes"],
                    "checkpoint": str(ROOT / "models" / "resnet_outfit.pth"),
                    "scope": audit["evaluation_set"]["policy"],
                    "includes_upload_pipeline": False,
                    "per_class": classification["per_class"],
                }
                return self.resnet_benchmark
            import numpy as np
            import torch
            from torchvision import transforms
            if self.resnet_benchmark is not None and not force:
                return self.resnet_benchmark
            self.catalog()
            self.setup_device()
            checkpoint_path = Path(os.getenv(
                "WEB_RESNET_MODEL", str(ROOT / "models/resnet_outfit.pth")
            ))
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
            classes = list(checkpoint.get("classes", RESNET_CLASSES))
            eligible = [
                index for index, product in enumerate(self.products)
                if product.get("articleType") in classes
            ]
            if len(eligible) < 6:
                raise ValueError("Không đủ ảnh thuộc các class ResNet để benchmark.")
            requested = max(6, int(os.getenv("WEB_RESNET_BENCHMARK_SAMPLES", "120")))
            sample_count = min(requested, len(eligible))
            positions = np.linspace(0, len(eligible) - 1, sample_count, dtype=int)
            indices = [eligible[position] for position in positions]
            images = [self.product_image(index) for index in indices]
            labels = [self.products[index].get("articleType", "") for index in indices]
            custom_root = Path(os.getenv(
                "WEB_RESNET_CUSTOM_DATA",
                str(ROOT.parent / "ResNet_Model" / "custom_data"),
            ))
            custom_limit = max(1, int(os.getenv(
                "WEB_RESNET_BENCHMARK_CUSTOM_PER_CLASS", "20"
            )))
            from PIL import Image
            for category in classes:
                directory = custom_root / category
                paths = sorted(
                    path for path in directory.glob("**/*")
                    if path.is_file() and path.suffix.lower() in {
                        ".jpg", ".jpeg", ".png", ".webp"
                    }
                )[:custom_limit] if directory.is_dir() else []
                for path in paths:
                    with Image.open(path) as source:
                        images.append(source.convert("RGB"))
                    labels.append(category)
            sample_count = len(labels)
            transform = transforms.Compose([
                transforms.Resize((256, 256)), transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                ),
            ])
            tensors = torch.stack([transform(image) for image in images])
            rows = []
            variants = (
                ("baseline (untrained head)", False, None),
                ("ImageNet pretrained (feature only)", True, None),
                ("fashion fine-tuned", False,
                 checkpoint.get("model_state_dict", checkpoint)),
            )
            for name, pretrained, state in variants:
                self.status = f"Đang benchmark ResNet18 {name}..."
                started = time.perf_counter()
                try:
                    torch.manual_seed(42)
                    model = create_resnet(len(classes), pretrained=pretrained)
                    if state is not None:
                        model.load_state_dict(state)
                    model = model.to(self.device).eval()
                    backbone = torch.nn.Sequential(*list(model.model.children())[:-1])
                    feature_parts, probability_parts = [], []
                    with torch.inference_mode():
                        for offset in range(0, len(tensors), self.batch_size):
                            batch = tensors[offset:offset + self.batch_size].to(self.device)
                            features = backbone(batch).flatten(1)
                            feature_parts.append(
                                torch.nn.functional.normalize(features, dim=1).cpu()
                            )
                            probability_parts.append(
                                model.model.fc(features).softmax(-1).cpu()
                            )
                    features = torch.cat(feature_parts).numpy()
                    probabilities = torch.cat(probability_parts).numpy()
                    predictions = [classes[index] for index in probabilities.argmax(1)]
                    similarities = features @ features.T
                    np.fill_diagonal(similarities, -np.inf)
                    ranked = np.argsort(-similarities, axis=1, kind="stable")
                    top1 = np.mean([
                        labels[ranking[0]] == labels[row]
                        for row, ranking in enumerate(ranked)
                    ])
                    top5 = np.mean([
                        labels[row] in {labels[index] for index in ranking[:5]}
                        for row, ranking in enumerate(ranked)
                    ])
                    rows.append({
                        "model": name,
                        "samples": sample_count,
                        # Baseline/ImageNet heads are newly initialized for
                        # these six classes. Reporting their random-head
                        # accuracy as pretrained accuracy is misleading.
                        "classification_accuracy": (
                            float(np.mean(
                                np.asarray(predictions) == np.asarray(labels)
                            )) if state is not None else None
                        ),
                        "retrieval_top1": float(top1),
                        "retrieval_top5": float(top5),
                        "average_confidence": (
                            float(probabilities.max(1).mean())
                            if state is not None else None
                        ),
                        "milliseconds_per_image": float(
                            (time.perf_counter() - started) * 1000 / sample_count
                        ),
                        "status": "ok",
                    })
                except Exception as error:
                    logging.exception("ResNet benchmark failed: %s", name)
                    rows.append({
                        "model": name,
                        "samples": sample_count,
                        "status": "error",
                        "error": str(error),
                    })
            self.resnet_benchmark = {
                "rows": rows,
                "classes": classes,
                "checkpoint": str(checkpoint_path),
                "scope": "catalog_images_only",
                "includes_upload_pipeline": False,
            }
            self.status = "Benchmark ResNet hoàn tất. ResNet Web Test sẵn sàng."
            return self.resnet_benchmark

    @staticmethod
    def remember(cache, key, value):
        cache[key] = value
        cache.move_to_end(key)
        if len(cache) > 128:
            cache.popitem(last=False)
        return value

    def query_embedding(self, query):
        if query in self.query_cache:
            self.query_cache.move_to_end(query)
            return self.query_cache[query]
        vector = self.text_model.encode(query, normalize_embeddings=True, convert_to_numpy=True)
        return self.remember(self.query_cache, query, vector)

    def classify_text(self, query):
        if query in self.bert_cache:
            self.bert_cache.move_to_end(query)
            return dict(self.bert_cache[query])
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.setup_device()
        if self.bert_model is None:
            self.status = "Đang tải BERT phân loại sản phẩm..."
            path = ROOT / "models/bert_fashion_model"
            tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True)
            self.bert_tokenizer = tokenizer
            self.bert_model = model.to(self.device).eval()
        tokens = self.bert_tokenizer(query, return_tensors="pt", truncation=True, max_length=64)
        with torch.inference_mode():
            probabilities = self.bert_model(**{k: v.to(self.device) for k, v in tokens.items()}).logits.softmax(-1)[0]
        index = int(probabilities.argmax())
        result = {"category": self.bert_model.config.id2label[index],
                  "confidence": float(probabilities[index])}
        return dict(self.remember(self.bert_cache, query, result))

    def search(self, query="", image=None, mode="text", top_k=12, category="",
               text_weight=0.5, use_bert=False):
        if mode != "image":
            raise ValueError("SBERT và Combine đang tạm tắt; chỉ dùng ResNet image search.")
        query = query.strip()
        if mode in ("text", "combined") and not query:
            raise ValueError("Vui lòng nhập mô tả sản phẩm.")
        if mode in ("image", "combined") and image is None:
            raise ValueError("Vui lòng chọn ảnh sản phẩm.")
        if not 1 <= top_k <= 48 or not 0 <= text_weight <= 1:
            raise ValueError("Số kết quả hoặc trọng số không hợp lệ.")
        import numpy as np
        with self.lock:
            self.catalog()
            candidates = np.asarray([i for i, p in enumerate(self.products)
                                     if not category or p.get("articleType") == category], dtype=np.int64)
            text_scores = image_scores = color_scores = None
            metadata = {}
            if len(candidates):
                if mode in ("text", "combined"):
                    self.load_text()
                    text_scores = self.text_embeddings @ self.query_embedding(query)
                    if use_bert:
                        metadata["bert"] = self.classify_text(query)
                if mode in ("image", "combined"):
                    import torch
                    from PIL import ImageOps
                    self.load_image()
                    image = self.prepare_uploaded_image(image)
                    person_image, person_detection = self.detect_clothing_region(image)
                    processed_image, background_removed = self.remove_background(person_image)
                    # Dual path protects product-only images from rembg errors:
                    # retain the original signal and add a smaller foreground signal.
                    base_views = [person_image]
                    if background_removed:
                        base_views.append(processed_image)
                    views = base_views + [ImageOps.mirror(view) for view in base_views]
                    tensor = torch.stack([
                        self.transform(view) for view in views
                    ]).to(self.device)
                    with torch.inference_mode():
                        # FC needs raw features, not normalized retrieval features.
                        view_features = self.backbone(tensor).flatten(1)
                        classification_logits = self.image_model.model.fc(
                            view_features
                        ).mean(dim=0, keepdim=True)
                        probabilities = classification_logits.softmax(-1)[0]
                        normalized = torch.nn.functional.normalize(view_features, dim=1)
                        features = normalized[:1]
                        if background_removed:
                            features = 0.75 * normalized[:1] + 0.25 * normalized[1:2]
                        vector = torch.nn.functional.normalize(features, dim=1)
                        gender_probabilities = None
                        if self.gender_head is not None:
                            # Gender head was trained on the original single-view
                            # embedding; keep inference in the same distribution.
                            gender_vector = normalized[0].unsqueeze(0)
                            gender_probabilities = self.gender_head(
                                gender_vector
                            ).softmax(-1)[0]
                    metadata["resnet"] = {
                        "category": self.classes[int(probabilities.argmax())],
                        "confidence": float(probabilities.max()),
                        "background_removed": background_removed,
                        "person_detection": person_detection,
                        "tta": "original+foreground+horizontal_flips" if background_removed
                               else "original+horizontal_flip",
                        "foreground_embedding_weight": 0.25 if background_removed else 0.0,
                        "input_size": [image.width, image.height],
                    }
                    if gender_probabilities is not None:
                        gender_index = int(gender_probabilities.argmax())
                        gender_confidence = float(gender_probabilities[gender_index])
                        predicted_gender = self.gender_classes[gender_index]
                        metadata["resnet"]["gender"] = predicted_gender
                        metadata["resnet"]["gender_confidence"] = gender_confidence
                        metadata["resnet"]["gender_probabilities"] = {
                            name: float(gender_probabilities[index])
                            for index, name in enumerate(self.gender_classes)
                        }
                    image_scores = self.image_embeddings @ vector.cpu().numpy()[0]
                    color_query = self.color_descriptor(processed_image)
                    color_scores = self.color_embeddings @ color_query
                    color_weight = float(os.getenv("WEB_COLOR_WEIGHT", "0.10"))
                    color_weight = max(0.0, min(color_weight, 0.75))
                    image_scores = (
                        (1.0 - color_weight) * image_scores
                        + color_weight * color_scores
                    )
                    metadata["resnet"]["color_weight"] = color_weight
                    if gender_probabilities is not None and gender_confidence >= 0.80:
                        gender_weight = float(os.getenv("WEB_GENDER_WEIGHT", "0.05"))
                        gender_scores = np.asarray([
                            gender_confidence if (
                                ("Men" if p.get("gender", "") in {"Men", "Boys"}
                                 else "Women" if p.get("gender", "") in {"Women", "Girls"}
                                 else "Unisex") in {predicted_gender, "Unisex"}
                            ) else 0.0
                            for p in self.products
                        ], dtype=np.float32)
                        image_scores = image_scores + gender_weight * gender_scores
                        metadata["resnet"]["gender_weight"] = gender_weight
                    # Classification is a soft prior; cosine similarity stays
                    # dominant and no product category is excluded.
                    category_weight = float(os.getenv("WEB_CATEGORY_WEIGHT", "0.15"))
                    class_probabilities = {
                        name: float(probabilities[index])
                        for index, name in enumerate(self.classes)
                    }
                    image_scores = image_scores + category_weight * np.asarray([
                        class_probabilities.get(p.get("articleType", ""), 0.0)
                        for p in self.products
                    ], dtype=np.float32)
            if not len(candidates):
                selected = []
            elif mode == "combined":
                rankings = [candidates[np.argsort(-scores[candidates], kind="stable")]
                            for scores in (text_scores, image_scores)]
                fused = fused_rank_scores(rankings, (text_weight, 1 - text_weight), len(self.products))
                indices = top_indices(fused, candidates, top_k)
                selected = [(int(i), float(fused[i])) for i in indices]
            else:
                scores = text_scores if text_scores is not None else image_scores
                selected = [(int(i), float(scores[i])) for i in top_indices(scores, candidates, top_k)]
            results = []
            for index, score in selected:
                p = self.products[index]
                results.append({
                    "id": index, "name": p["productDisplayName"],
                    "category": p.get("articleType", ""), "colour": p.get("baseColour", ""),
                    "gender": p.get("gender", ""), "image": f"/api/products/{index}/image",
                    "score": score,
                    "text_score": float(text_scores[index]) if text_scores is not None else None,
                    "image_score": float(image_scores[index]) if image_scores is not None else None,
                    "color_score": float(color_scores[index]) if color_scores is not None else None})
            self.status = f"Đã sẵn sàng · {len(self.products)} sản phẩm · {self.device or 'chưa tải model'}"
            return {"results": results, "metadata": metadata, "mode": mode,
                    "catalog_size": len(self.products), "categories": self.categories,
                    "cache_hits": dict(self.cache_hits), "device": self.device}

    def warmup(self, mode="combined", use_bert=False):
        with self.lock:
            self.catalog()
            image = self.product_image(0) if mode != "text" else None
            return self.search(query="shirt", image=image, mode=mode, top_k=1, use_bert=use_bert)


# 4. Giao dien: HTML, CSS va JavaScript
PAGE = r"""<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ResNet Web Test - Tìm sản phẩm</title>
<style>
* { box-sizing: border-box; }
body { max-width: 1100px; margin: 24px auto; padding: 0 16px; font: 14px Arial, sans-serif; color: #222; }
h1 { font-size: 24px; }
h2 { font-size: 18px; }
form { max-width: 650px; }
label { display: block; margin: 14px 0 6px; }
textarea, select, button { font: inherit; padding: 8px; }
textarea { width: 100%; resize: vertical; }
button { cursor: pointer; }
button:disabled { cursor: wait; }
.tabs { display: flex; gap: 8px; margin: 16px 0; flex-wrap: wrap; }
.tabs .active { background: #ddd; border: 2px solid #333; }
.options { display: flex; gap: 24px; flex-wrap: wrap; }
#preview { display: block; max-width: 200px; max-height: 180px; margin: 12px 0; }
#weight { width: 260px; max-width: 100%; }
#submit { margin-top: 16px; }
#status, #result-info, small { color: #555; }
#error { color: #a00; margin: 16px 0; }
#grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(210px, 1fr)); gap: 16px; }
.card { border: 1px solid #ccc; padding: 12px; }
.card img { width: 100%; height: 180px; object-fit: contain; }
.card h3 { font-size: 14px; }
.score { margin: 10px 0; }
.benchmark-wrap { overflow-x: auto; margin: 12px 0 24px; }
.benchmark { width: 100%; border-collapse: collapse; }
.benchmark th, .benchmark td { border: 1px solid #ccc; padding: 8px; text-align: right; }
.benchmark th:first-child, .benchmark td:first-child { text-align: left; }
.benchmark th { background: #f3f3f3; }
[hidden] { display: none !important; }
</style>
</head>
<body>
<h1>ResNet Web Test - Tìm sản phẩm</h1>
<section aria-labelledby="benchmark-title">
<h2 id="benchmark-title">Kiểm thử ResNet18 trên ảnh catalog</h2>
<p>Baseline, pretrained và trained được kiểm thử trước trên cùng tập ảnh. Combine dùng checkpoint trained.</p>
<button id="rerun-benchmark" type="button">Chạy lại benchmark</button>
<p id="benchmark-status" role="status">Đang chuẩn bị bảng thông số...</p>
<div class="benchmark-wrap">
<table class="benchmark">
<thead><tr><th>Model</th><th>Số ảnh</th><th>Classification Accuracy</th><th>Retrieval Top-1</th><th>Retrieval Top-5</th><th>Confidence</th><th>ms/ảnh</th></tr></thead>
<tbody id="benchmark-body"></tbody>
</table>
</div>
</section>
<hr>
<form id="search-form">
<div class="tabs" role="group" aria-label="Chế độ tìm kiếm">
<button type="button" data-mode="text" hidden disabled aria-pressed="false">Mô tả (SBERT) · Tạm tắt</button>
<button type="button" data-mode="image" class="active" aria-pressed="true">Ảnh (ResNet18)</button>
<button type="button" data-mode="combined" hidden disabled aria-pressed="false">Kết hợp (RRF) · Tạm tắt</button>
</div>
<div id="text-field" hidden>
<label for="query">Mô tả sản phẩm</label>
<textarea id="query" rows="3" maxlength="1000" placeholder="Ví dụ: áo sơ mi xanh dành cho nam"></textarea>
</div>
<div id="image-field">
<label for="upload">Ảnh sản phẩm (JPG, PNG, WEBP; tối đa 8 MB)</label>
<input id="upload" type="file" accept="image/jpeg,image/png,image/webp">
<img id="preview" alt="Ảnh đã chọn" hidden>
<button type="button" id="remove-image" hidden>Xóa ảnh</button>
</div>
<div class="options">
<div><label for="category">Loại sản phẩm</label><select id="category"><option value="">Tất cả sản phẩm</option></select></div>
<div><label for="top-k">Số kết quả</label><select id="top-k"><option>6</option><option selected>12</option><option>24</option><option>48</option></select></div>
</div>
<div id="weight-field" hidden>
<label for="weight">Trọng số mô tả: <output id="weight-label">50%</output></label>
<input id="weight" type="range" min="0" max="100" value="50">
<small>Phần còn lại là trọng số ảnh.</small>
</div>
<label id="bert-field" hidden><input id="use-bert" type="checkbox"> Phân loại thêm bằng BERT</label>
<button id="submit" type="submit">Tìm sản phẩm</button>
<p>Lần tìm đầu cần tải model và xử lý dữ liệu.</p>
<p id="status" role="status" aria-live="polite"></p>
</form>
<hr>
<h2>Kết quả</h2>
<p id="result-count">Chưa có tìm kiếm</p>
<p id="result-info"></p>
<div id="error" role="alert" hidden></div>
<div id="metadata"></div>
<div id="empty"><h3>Chưa có kết quả</h3><p>Nhập mô tả hoặc chọn ảnh rồi bấm Tìm sản phẩm.</p></div>
<div id="grid" aria-live="polite"></div>
<script>
const $ = id => document.getElementById(id);
let mode = "image", imageData = null, busy = false, imageVersion = 0;
const showError = message => { $("error").textContent = message; $("error").hidden = !message; };
document.querySelectorAll("[data-mode]").forEach(button => button.onclick = () => {
  if (busy) return;
  mode = button.dataset.mode;
  document.querySelectorAll("[data-mode]").forEach(b => { b.classList.toggle("active", b === button); b.setAttribute("aria-pressed", String(b === button)); });
  $("text-field").hidden = mode === "image";
  $("image-field").hidden = mode === "text";
  $("weight-field").hidden = mode !== "combined";
  $("bert-field").hidden = mode === "image";
});
$("weight").oninput = () => $("weight-label").textContent = $("weight").value + "%";
function clearImage() {
  imageVersion++; imageData = null; $("upload").value = "";
  $("preview").hidden = true; $("preview").removeAttribute("src"); $("remove-image").hidden = true;
}
$("remove-image").onclick = clearImage;
$("upload").onchange = async () => {
  const file = $("upload").files[0];
  const version = ++imageVersion;
  imageData = null; $("preview").hidden = true; $("remove-image").hidden = true;
  if (!file) return;
  if (file.size > 8 * 1024 * 1024 || !["image/jpeg", "image/png", "image/webp"].includes(file.type)) {
    clearImage(); showError("Hãy chọn ảnh JPG, PNG hoặc WEBP dưới 8 MB."); return;
  }
  const reader = new FileReader();
  reader.onload = () => {
    if (version !== imageVersion) return;
    imageData = String(reader.result).split(",")[1];
    $("preview").src = reader.result; $("preview").hidden = false; $("remove-image").hidden = false; showError("");
  };
  reader.onerror = () => showError("Không đọc được ảnh. Vui lòng chọn lại.");
  reader.readAsDataURL(file);
};
function element(tag, text, className) {
  const node = document.createElement(tag); node.textContent = text;
  if (className) node.className = className;
  return node;
}
async function loadBenchmark(force = false) {
  const button = $("rerun-benchmark");
  button.disabled = true;
  $("benchmark-status").textContent = "Đang kiểm thử từng ResNet...";
  try {
    const response = await fetch("/api/resnet-benchmark" + (force ? "?force=1" : ""));
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Không chạy được benchmark ResNet.");
    $("benchmark-body").replaceChildren();
    data.rows.forEach(row => {
      const tr = document.createElement("tr");
      if (row.status !== "ok") {
        tr.append(element("td", row.model), element("td", row.samples), element("td", row.error || "Lỗi"));
        const errorCell = tr.lastChild; errorCell.colSpan = 5;
      } else {
        const values = [row.model, row.samples,
          row.classification_accuracy == null ? "N/A · chưa fine-tune" :
            (row.classification_accuracy * 100).toFixed(2) + "%",
          (row.retrieval_top1 * 100).toFixed(2) + "%",
          (row.retrieval_top5 * 100).toFixed(2) + "%",
          row.average_confidence == null ? "N/A" :
            (row.average_confidence * 100).toFixed(2) + "%",
          row.milliseconds_per_image == null ? "N/A" :
            row.milliseconds_per_image.toFixed(2)];
        values.forEach(value => tr.append(element("td", value)));
      }
      $("benchmark-body").append(tr);
    });
    $("benchmark-status").textContent = "Đã kiểm thử " + data.rows.length + " biến thể · " + data.classes.join(", ");
  } catch (error) {
    $("benchmark-status").textContent = error.message;
  } finally {
    button.disabled = false;
  }
}
$("rerun-benchmark").onclick = () => loadBenchmark(true);
function render(data) {
  $("grid").replaceChildren(); $("metadata").replaceChildren();
  $("empty").hidden = data.results.length > 0;
  if (!data.results.length) {
    $("empty").querySelector("h3").textContent = "Chưa tìm thấy sản phẩm";
    $("empty").querySelector("p").textContent = "Thử đổi mô tả hoặc chọn tất cả loại sản phẩm.";
  }
  const method = {text: "SBERT", image: "ResNet18", combined: "SBERT + ResNet18 · RRF"}[data.mode];
  $("result-count").textContent = data.results.length + " sản phẩm";
  $("result-info").textContent = method + " · " + data.catalog_size + " sản phẩm trong danh mục · " + data.elapsed + " giây";
  for (const [name, prediction] of Object.entries(data.metadata)) {
    const background = prediction.background_removed ? " · Đã xóa nền" : "";
    const gender = prediction.gender ? " · " + prediction.gender + " " +
      (prediction.gender_confidence * 100).toFixed(1) + "%" : "";
    $("metadata").append(element("p", name.toUpperCase() + " nhận diện: " + prediction.category + " · Độ tin cậy " + (prediction.confidence * 100).toFixed(1) + "%" + gender + background));
  }
  data.results.forEach((item, index) => {
    const card = element("article", "", "card");
    const img = document.createElement("img"); img.src = item.image; img.alt = item.name; img.loading = "lazy";
    const body = element("div", "", "card-body");
    body.append(element("span", item.category, "category"), element("h3", item.name), element("small", [item.colour, item.gender].filter(Boolean).join(" · ")));
    body.append(element("div", "#" + (index + 1) + " · " + (data.mode === "combined" ? "Điểm RRF " + item.score.toFixed(5) : "Cosine " + item.score.toFixed(3)), "score"));
    if (data.mode === "combined") body.append(element("small", "Mô tả " + item.text_score.toFixed(3) + " · Ảnh " + item.image_score.toFixed(3)));
    card.append(img, body); $("grid").append(card);
  });
  const previous = $("category").value;
  $("category").replaceChildren(new Option("Tất cả sản phẩm", ""));
  data.categories.forEach(category => $("category").add(new Option(category, category)));
  $("category").value = previous;
}
$("search-form").onsubmit = async event => {
  event.preventDefault(); if (busy) return;
  const query = $("query").value.trim();
  if (mode !== "image" && !query) return showError("Nhập mô tả sản phẩm bạn muốn tìm.");
  if (mode !== "text" && !imageData) return showError("Vui lòng chọn và chờ ảnh tải xong.");
  const payload = {mode, query, image: mode === "text" ? null : imageData, top_k: Number($("top-k").value),
    category: $("category").value, text_weight: Number($("weight").value) / 100, use_bert: $("use-bert").checked};
  busy = true; showError(""); $("submit").textContent = "Đang tìm sản phẩm...";
  $("search-form").setAttribute("aria-busy", "true");
  const controls = [...document.querySelectorAll("form button,form input,form textarea,form select")];
  controls.forEach(control => control.disabled = true);
  $("status").textContent = "Đang chuẩn bị model và danh mục...";
  const timer = setInterval(async () => {
    try { const response = await fetch("/api/status"); const data = await response.json(); if (busy) $("status").textContent = data.message; } catch {}
  }, 1500);
  try {
    const response = await fetch("/api/search", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Không tìm kiếm được sản phẩm.");
    render(data); $("status").textContent = "Đã tìm xong.";
  } catch (error) {
    showError(error.message || "Mất kết nối đến máy chủ."); $("status").textContent = "Tìm kiếm chưa hoàn tất. Vui lòng thử lại.";
  } finally {
    clearInterval(timer); busy = false; controls.forEach(control => control.disabled = false);
    $("search-form").removeAttribute("aria-busy"); $("submit").textContent = "Tìm sản phẩm";
  }
};

loadBenchmark();

</script>
</body>
</html>
"""

# 5. HTTP API va khoi dong web
ENGINE = RecommendationEngine()
MAX_BODY = 12 * 1024 * 1024


class Handler(BaseHTTPRequestHandler):
    def send(self, status, body, content_type="application/json; charset=utf-8"):
        if not isinstance(body, bytes):
            body = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/api/status":
            return self.send(200, {"message": ENGINE.status, "catalog_size": len(ENGINE.products)})
        if path == "/api/resnet-benchmark":
            try:
                force = "force=1" in self.path.partition("?")[2]
                return self.send(200, ENGINE.benchmark_resnet_models(force=force))
            except Exception as error:
                logging.exception("ResNet benchmark failed")
                return self.send(503, {"error": str(error)})
        if path.startswith("/api/products/") and path.endswith("/image"):
            try:
                index = int(path.split("/")[3])
                if index < 0 or index >= len(ENGINE.products):
                    raise IndexError()
                image = ENGINE.product_image(index)
                image.thumbnail((480, 600))
                output = io.BytesIO()
                image.save(output, format="JPEG", quality=85)
                return self.send(200, output.getvalue(), "image/jpeg")
            except (ValueError, IndexError, KeyError, OSError):
                return self.send(404, {"error": "Không tìm thấy tài nguyên."})
        if path == "/":
            return self.send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        self.send(404, {"error": "Không tìm thấy tài nguyên."})

    def do_POST(self):
        if self.path != "/api/search":
            return self.send(404, {"error": "Không tìm thấy tài nguyên."})
        origin = self.headers.get("Origin")
        if origin and origin != f"http://{self.headers.get('Host')}":
            return self.send(403, {"error": "Nguồn yêu cầu không hợp lệ."})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_BODY:
                return self.send(413, {"error": "Ảnh quá lớn. Chọn ảnh dưới 8 MB."})
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError("Dữ liệu yêu cầu không hợp lệ.")
            query = data.get("query", "")
            category = data.get("category", "")
            mode = data.get("mode", "image")
            if not all(isinstance(v, str) for v in (query, category, mode)) or len(query) > 1000:
                raise ValueError("Mô tả tối đa 1.000 ký tự.")
            image = None
            if data.get("image"):
                from PIL import Image, UnidentifiedImageError
                encoded = data["image"]
                if not isinstance(encoded, str):
                    raise ValueError("Ảnh không hợp lệ.")
                try:
                    raw = base64.b64decode(encoded, validate=True)
                    if len(raw) > 8 * 1024 * 1024:
                        raise ValueError("Ảnh phải nhỏ hơn 8 MB.")
                    image = Image.open(io.BytesIO(raw))
                    if image.width * image.height > 20_000_000:
                        raise ValueError("Ảnh tối đa 20 triệu điểm ảnh.")
                    image.load()
                    image = image.convert("RGB")
                except (binascii.Error, UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
                    raise ValueError("Không đọc được ảnh. Hãy dùng JPG, PNG hoặc WEBP.") from exc
            start = time.monotonic()
            result = ENGINE.search(query=query, image=image, mode=mode,
                                   top_k=int(data.get("top_k", 12)), category=category,
                                   text_weight=float(data.get("text_weight", 0.5)),
                                   use_bert=data.get("use_bert") is True)
            result["elapsed"] = round(time.monotonic() - start, 2)
            self.send(200, result)
        except (ValueError, TypeError) as exc:
            self.send(400, {"error": str(exc)})
        except Exception:
            logging.exception("Model inference failed")
            ENGINE.status = "Chưa tải được model hoặc dữ liệu. Kiểm tra terminal và thử lại."
            self.send(503, {"error": ENGINE.status})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--warmup", choices=("image",))
    parser.add_argument("--prepare-only", action="store_true",
                        help="Build/validate both indexes, then exit.")
    parser.add_argument("--benchmark-resnet", action="store_true",
                        help="Benchmark baseline, pretrained and trained ResNet, then exit.")
    args = parser.parse_args()
    if args.benchmark_resnet:
        print(json.dumps(
            ENGINE.benchmark_resnet_models(force=True),
            ensure_ascii=False,
            indent=2,
        ))
        return
    if args.prepare_only:
        ENGINE.warmup(args.warmup or "image")
        print("Index ready:", ENGINE.cache_hits, flush=True)
        return
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    if args.warmup:
        def prepare():
            try:
                ENGINE.warmup(args.warmup)
            except Exception:
                logging.exception("Warmup failed")
                ENGINE.status = "Chưa tải được model hoặc dữ liệu. Kiểm tra terminal và thử lại."
        threading.Thread(target=prepare, daemon=True).start()
    print(f"ResNet Web Test: http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
