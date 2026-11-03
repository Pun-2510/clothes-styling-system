"""Frozen pretrained encoders for the controlled retrieval benchmark."""

from __future__ import annotations

from pathlib import Path

import numpy as np


SBERT_MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
BERT_MODEL_ID = "google-bert/bert-base-uncased"


def _normalize(matrix):
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("The encoder returned an invalid embedding matrix.")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if (norms == 0).any():
        raise ValueError("The encoder returned a zero embedding.")
    return matrix / norms


def encode_sbert(texts, device="cpu", batch_size=16, model_id=SBERT_MODEL_ID):
    """Encode product name and description with the existing SBERT baseline."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "sentence-transformers is required. Install experiments/model_comparison/requirements.txt."
        ) from error

    model = SentenceTransformer(model_id, device=device)
    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    metadata = {
        "architecture": "Sentence-BERT",
        "source": model_id,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "embedding_dimension": int(embeddings.shape[1]),
    }
    return _normalize(embeddings), metadata


def encode_bert(
    texts,
    device="cpu",
    batch_size=16,
    model_id=BERT_MODEL_ID,
    max_length=128,
):
    """Encode text with a frozen pretrained BERT and mask-aware mean pooling."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device).eval()
    batches = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            inputs = tokenizer(
                list(texts[start:start + batch_size]),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            output = model(**inputs, return_dict=True).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).to(output.dtype)
            pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            batches.append(torch.nn.functional.normalize(pooled, dim=-1).cpu().numpy())
    embeddings = np.concatenate(batches)
    metadata = {
        "architecture": "BERT",
        "source": model_id,
        "pretrained": True,
        "pooling": "attention-mask mean pooling",
        "max_length": int(max_length),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "embedding_dimension": int(embeddings.shape[1]),
        "category_in_evaluation_text": False,
    }
    del model, tokenizer
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return _normalize(embeddings), metadata


def rank_tfidf(
    texts,
    query_positions,
    query_ids,
    gallery_ids,
    max_features=20000,
):
    """Fit TF-IDF on the gallery and rank held-out rows by cosine similarity."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=max_features,
        norm="l2",
        dtype=np.float32,
    )
    gallery = vectorizer.fit_transform(list(texts))
    scores = (gallery[np.asarray(query_positions)] @ gallery.T).toarray()
    query_ids = np.asarray(query_ids).astype(str)
    gallery_ids = np.asarray(gallery_ids).astype(str)
    for index, query_id in enumerate(query_ids):
        scores[index, gallery_ids == query_id] = -np.inf
    rankings = np.argsort(-scores, axis=1, kind="stable")
    metadata = {
        "architecture": "TF-IDF",
        "source": "fitted on category-free gallery text",
        "pretrained": False,
        "ngram_range": [1, 2],
        "max_features": int(max_features),
        "embedding_dimension": int(gallery.shape[1]),
        "vocabulary_size": len(vectorizer.vocabulary_),
        "category_in_evaluation_text": False,
    }
    return rankings, metadata


def encode_clip_text(texts, source, device="cpu", batch_size=16):
    """Encode the same category-free text with a CLIP text encoder."""
    import torch
    from transformers import AutoProcessor, CLIPModel

    source_path = Path(source)
    source_name = str(source_path.resolve()) if source_path.exists() else str(source)
    processor = AutoProcessor.from_pretrained(source_name)
    model = CLIPModel.from_pretrained(source_name).to(device).eval()
    batches = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            inputs = processor(
                text=list(texts[start:start + batch_size]),
                padding=True,
                truncation=True,
                max_length=model.config.text_config.max_position_embeddings,
                return_tensors="pt",
            ).to(device)
            output = model.text_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                return_dict=True,
            )
            features = model.text_projection(output.pooler_output)
            batches.append(torch.nn.functional.normalize(features, dim=-1).cpu().numpy())
    embeddings = np.concatenate(batches)
    metadata = {
        "architecture": "CLIP text encoder",
        "source": source_name,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "embedding_dimension": int(embeddings.shape[1]),
        "category_in_evaluation_text": False,
    }
    del model, processor
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return _normalize(embeddings), metadata


def encode_clip_images(image_paths, source, device="cpu", batch_size=16):
    """Encode held-out images with the selected CLIP vision encoder."""
    import torch
    from PIL import Image
    from transformers import AutoProcessor, CLIPModel

    source_path = Path(source)
    source_name = str(source_path.resolve()) if source_path.exists() else str(source)
    processor = AutoProcessor.from_pretrained(source_name)
    model = CLIPModel.from_pretrained(source_name).to(device).eval()
    paths = [Path(path) for path in image_paths]
    batches = []
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            images = []
            for path in paths[start:start + batch_size]:
                with Image.open(path) as image:
                    images.append(image.convert("RGB").copy())
            inputs = processor(images=images, return_tensors="pt").to(device)
            output = model.vision_model(
                pixel_values=inputs["pixel_values"],
                return_dict=True,
            )
            features = model.visual_projection(output.pooler_output)
            batches.append(torch.nn.functional.normalize(features, dim=-1).cpu().numpy())
    embeddings = np.concatenate(batches)
    metadata = {
        "architecture": "CLIP vision encoder",
        "source": source_name,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "embedding_dimension": int(embeddings.shape[1]),
    }
    del model, processor
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return _normalize(embeddings), metadata


def _load_resnet18(device, source, checkpoint):
    import torch
    from torchvision import models, transforms

    if source == "imagenet":
        weights = models.ResNet18_Weights.DEFAULT
        network = models.resnet18(weights=weights)
        preprocess = weights.transforms()
        source_name = str(weights)
    else:
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"ResNet18 checkpoint not found: {checkpoint}. "
                "Use --resnet-source imagenet or supply --resnet-checkpoint."
            )
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = saved.get("model_state_dict", saved)
        state = {key.removeprefix("model."): value for key, value in state.items()}
        if "fc.weight" not in state:
            raise ValueError("The checkpoint is not a compatible ResNet18 classifier.")
        network = models.resnet18(weights=None)
        network.fc = torch.nn.Linear(network.fc.in_features, state["fc.weight"].shape[0])
        network.load_state_dict(state)
        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        source_name = str(checkpoint.resolve())
    parameters = sum(parameter.numel() for parameter in network.parameters())
    encoder = torch.nn.Sequential(*list(network.children())[:-1]).to(device).eval()
    return encoder, preprocess, source_name, parameters


def encode_resnet18(image_paths, device="cpu", batch_size=16,
                    source="project-trained", checkpoint=None):
    """Extract 512-dimensional penultimate-layer ResNet18 image features."""
    import torch
    from PIL import Image

    encoder, preprocess, source_name, parameters = _load_resnet18(
        device, source, checkpoint
    )
    paths = [Path(path) for path in image_paths]
    batches = []
    for start in range(0, len(paths), batch_size):
        tensors = []
        for path in paths[start:start + batch_size]:
            with Image.open(path) as image:
                tensors.append(preprocess(image.convert("RGB")))
        inputs = torch.stack(tensors).to(device)
        with torch.inference_mode():
            batches.append(encoder(inputs).flatten(1).cpu().numpy())
    embeddings = np.concatenate(batches)
    metadata = {
        "architecture": "ResNet18",
        "source": source_name,
        "weights": source,
        "parameters": parameters,
        "embedding_dimension": int(embeddings.shape[1]),
    }
    return _normalize(embeddings), metadata


def _encode_torchvision_model(
    image_paths,
    device,
    batch_size,
    architecture,
):
    import torch
    from PIL import Image
    from torchvision import models

    if architecture == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT
        network = models.efficientnet_b0(weights=weights)
        network.classifier = torch.nn.Identity()
        display_name = "EfficientNet-B0"
    elif architecture == "vit_b_16":
        weights = models.ViT_B_16_Weights.DEFAULT
        network = models.vit_b_16(weights=weights)
        network.heads = torch.nn.Identity()
        display_name = "ViT-B/16"
    else:
        raise ValueError(f"Unsupported torchvision architecture: {architecture}")

    preprocess = weights.transforms()
    network = network.to(device).eval()
    paths = [Path(path) for path in image_paths]
    batches = []
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            tensors = []
            for path in paths[start:start + batch_size]:
                with Image.open(path) as image:
                    tensors.append(preprocess(image.convert("RGB")))
            batches.append(network(torch.stack(tensors).to(device)).cpu().numpy())
    embeddings = np.concatenate(batches)
    metadata = {
        "architecture": display_name,
        "source": str(weights),
        "weights": "ImageNet pretrained",
        "pretrained": True,
        "parameters": sum(parameter.numel() for parameter in network.parameters()),
        "embedding_dimension": int(embeddings.shape[1]),
    }
    del network
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return _normalize(embeddings), metadata


def encode_efficientnet_b0(image_paths, device="cpu", batch_size=16):
    """Extract ImageNet-pretrained EfficientNet-B0 pooled image features."""
    return _encode_torchvision_model(
        image_paths, device, batch_size, "efficientnet_b0"
    )


def encode_vit_b16(image_paths, device="cpu", batch_size=16):
    """Extract ImageNet-pretrained ViT-B/16 class-token image features."""
    return _encode_torchvision_model(image_paths, device, batch_size, "vit_b_16")
