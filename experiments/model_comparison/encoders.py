"""ResNet18 and SBERT feature encoders for the controlled benchmark."""

from __future__ import annotations

from pathlib import Path

import numpy as np


SBERT_MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


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
