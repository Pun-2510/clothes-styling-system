"""Gender head operating on the trained ResNet garment embedding."""

from dataclasses import dataclass

import torch
import torch.nn as nn


GENDER_CLASSES = ["Men", "Women"]


class GenderHead(nn.Module):
    def __init__(self, embedding_dim=512, num_classes=2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, num_classes),
        )

    def forward(self, embedding):
        return self.classifier(embedding)


@dataclass(frozen=True)
class GenderResult:
    gender: str
    confidence: float
    probabilities: dict[str, float]
    is_uncertain: bool


@torch.inference_mode()
def classify_gender(head, embedding, classes=GENDER_CLASSES, threshold=0.65):
    tensor = torch.as_tensor(embedding, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    device = next(head.parameters()).device
    probabilities = head(tensor.to(device)).softmax(1)[0].cpu()
    index = int(probabilities.argmax())
    confidence = float(probabilities[index])
    return GenderResult(
        gender=classes[index],
        confidence=confidence,
        probabilities={name: float(value) for name, value in zip(classes, probabilities)},
        is_uncertain=confidence < threshold,
    )
