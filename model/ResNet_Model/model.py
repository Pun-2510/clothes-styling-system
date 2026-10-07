import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet18_Weights

class FashionResNet(nn.Module):

    def __init__(
        self,
        num_classes,
        pretrained=True
    ):

        super().__init__()

        if pretrained:
            self.model = models.resnet18(
                weights=ResNet18_Weights.DEFAULT
            )
        else:
            self.model = models.resnet18(
                weights=None
            )

        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            num_classes
        )


    def encode_image(self, x, normalize=True):
        """Encode images in a CLIP-like normalized retrieval space."""
        backbone = self.model
        features = backbone.conv1(x)
        features = backbone.bn1(features)
        features = backbone.relu(features)
        features = backbone.maxpool(features)
        features = backbone.layer1(features)
        features = backbone.layer2(features)
        features = backbone.layer3(features)
        features = backbone.layer4(features)
        features = backbone.avgpool(features).flatten(1)
        return F.normalize(features, dim=1) if normalize else features


    def forward(self, x, return_embedding=False):

        embedding = self.encode_image(x, normalize=False)
        logits = self.model.fc(embedding)
        if return_embedding:
            return logits, F.normalize(embedding, dim=1)
        return logits
