import torch.nn as nn
from torchvision import models

class FashionResNet(nn.Module):

    def __init__(
        self,
        num_classes,
        pretrained=True
    ):

        super().__init__()

        if pretrained:
            self.model = models.resnet18(
                weights="DEFAULT"
            )
        else:
            self.model = models.resnet18(
                weights=None
            )

        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            num_classes
        )


    def forward(self, x):

        return self.model(x)
