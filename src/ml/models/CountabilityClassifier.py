import torch
import torch.nn as nn
from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights,
    resnet34,
    ResNet34_Weights,
)

class CountabilityClassifier(nn.Module):
    """
    Binary classifier to decide if a well image is 'countable'
    or 'uncountable'.

    backbone: "efficientnet_b0" or "resnet34"
    pretrained: loads ImageNet weights if True
    """

    def __init__(self, backbone="efficientnet_b0", pretrained=True):
        super().__init__()

        self.backbone_name = backbone

        # --------------- EfficientNet-B0 -----------------
        if backbone == "efficientnet_b0":
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None

            self.backbone = efficientnet_b0(weights=weights)

            # EfficientNet classifier: Dropout + Linear
            # Linear layer is always index 1
            in_features = self.backbone.classifier[1].in_features

            # Remove original classifier head
            self.backbone.classifier = nn.Identity()

        # ---------------- ResNet-34 -----------------------
        elif backbone == "resnet34":
            weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None

            self.backbone = resnet34(weights=weights)

            # ResNet classifier head is called `.fc`
            in_features = self.backbone.fc.in_features

            self.backbone.fc = nn.Identity()

        else:
            raise ValueError(
                f"Unsupported backbone '{backbone}'. "
                "Available: 'efficientnet_b0', 'resnet34'"
            )

        # -------- New binary classification head ----------
        self.head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 2),  # 2 classes
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)
