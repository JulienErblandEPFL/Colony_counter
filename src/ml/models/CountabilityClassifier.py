import torch
import torch.nn as nn
from .EfficientNet import EfficientNetB0  # or ResNet34

class CountabilityClassifier(nn.Module):
    def __init__(self, backbone="efficientnet_b0", pretrained=True):
        super().__init__()

        if backbone == "efficientnet_b0":
            self.backbone = EfficientNetB0(pretrained=pretrained)
            in_features = self.backbone.classifier.in_features

            # Replace final classifier layer
            self.backbone.classifier = nn.Identity()

        elif backbone == "resnet34":
            from .ResNet34 import get_resnet34_model
            self.backbone = get_resnet34_model(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        # NEW: 2-class classifier head
        self.head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)   # 2 classes
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)
