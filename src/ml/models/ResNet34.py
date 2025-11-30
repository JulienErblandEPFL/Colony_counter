import torch
import torch.nn as nn
from torchvision import models


class ResNet34Regressor(nn.Module):
    """
    ResNet-34 backbone + small regression head to predict
    a single colony count per image.

    Output shape: (N, 1)
    """

    def __init__(
        self,
        pretrained: bool = True,
        dropout_p: float = 0.5,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        # ---- Load ResNet-34 backbone (handle older/newer torchvision) ----
        try:
            # Newer torchvision (0.13+)
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.resnet34(weights=weights)
        except AttributeError:
            # Older torchvision
            backbone = models.resnet34(pretrained=pretrained)

        # Take all layers up to (and including) global average pool
        # children() = [conv1, bn1, relu, maxpool, layer1..4, avgpool, fc]
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        feat_dim = backbone.fc.in_features  # 512 for ResNet-34

        # Optionally freeze backbone
        if freeze_backbone:
            for p in self.feature_extractor.parameters():
                p.requires_grad = False

        # ---- Regression head ----
        self.regressor = nn.Sequential(
            nn.Flatten(),              # (N, feat_dim)
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(256, 1),         # scalar colony count
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, 3, H, W)
        returns: (N, 1)
        """
        feats = self.feature_extractor(x)
        out = self.regressor(feats)
        return out
