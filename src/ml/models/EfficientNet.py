import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class EfficientNetB0Regressor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        if pretrained:
            self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            self.model = efficientnet_b0(weights=None)

        # Replace classification head with a regression layer
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, 1)

    def forward(self, x):
        return torch.relu(self.model(x))
        # return self.model(x)

