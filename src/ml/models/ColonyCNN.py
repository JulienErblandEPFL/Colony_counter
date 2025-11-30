import torch
import torch.nn as nn


class ColonyCNNRegressor(nn.Module):
    """
    Colony-counting CNN: 4 convolutional blocks + global average pooling +
    a small fully-connected head with 1 regression output.

    Designed to be compatible with:
      - src/ml/training.py   (expects model(imgs) -> (B, 1))
      - src/ml/evaluate.py   (does .squeeze(1) on outputs)

    Args:
        img_size (int): Kept for backward compatibility with config/kwargs,
                        but not strictly needed thanks to AdaptiveAvgPool.
        dropout_p (float): Dropout probability in the fully connected layer.
    """

    def __init__(self, dropout_p: float = 0.5):
        super().__init__()

        # ------------------------------
        # Convolutional feature extractor
        # ------------------------------
        # Block 1: 3 -> 20 channels
        self.conv1 = nn.Conv2d(3, 20, kernel_size=5, padding=2)
        self.bn1   = nn.BatchNorm2d(20)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # /2

        # Block 2: 20 -> 50 channels
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5, padding=2)
        self.bn2   = nn.BatchNorm2d(50)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # /2

        # Block 3: 50 -> 100 channels
        self.conv3 = nn.Conv2d(50, 100, kernel_size=4, padding=1)
        self.bn3   = nn.BatchNorm2d(100)

        # Block 4: 100 -> 200 channels
        self.conv4 = nn.Conv2d(100, 200, kernel_size=4, padding=1)
        self.bn4   = nn.BatchNorm2d(200)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # /2

        # Adaptive global average pooling: any H×W -> 1×1
        # This makes the model independent of the exact input size,
        # as long as it's large enough spatially.
        self.gap = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # ------------------------------
        # Fully-connected regression head
        # ------------------------------
        # After GAP, feature shape is (B, 200, 1, 1) -> (B, 200)
        fc_in_dim = 200

        self.fc1 = nn.Linear(fc_in_dim, 128)
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc_out = nn.Linear(128, 1)   # final scalar output per image

        # Optional: weight initialization (simple variant)
        self._init_weights()

    def _init_weights(self):
        """
        Simple Kaiming initialization for conv + linear layers.
        Not strictly required, but often helps convergence.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the convolutional feature extractor
        (before global pooling and FC head).
        """
        # Block 1
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        # Block 2
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        # Block 3
        x = torch.relu(self.bn3(self.conv3(x)))

        # Block 4
        x = torch.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)

        # Global Average Pooling -> (B, 200, 1, 1)
        x = self.gap(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass: returns predictions of shape (B, 1).

        We DO NOT squeeze here because the training/eval scripts
        expect (batch_size, 1) and handle any squeezing themselves.
        """
        x = self._forward_features(x)      # (B, 200, 1, 1)
        x = torch.flatten(x, 1)           # (B, 200)
        x = torch.relu(self.fc1(x))       # (B, 128)
        x = self.dropout(x)
        x = self.fc_out(x)                # (B, 1)
        return x
