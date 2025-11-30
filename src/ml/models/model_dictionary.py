from src.ml.models.EfficientNet import EfficientNetB0Regressor
from src.ml.models.ColonyCNN import ColonyCNNRegressor
from src.ml.models.ResNet34 import ResNet34Regressor


MODEL_DICTIONARY = {
    "EfficientNet": {
        "class": EfficientNetB0Regressor,
        "kwargs": {"pretrained": False},
        "weights": "efficientnet_b0_colony.pth",
    },
    "ColonyCNN": {
        "class": ColonyCNNRegressor,
        "kwargs": {},
        "weights": "colony_cnn.pth",
    },
    "ResNet34": {
        "class": ResNet34Regressor,
        "kwargs": {
            "pretrained": True,
            "dropout_p": 0.5,
            "freeze_backbone": False
        },
        "weights": "resnet34_colony.pth",
    }
}
