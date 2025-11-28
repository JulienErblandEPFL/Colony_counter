from src.ml.models.EfficientNet import EfficientNetB0Regressor

MODEL_DICTIONARY = {
    "EfficientNet": {
        "class": EfficientNetB0Regressor,
        "kwargs": {"pretrained": False},
        "weights": "efficientnet_b0_colony.pth",
    }
}
