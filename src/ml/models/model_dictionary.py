from src.ml.models.EfficientNet import EfficientNetB0Regressor
from src.ml.models.ColonyCNN import ColonyCNNRegressor


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
    }
}
