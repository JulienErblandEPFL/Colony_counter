import sys
import os

# Find project root (the folder that contains "src/")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.insert(0, ROOT)


import torch
import pandas as pd
from torch.utils.data import DataLoader

from src.ml.data.dataset import ColonyDataset
from src.ml.data.transforms import get_train_transforms, get_test_transforms
from src.ml.models.EfficientNet import EfficientNetB0Regressor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def evaluate_model(csv_path, model_path, batch_size=16):
    # Load data
    df = pd.read_csv(csv_path).dropna(subset=["value"])
    df = df[df["value"] >= 0].reset_index(drop=True)
    test_ds = ColonyDataset(df=df, transform=get_test_transforms())
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Load model
    model = EfficientNetB0Regressor(pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    all_preds = []
    all_labels = []

    # Inference
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            preds = model(imgs).squeeze(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Convert to series
    preds = torch.tensor(all_preds, dtype=torch.float32)
    labels = torch.tensor(all_labels, dtype=torch.float32)

    # Metrics
    mae = torch.mean(torch.abs(preds - labels)).item()
    mse = torch.mean((preds - labels)**2).item()
    rmse = mse**0.5
    r2 = 1 - mse / torch.var(labels).item()

    print("\n--- Evaluation ---")
    print("MAE :", mae)
    print("MSE :", mse)
    print("RMSE:", rmse)
    print("R^2  :", r2)

    return preds, labels

if __name__ == "__main__":
    import random
    df = pd.read_csv("data/dataset.csv")
    df = df[df["value"] >= 0].reset_index(drop=True)

    preds, labels = evaluate_model("data/dataset.csv", "efficientnet_b0_colony.pth")

    print("\n--- Random Sample Predictions (Valid Wells Only) ---")
    indices = random.sample(range(len(df)), 10)

    for idx in indices:
        row = df.iloc[idx]
        filename = os.path.basename(row["path"])
        plate_name = next((p for p in filename.split("_") if "plate" in p.lower()), "unknown")
        
        print(
            f"Plate: {plate_name:10s} | "
            f"File: {filename:30s} | "
            f"Pred: {preds[idx].item():6.2f} | "
            f"True: {labels[idx].item():6.2f}"
        )
