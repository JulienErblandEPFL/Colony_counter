import sys
import os

# Find project root (the folder that contains "src/")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, ROOT)


import torch
import pandas as pd
from torch.utils.data import DataLoader

from src.ml.data.dataset import ColonyDataset
from src.ml.data.transforms import get_train_transforms, get_test_transforms
from src.ml.models.EfficientNet import EfficientNetB0Regressor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def evaluate_model(
    csv_path: str,
    model_path: str,
    model_class,             # <-- pass your model class here
    model_kwargs=None,       # <-- extra args for the model
    batch_size: int = 16,
    transforms=None
):
    """
    Generic evaluation function usable for ANY regression model.

    Args:
        csv_path: Path to dataset CSV.
        model_path: Path to the saved .pth model.
        model_class: Model class to instantiate.
        model_kwargs: Optional dict of model arguments.
        batch_size: Batch size for DataLoader.
        transforms: Transform function; defaults to test transforms.

    Returns:
        (preds, labels) tensors
    """

    model_kwargs = model_kwargs or {}
    transforms = transforms or get_test_transforms()

    # --- Load dataset ---
    df = pd.read_csv(csv_path).dropna(subset=["value"])
    df = df[df["value"] >= 0].reset_index(drop=True)

    test_ds = ColonyDataset(df=df, transform=transforms)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # --- Load model ---
    model = model_class(**model_kwargs)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE).eval()

    all_preds = []
    all_labels = []

    # --- Inference ---
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            preds = model(imgs).squeeze(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # --- Convert to tensors ---
    preds = torch.tensor(all_preds, dtype=torch.float32)
    labels = torch.tensor(all_labels, dtype=torch.float32)

    # --- Metrics ---
    mae = torch.mean(torch.abs(preds - labels)).item()
    mse = torch.mean((preds - labels) ** 2).item()
    rmse = mse ** 0.5
    r2 = 1 - mse / torch.var(labels).item()

    # Accuracy within 10% relative error
    tolerance = 0.10
    accuracy_mask = torch.abs(preds - labels) <= tolerance * torch.clamp(labels, min=1e-8)
    accuracy = accuracy_mask.float().mean().item()

    print("\n--- Evaluation ---")
    print(f"MAE  : {mae:.4f}")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R^2  : {r2:.4f}")
    print(f"Accuracy (+/- 10%): {accuracy*100:.2f} %")

    return preds, labels


if __name__ == "__main__":
    import random
    df = pd.read_csv("data/dataset.csv")
    df = df[df["value"] >= 0].reset_index(drop=True)

    preds, labels = evaluate_model("data/dataset.csv", "efficientnet_b0_colony.pth", EfficientNetB0Regressor)

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
