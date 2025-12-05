import sys
import os

# Find project root (the folder that contains "src/")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, ROOT)


import torch
import pandas as pd
from torch.utils.data import DataLoader

from src.ml.data.dataset import ColonyDataset
from src.ml.data.transforms import get_counter_test_transforms
from src.ml.models.EfficientNet import EfficientNetB0Regressor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
CSV_PATH = "data/test.csv"

def evaluate_model(
    model_path: str,
    model_class,             # <-- pass your model class here
    csv_path: str = CSV_PATH,
    model_kwargs=None,       # <-- extra args for the model
    batch_size: int = BATCH_SIZE,
    transforms=None,
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
        (preds, labels) tensors -- preds are rounded to nearest integer
    """

    model_kwargs = model_kwargs or {}
    transforms = transforms or get_counter_test_transforms()

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

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    # --- Convert to tensors ---
    preds = torch.cat(all_preds, dim=0).float()
    labels = torch.cat(all_labels, dim=0).float()

    # Keep raw predictions for metric computation
    preds_raw = preds.clone()

    # --- Metrics ---
    abs_error = torch.abs(preds_raw - labels)

    mae = abs_error.mean().item()
    mse = ((preds_raw - labels) ** 2).mean().item()
    rmse = mse ** 0.5

    y_mean = labels.mean()
    ss_res = ((preds_raw - labels) ** 2).sum().item()
    ss_tot = ((labels - y_mean) ** 2).sum().item()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


    # Accuracy
    # Hybrid tolerance: 10% of true value + 1. For labels == 0, allow absolute error <= 1.
    tolerance_rel = 0.10
    per_sample_tol = tolerance_rel * labels + 1

    zero_mask = labels == 0
    non_zero_mask = ~zero_mask

    accuracy_mask = torch.zeros_like(abs_error, dtype=torch.bool)
    accuracy_mask[zero_mask] = abs_error[zero_mask] <= 1
    accuracy_mask[non_zero_mask] = abs_error[non_zero_mask] <= per_sample_tol[non_zero_mask]

    accuracy = accuracy_mask.float().mean().item()

    print("\n--- Evaluation ---")
    print(f"MAE  : {mae:.4f}")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R^2  : {r2:.4f}")
    print(f"Accuracy (+/- 10%+1): {accuracy*100:.2f} %")

    # Toujours arrondir les prédictions pour affichage / retour
    preds = preds_raw.round().clamp(min=0)

    return preds, labels


import argparse
import random
from src.ml.models.model_dictionary import MODEL_DICTIONARY

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate a colony counting model")

    parser.add_argument("--model", required=True,
                        help="Model key from MODEL_DICTIONARY, e.g. 'EfficientNet'")

    parser.add_argument("--csv", default=CSV_PATH,
                        help="Path to dataset CSV(default: data/test_split_from_training.csv)")

    parser.add_argument("--weights", default=None,
                        help="Optional override for weight file")

    parser.add_argument("--samples", type=int, default=10,
                        help="Number of random samples to show(default: 10)")

    args = parser.parse_args()

    #Load and filter dataset
    df = pd.read_csv(args.csv)
    df = df[df["value"] >= 0].reset_index(drop=True)

    #Load model arguments from dictionary
    if args.model not in MODEL_DICTIONARY:
        raise ValueError(
            f"Unknown model '{args.model}'. Available: {list(MODEL_DICTIONARY.keys())}"
        )

    entry = MODEL_DICTIONARY[args.model]

    model_class  = entry["class"]
    model_kwargs = entry.get("kwargs", {})
    model_path   = args.weights or entry["weights"]

    print(f"\nUsing model: {args.model}")
    print(f"Model class: {model_class.__name__}")
    print(f"Loading weights: {model_path}")

    #Run evaluation
    preds, labels = evaluate_model(
        csv_path=args.csv,
        model_path=model_path,
        model_class=model_class,
        model_kwargs=model_kwargs,
    )

    #Print random samples
    print("\n--- Random Sample Predictions (Valid Wells Only) ---")

    n_samples = min(args.samples, len(df))
    indices = random.sample(range(len(df)), n_samples)

    for idx in indices:
        row = df.iloc[idx]
        filename = os.path.basename(row["path"])
        plate_name = next(
            (p for p in filename.split("_") if "plate" in p.lower()), "unknown"
        )

        print(
            f"Plate: {plate_name:10s} | "
            f"File:  {filename:30s} | "
            f"Pred: {int(preds[idx].item()):6d} | "
            f"True: {labels[idx].item():6.2f}"
        )
