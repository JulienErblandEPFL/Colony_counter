import sys
import os
import time

# Find project root (the folder that contains "src/")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, ROOT)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import argparse

from src.ml.data.dataset import ColonyDataset
from src.ml.data.transforms import get_train_transforms, get_test_transforms
from src.ml.models.model_dictionary import MODEL_DICTIONARY


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CRITERION = nn.L1Loss()
IMG_SIZE = 224
BATCH_SIZE = 16


def train_model(
    csv_path: str,
    model_class,             # Example: EfficientNetB0Regressor
    model_kwargs=None,       # Extra args for model
    optimizer_class=optim.Adam,
    optimizer_kwargs=None,
    criterion=CRITERION,
    batch_size: int = BATCH_SIZE,
    epochs: int = 30,
    lr: float = 1e-4,
    img_size: int = IMG_SIZE,
    save_path: str = "model.pth",
):
    """
    General training loop for ANY PyTorch regression model.

    Args:
        csv_path: Path to dataset CSV.
        model_class: Model class to instantiate.
        model_kwargs: Arguments passed to the model constructor.
        optimizer_class: Optimizer (Adam, SGD...)
        optimizer_kwargs: Extra optimizer args.
        criterion: Loss function.
        batch_size: Batch size.
        epochs: Number of epochs.
        lr: Learning rate (also used if optimizer_kwargs is empty)
        img_siz: Image size needed from the model.
        save_path: Path where model is saved.

    Returns:
        model, (train_losses, val_losses)
    """

    model_kwargs = model_kwargs or {}
    optimizer_kwargs = optimizer_kwargs or {"lr": lr}

    # --- Load and clean data ---
    df = pd.read_csv(csv_path).dropna(subset=["value"])
    df = df[df["value"] >= 0]  # Keep only valid labels

    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df   = train_test_split(temp_df, test_size=0.5, random_state=42)
    test_df.to_csv("data/test_split_from_training.csv", index=False)

    # --- Dataset & loaders ---
    train_ds = ColonyDataset(df=train_df, transform=get_train_transforms())
    val_ds   = ColonyDataset(df=val_df,  transform=get_test_transforms())


    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # --- Model ---
    model = model_class(**model_kwargs).to(DEVICE)

    optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)

    train_losses = []
    val_losses = []

    start_time = time.perf_counter()
    print(f"{'Epoch':^12} | {'Train Loss':^12} | {'Val Loss':^12} | {'Epoch time':^12}")
    print("-" * 60)

    # --- Training Loop ---
    for epoch in range(epochs):
        epoch_start = time.perf_counter()

        # Train
        model.train()
        total_train_loss = 0.0

        for imgs, labels in train_loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE).unsqueeze(1)

            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * imgs.size(0) # last batch can have different size

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(DEVICE)
                labels = labels.to(DEVICE).unsqueeze(1)
                preds = model(imgs)
                total_val_loss += criterion(preds, labels).item() * imgs.size(0)

        avg_train_loss = total_train_loss / len(train_ds)
        avg_val_loss = total_val_loss / len(val_ds)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        epoch_time = time.perf_counter() - epoch_start
        total_time = time.perf_counter() - start_time

        epoch_str = f"{epoch+1}/{epochs}"
        time_str = f"{epoch_time:.2f}s"
        print(
            f"{epoch_str:^12} | "
            f"{avg_train_loss:^12.4f} | "
            f"{avg_val_loss:^12.4f} | "
            f"{time_str:^12}  "
        )
        print("-" * 60)


    print("-" * 60)
    print(f"Training finished in {total_time:0.1f} seconds.")

    # Save model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    return model, (train_losses, val_losses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="Model name from MODEL_DICTIONARY")
    parser.add_argument("--csv", type=str, default="data/dataset.csv")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    # Lookup model
    if args.model not in MODEL_DICTIONARY:
        raise ValueError(f"Unknown model '{args.model}'. Available: {list(MODEL_DICTIONARY.keys())}")

    entry = MODEL_DICTIONARY[args.model]

    model_class  = entry["class"]
    model_kwargs = entry.get("kwargs", {})
    save_path    = entry["weights"]          # automatic save file name

    print(f"\nUsing model: {args.model}")
    # Run training
    train_model(
        csv_path=args.csv,
        model_class=model_class,
        model_kwargs=model_kwargs,
        epochs=args.epochs,
        save_path=save_path,
    )
