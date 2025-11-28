import sys
import os

# Find project root (the folder that contains "src/")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, ROOT)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd

from src.ml.data.dataset import ColonyDataset
from src.ml.data.transforms import get_train_transforms, get_test_transforms


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_model(
    csv_path: str,
    model_class,             # Example: EfficientNetB0Regressor
    model_kwargs=None,       # Extra args for model
    optimizer_class=optim.Adam,
    optimizer_kwargs=None,
    criterion=nn.MSELoss(),
    batch_size: int = 16,
    epochs: int = 10,
    lr: float = 1e-4,
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
        criterion: Loss function (default MSE).
        batch_size: Batch size.
        epochs: Number of epochs.
        lr: Learning rate (also used if optimizer_kwargs is empty)
        save_path: Path where model is saved.

    Returns:
        model, (train_losses, val_losses)
    """

    model_kwargs = model_kwargs or {}
    optimizer_kwargs = optimizer_kwargs or {"lr": lr}

    # --- Load and clean data ---
    df = pd.read_csv(csv_path).dropna(subset=["value"])
    df = df[df["value"] >= 0]  # Keep only valid labels

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

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

    # --- Training Loop ---
    for epoch in range(epochs):
        # Train
        model.train()
        total_train_loss = 0

        for imgs, labels in train_loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE).unsqueeze(1)

            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(DEVICE)
                labels = labels.to(DEVICE).unsqueeze(1)
                preds = model(imgs)
                total_val_loss += criterion(preds, labels).item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train MSE={avg_train_loss:.4f} | "
            f"Test MSE={avg_val_loss:.4f}"
        )

    # Save model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    return model, (train_losses, val_losses)


if __name__ == "__main__":
    from src.ml.models.EfficientNet import EfficientNetB0Regressor

    train_model(
        csv_path="data/dataset.csv",
        model_class=EfficientNetB0Regressor,
        model_kwargs={"pretrained": True},
        batch_size=16,
        epochs=10,
        lr=1e-4,
        save_path="efficientnet_b0_colony.pth"
    )
