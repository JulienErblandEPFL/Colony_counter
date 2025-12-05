import sys
import os
import time

# Find project root (the folder that contains "src/")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, ROOT)

import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd

from src.ml.data.dataset import ColonyDataset
from src.ml.data.transforms import get_classifier_train_transforms, get_classifier_test_transforms
from src.ml.models.CountabilityClassifier import CountabilityClassifier

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate_classifier(model, loader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            logits = model(imgs)
            preds = logits.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total if total > 0 else 0


def train_classifier(
    csv_path,
    backbone="efficientnet_b0",
    epochs=10,
    batch=32,
    lr=1e-4,
    save_as="countability_classifier.pth"
):
    print(f"[INFO] Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path)

    df_train, df_val = train_test_split(df, test_size=0.2, random_state=42)

    train_ds = ColonyDataset(df=df_train, transform=get_classifier_train_transforms(), task="classify")
    val_ds   = ColonyDataset(df=df_val,   transform=get_classifier_test_transforms(),   task="classify")

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch)

    model = CountabilityClassifier(backbone=backbone).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    print(f"[INFO] Starting training for {epochs} epochs")

    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            logits = model(imgs)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        val_acc = evaluate_classifier(model, val_loader)

        print(f"Epoch [{ep}/{epochs}]  Loss={train_loss:.3f}  ValAcc={val_acc:.3f}")

    torch.save(model.state_dict(), save_as)
    print(f"[INFO] Model saved to {save_as}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Countability Classifier")
    parser.add_argument("--csv", default="data/training.csv", help="Path to dataset CSV")
    parser.add_argument("--backbone", default="efficientnet_b0", choices=["efficientnet_b0", "resnet34"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_as", default="countability_classifier.pth")

    args = parser.parse_args()

    train_classifier(
        csv_path=args.csv,
        backbone=args.backbone,
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        save_as=args.save_as,
    )



