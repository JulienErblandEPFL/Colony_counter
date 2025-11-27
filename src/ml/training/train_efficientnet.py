import sys
import os

# Find project root (the folder that contains "src/")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.insert(0, ROOT)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd

from src.ml.data.dataset import ColonyDataset
from src.ml.data.transforms import get_train_transforms, get_test_transforms
from src.ml.models.EfficientNet import EfficientNetB0Regressor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_efficientnet(csv_path, batch_size=16, epochs=10, lr=1e-4):
    # Load data
    df = pd.read_csv(csv_path).dropna(subset=["value"])
    
    #to not train on all the -1 values (WE SHOULD DO A CLASSIFIER LATER WITH THOSE)
    df = df[df["value"] >= 0]

    #to make the model understand that -1 means "A LOT"
    #df.loc[df["value"] == -1, "value"] = 200

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # Datasets
    train_ds = ColonyDataset(df=train_df, transform=get_train_transforms())
    val_ds   = ColonyDataset(df=val_df,   transform=get_test_transforms())

    # Dataloaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    # Model
    model = EfficientNetB0Regressor(pretrained=True).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for imgs, labels in train_loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE).unsqueeze(1)

            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(DEVICE)
                labels = labels.to(DEVICE).unsqueeze(1)
                preds = model(imgs)
                val_loss += criterion(preds, labels).item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)     
        print(f"Epoch {epoch+1}/{epochs} | Train set MSE={avg_train_loss:.4f} | Validation set MSE={avg_val_loss:.4f}")

    torch.save(model.state_dict(), "efficientnet_b0_colony.pth")
    print("Model saved to efficientnet_b0_colony.pth")

if __name__ == "__main__":
    train_efficientnet("data/dataset.csv", batch_size=16, epochs=20, lr=1e-4)