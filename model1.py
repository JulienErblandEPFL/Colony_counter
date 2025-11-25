import os
import random

import numpy as np
import pandas as pd
from PIL import Image

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


# ---------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------
LABELS_XLSX = "labels.xlsx"         # path to your Excel file
IMAGES_DIR = "dataset_wells"        # folder containing all images
TRAIN_RATIO = 0.7
BATCH_SIZE = 64
NUM_EPOCHS = 30               # change as needed
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.0005
MOMENTUM = 0.9
IMG_SIZE = 128                # network in the paper uses 128x128

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


# ---------------------------------------------------------------------
# 2. Dataset / Dataloader
# ---------------------------------------------------------------------
class ColoniesDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        """
        df must have columns: 'filename', 'count'
        """
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row["filename"]
        label = float(row["count"])

        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # regression target is a single float
        label = torch.tensor(label, dtype=torch.float32)

        return image, label


# transforms: resize, convert to tensor, normalize
train_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    # optional augmentations:
    # T.RandomHorizontalFlip(),
    # T.RandomRotation(10),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

test_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


# ---------------------------------------------------------------------
# 3. Model1
#    4 conv layers + 1 fully connected + regression output
# ---------------------------------------------------------------------
class ColonyCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # 1st conv layer: 3 -> 20, 5x5, LRN + maxpool
        self.conv1 = nn.Conv2d(3, 20, kernel_size=5, padding=2)
        self.lrn1 = nn.LocalResponseNorm(5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 2nd conv layer: 20 -> 50, 5x5, LRN + maxpool
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5, padding=2)
        self.lrn2 = nn.LocalResponseNorm(5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 3rd conv layer: 50 -> 100, 4x4
        self.conv3 = nn.Conv2d(50, 100, kernel_size=4, padding=1)

        # 4th conv layer: 100 -> 200, 4x4 + maxpool
        self.conv4 = nn.Conv2d(100, 200, kernel_size=4, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # compute flattened size for 128x128 input
        with torch.no_grad():
            dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
            x = self._forward_features(dummy)
            flat_dim = x.view(1, -1).shape[1]

        # fully connected layers: 500 units then 1 output (regression)
        self.fc1 = nn.Linear(flat_dim, 500)
        self.dropout = nn.Dropout(p=0.5)
        self.fc_out = nn.Linear(500, 1)

    def _forward_features(self, x):
        x = torch.relu(self.conv1(x))
        x = self.lrn1(x)
        x = self.pool1(x)

        x = torch.relu(self.conv2(x))
        x = self.lrn2(x)
        x = self.pool2(x)

        x = torch.relu(self.conv3(x))

        x = torch.relu(self.conv4(x))
        x = self.pool4(x)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc_out(x)           # shape: (batch, 1)
        x = x.squeeze(1)             # shape: (batch,)
        return x


# ---------------------------------------------------------------------
# 4. Training / evaluation functions
# ---------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    mae = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            mae += torch.abs(outputs - labels).sum().item()

    mse = running_loss / len(loader.dataset)
    mae = mae / len(loader.dataset)
    return mse, mae


# ---------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------
def main():
    # 5.1 load Excel and split 70/30
    df = pd.read_excel(LABELS_XLSX)

    # assume first column = filename, second = count
    df.columns = ["filename", "count"]

    train_df, test_df = train_test_split(
        df, train_size=TRAIN_RATIO, random_state=RANDOM_SEED, shuffle=True
    )

    print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")

    # 5.2 datasets and loaders
    train_dataset = ColoniesDataset(train_df, IMAGES_DIR, transform=train_transform)
    test_dataset = ColoniesDataset(test_df, IMAGES_DIR, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 5.3 model, loss, optimizer
    model = ColonyCNN().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )

    # 5.4 training loop
    best_test_mse = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        test_mse, test_mae = evaluate(model, test_loader, criterion)

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} "
            f"- train MSE: {train_loss:.4f} "
            f"- test MSE: {test_mse:.4f}, test MAE: {test_mae:.4f}"
        )

        if test_mse < best_test_mse:
            best_test_mse = test_mse
            torch.save(model.state_dict(), "best_colony_cnn.pth")

    print("Training done. Best test MSE:", best_test_mse)


if __name__ == "__main__":
    main()
