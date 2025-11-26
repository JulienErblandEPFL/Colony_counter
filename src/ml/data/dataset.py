import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torch

class ColonyDataset(Dataset):
    """
    Dataset for colony counting.
    Accepts either:
        - csv_path: path to CSV file
        - df: a pandas DataFrame
    """
    def __init__(self, csv_path=None, df=None, transform=None):
        # Load from DataFrame
        if df is not None:
            self.df = df.copy()
        # Load from CSV path
        elif csv_path is not None:
            self.df = pd.read_csv(csv_path)
        else:
            raise ValueError("You must provide either csv_path or df.")

        # Remove unreadable wells (-1)
        self.df = self.df[self.df["value"] >= 0]

        # Remove missing labels
        self.df = self.df.dropna(subset=["value"])

        # Reset index for safety
        self.df = self.df.reset_index(drop=True)

        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(float(row["value"]), dtype=torch.float32)
        return img, label
