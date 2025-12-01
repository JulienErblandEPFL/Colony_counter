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
    def __init__(self, csv_path=None, df=None, transform=None, task="count"):
        # Load the dataframe
        if df is not None:
            self.df = df.copy()
        elif csv_path is not None:
            self.df = pd.read_csv(csv_path)
        else:
            raise ValueError("Provide either 'csv_path' or 'df'.")

        self.task = task
        self.transform = transform

        # Validate and clean dataset based on task
        if self.task == "count":
            if "value" not in self.df.columns:
                raise ValueError(
                    "Counting task requires 'value' column in the dataset."
                )
            # Remove unreadable or ambiguous wells
            self.df = self.df[self.df["value"] >= 0]
            self.df = self.df.dropna(subset=["value"])

        elif self.task == "classify":
            if "is_countable" not in self.df.columns:
                raise ValueError(
                    "Classification task requires 'is_countable' column in the dataset."
                )
        else:
            raise ValueError("task must be 'count' or 'classify'")

        # Reset index for clean indexing
        self.df = self.df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Open image
        img = Image.open(row["path"]).convert("RGB")

        # Apply transforms
        if self.transform:
            img = self.transform(img)

        # Output format depends on the task
        if self.task == "classify":
            label = int(row["is_countable"])
            label = torch.tensor(label, dtype=torch.long)    # CE loss expects long
            return img, label

        # Counting mode → return colony count
        value = float(row["value"])
        label = torch.tensor(value, dtype=torch.float32)
        return img, label