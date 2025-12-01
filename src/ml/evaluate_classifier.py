import sys
import os

# Find project root (the folder that contains "src/")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, ROOT)

import argparse
import torch
import pandas as pd
from torch.utils.data import DataLoader

from ml.data.dataset import ColonyDataset
from ml.data.transforms import get_val_transforms
from ml.models.CountabilityClassifier import CountabilityClassifier


@torch.no_grad()
def evaluate_classifier(model, loader):
    model.eval()
    correct, total = 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.cuda(), labels.cuda()

        logits = model(imgs)
        preds = logits.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total if total > 0 else 0


def run_eval(csv_path, weights_path, backbone):
    print(f"[INFO] Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path)

    ds = ColonyDataset(df, transform=get_val_transforms(), task="classify")
    loader = DataLoader(ds, batch_size=32)

    model = CountabilityClassifier(backbone=backbone)
    model.load_state_dict(torch.load(weights_path))
    model.cuda()

    acc = evaluate_classifier(model, loader)
    print(f"[RESULT] Accuracy on dataset: {acc:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Countability Classifier")
    parser.add_argument("--csv", default="data/dataset.csv", help="Dataset CSV containing paths and labels")
    parser.add_argument("--weights", default="countability_classifier.pth", help="Model weights .pth file")
    parser.add_argument("--backbone", default="efficientnet_b0", choices=["efficientnet_b0", "resnet34"])

    args = parser.parse_args()

    run_eval(
        csv_path=args.csv,
        weights_path=args.weights,
        backbone=args.backbone,
    )
