from src.ml.data.dataset import ColonyDataset
from src.ml.data.transforms import get_train_transforms

ds = ColonyDataset("data/dataset.csv", transform=get_train_transforms())
img, label = ds[0]

print(img.shape, label)
