import matplotlib.pyplot as plt
from src.ml.data.dataset import ColonyDataset
from src.ml.data.transforms import get_train_transforms

ds = ColonyDataset("data/dataset.csv",
                   transform=get_train_transforms())

img, label = ds[2]

# Convert tensor to a displayable image
img_np = img.permute(1, 2, 0).numpy()  # (H, W, C)

plt.imshow(img_np)
plt.title(f"Label = {label}")
plt.show()
