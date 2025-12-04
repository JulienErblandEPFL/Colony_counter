import matplotlib.pyplot as plt
from PIL import Image


IMG_PATHS = [ #Fill IMG_PATHS at the top with any wells you want to inspect.
    "data/cropped_wells/Brisbane/plate_1/Brisbane_plate1_A3.jpg",
    "data/cropped_wells/IBV/plate_5/IBV_plate5_C1.jpg",
    "data/cropped_wells/Panama/plate_4/Panama_plate4_C1.jpg"
]
IMG_SIZE = 224
NUM_VARIANTS = 3   # how many transformed samples per image

from src.ml.data.transforms import get_counter_train_transforms

def show_transforms():
    transform = get_counter_train_transforms(img_size=IMG_SIZE)

    for img_path in IMG_PATHS:
        img = Image.open(img_path).convert("RGB")

        print(f"\n=== Showing transforms for: {img_path} ===")

        plt.figure(figsize=(12, 2 + 3 * ((NUM_VARIANTS + 1)//3)))

        # 1) First show the original image
        ax = plt.subplot((NUM_VARIANTS+1)//3 + 1, 3, 1)
        ax.imshow(img)
        ax.set_title("Original")
        ax.axis("off")

        # 2) Now produce NUM_VARIANTS random transforms
        for i in range(NUM_VARIANTS):
            timg = transform(img)
            np_img = timg.permute(1, 2, 0).numpy()

            ax = plt.subplot((NUM_VARIANTS+1)//3 + 1, 3, i+2)
            ax.imshow(np_img)
            ax.set_title(f"Aug #{i+1}")
            ax.axis("off")

        plt.suptitle(img_path)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    if len(IMG_PATHS) == 0:
        print("Please fill IMG_PATHS at the top of the file.")
    else:
        show_transforms()
