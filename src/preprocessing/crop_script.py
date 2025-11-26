import os
from tqdm import tqdm
from crop_wells import crop_wells   # your function

root_dir = "cropped_data"   # only process images here

# Allowed image extensions
exts = {".jpg", ".jpeg", ".png"}

image_paths = []
for base, dirs, files in os.walk(root_dir):
    for f in files:
        ext = os.path.splitext(f)[1].lower()
        if ext in exts:
            full_path = os.path.join(base, f)
            image_paths.append(full_path)

print(f"Found {len(image_paths)} images to process.")

for img_path in tqdm(image_paths, desc="Cropping wells"):
    try:
        # Extract plate type from the parent directory name
        plate_type = os.path.basename(os.path.dirname(img_path))

        # Run your cropper
        crop_wells(img_path, plate_type)

    except Exception as e:
        print(f"\nError processing {img_path}: {e}")
