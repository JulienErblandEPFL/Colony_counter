import cv2
import numpy as np
import os
import sys
from tqdm import tqdm
from pathlib import Path

def crop_wells(file_path, plate_type, debug=False, save_files=True):
    # Extract plate name and number from file path
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    # Extract plate number ONLY from the "Plate_X" part
    name_lower = base_name.lower()
    if "plate_" in name_lower:
        plate_number = name_lower.split("plate_")[1]
    else:
        raise ValueError(f"Filename must contain 'Plate_X': {base_name}")

    plate_name = f"plate_{plate_number}"

    # Debug file path
    debug_path = f"{plate_name}_detected_debug.jpg"

    # New directory structure:
    # cropped_wells/{plate_type}/plate_x/

    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    sys.path.insert(0, ROOT)

    base_dir = Path(os.path.join(ROOT, "data", "cropped_wells"))
    type_dir = os.path.join(base_dir, plate_type)
    output_dir = os.path.join(type_dir, plate_name)

    # Ensure directories exist
    if save_files:
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(type_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

    # Load and preprocess the image
    img = cv2.imread(file_path)
    if img is None:
        raise ValueError(f"Image not found: {file_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)
    gray = cv2.equalizeHist(gray)

    
    # DYNAMIC ESTIMATION
    height, width = gray.shape[:2]  #load image
    estimated_radius = int(height / 8)  # For a 12-well plate (3 rows), the well radius is roughly height / 8
    
    params = dict(
        dp=1.2,
        minDist=int(estimated_radius),
        param1=100,
        param2=60,
        minRadius=int(estimated_radius * 0.8), # Search between 80%...
        maxRadius=int(estimated_radius * 1.2)  # ...and 120% of the estimate
    )


    # Try to find exactly 12 wells by tuning param2
    best_circles = None
    for p in range(60, 90, 5):
        params["param2"] = p
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, **params)
        if circles is not None and len(circles[0]) == 12:
            best_circles = np.around(circles[0, :]).astype(int)
            print(f"Found exactly 12 circles with param2={p}")
            break

    # If not found, use closest result
    if best_circles is None:
        raise RuntimeError("Did not find exactly 12 circles in the image.")

    # Group circles into rows using Y coordinate
    best_circles = sorted(best_circles, key=lambda c: c[1])
    ys = [c[1] for c in best_circles]
    row_threshold = np.std(ys) * 0.5

    rows = []
    current_row = [best_circles[0]]
    cropped_images = {}
    for c in best_circles[1:]:
        if abs(c[1] - np.mean([cc[1] for cc in current_row])) < row_threshold:
            current_row.append(c)
        else:
            rows.append(current_row)
            current_row = [c]
    rows.append(current_row)

    # Sort left→right then reverse (A1 on the right)
    rows = [list(reversed(sorted(row, key=lambda c: c[0]))) for row in rows]

    # Label wells A1–C4
    row_labels = ["A", "B", "C"]
    labels = []
    for row_idx, row in enumerate(rows):
        for col_idx, c in enumerate(row):
            label = f"{row_labels[row_idx]}{col_idx + 1}"
            labels.append((label, c))

    # Save debug image if required
    if debug:
        debug_img = img.copy()
        for label, (x, y, r) in labels:
            cv2.circle(debug_img, (x, y), r, (0, 255, 0), 3)
            cv2.circle(debug_img, (x, y), 2, (0, 0, 255), 3)
            cv2.putText(debug_img, label, (x - 30, y - r - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.imwrite(debug_path, debug_img)
        print(f"Saved debug image as {debug_path}")

    # Save cropped wells if enabled
    cropped_images = {}   # store A1 → image array, etc.

    for label, (x, y, r) in labels:
        x1, y1 = max(0, x - r), max(0, y - r)
        x2, y2 = min(img.shape[1], x + r), min(img.shape[0], y + r)
        crop = img[y1:y2, x1:x2]

        # store in dictionary
        cropped_images[label] = crop

        # Save to disk if needed
        if save_files:
            file_label = f"{plate_type}_plate{plate_number}_{label}.jpg"
            save_path = os.path.join(output_dir, file_label)
            cv2.imwrite(save_path, crop)

    if save_files:
        print(f"All cropped wells saved in {output_dir}/")

    return cropped_images



#===========================SCRIPT=====================================

if __name__ == "__main__":
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    sys.path.insert(0, ROOT)

    root_dir = Path(os.path.join(ROOT, "data", "cropped_data")) # only processed images here

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

    failed_images = [] 

    for img_path in tqdm(image_paths, desc="Cropping wells"):
        try:
            # Extract plate type from the parent directory name
            plate_type = os.path.basename(os.path.dirname(img_path))

            # Run your cropper
            crop_wells(img_path, plate_type)

        except Exception as e:
            print(f"\nError processing {img_path}: {e}")
            failed_images.append((img_path, str(e)))
    
    print("\n" + "="*50)
    print("                      SUMMARY")
    print("="*50)

    if len(failed_images) == 0:
        print("Success! All images have been processed.")
    else:
        print(f"{len(failed_images)} failures out of {len(image_paths)} images:\n")
        for path, error_msg in failed_images:
            print(f"• File : {path}")
            #print(f"  Error: {error_msg}")
    print("="*50)