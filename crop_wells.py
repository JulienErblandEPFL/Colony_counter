import cv2
import numpy as np
import os

def crop_wells(file_path, plate_type, debug=False, save_files=True):
    # Extract plate name and number from file path
    base_name = os.path.splitext(os.path.basename(file_path))[0]  # e.g. Plate_1
    plate_number = ''.join(filter(str.isdigit, base_name))        # "1", "7", etc.
    plate_name = base_name.lower()                                # "plate_1"

    # Debug file path
    debug_path = f"{plate_name}_detected_debug.jpg"

    # New directory structure:
    # cropped_wells/{plate_type}/plate_x/
    base_dir = "cropped_wells"
    type_dir = os.path.join(base_dir, plate_type)
    output_dir = os.path.join(type_dir, plate_name)

    # Ensure directories exist
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(type_dir, exist_ok=True)
    if save_files:
        os.makedirs(output_dir, exist_ok=True)

    # Load and preprocess the image
    img = cv2.imread(file_path)
    if img is None:
        raise ValueError(f"Image not found: {file_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)
    gray = cv2.equalizeHist(gray)

    # Base parameters for Hough Circles
    params = dict(
        dp=1.2,
        minDist=150,
        param1=100,
        param2=60,
        minRadius=130,
        maxRadius=160
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
        print("Did not find exactly 12 circles. Using the closest result.")
        best_diff = float("inf")
        best_p = None

        for p in range(40, 100, 5):
            params["param2"] = p
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, **params)
            if circles is not None:
                diff = abs(len(circles[0]) - 12)
                if diff < best_diff:
                    best_diff = diff
                    best_circles = np.around(circles[0, :]).astype(int)
                    best_p = p

        if best_circles is not None:
            print(f"Using {len(best_circles)} circles (param2={best_p})")
        else:
            raise RuntimeError("No circles detected.")

    # Group circles into rows using Y coordinate
    best_circles = sorted(best_circles, key=lambda c: c[1])
    ys = [c[1] for c in best_circles]
    row_threshold = np.std(ys) * 0.5

    rows = []
    current_row = [best_circles[0]]

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
    if save_files:
        for label, (x, y, r) in labels:
            x1, y1 = max(0, x - r), max(0, y - r)
            x2, y2 = min(img.shape[1], x + r), min(img.shape[0], y + r)
            crop = img[y1:y2, x1:x2]

            file_label = f"{plate_type}_plate{plate_number}_{label}.jpg"
            save_path = os.path.join(output_dir, file_label)
            cv2.imwrite(save_path, crop)

        print(f"All cropped wells saved in {output_dir}/")

    return labels
