import os
import pandas as pd


# CONFIG
LABEL_FILE = "data/global_counting_data.xlsx"
WELLS_DIR = "data/cropped_wells"
OUTPUT_CSV = "data/dataset.csv"


def build_dataset():
    # Load Excel file
    print("Loading labels from:", LABEL_FILE)
    df = pd.read_excel(LABEL_FILE)

    # Expect columns: ["name", "value"]
    df["filename"] = df["Name"].astype(str) + ".jpg"

    # Create lookup: "Panama_plate1_A1.jpg" → value
    label_dict = dict(zip(df["filename"], df["Value"]))

    rows = []

    # Walk through cropped well images
    print("Scanning images in:", WELLS_DIR)
    for root, _, files in os.walk(WELLS_DIR):
        for f in files:
            if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            full_path = os.path.join(root, f)

            value = label_dict.get(f, None)  # Match by filename

            if value is None:
                is_countable = 0
            else:
                try:
                    num_val = float(value)
                    is_countable = 1 if num_val >= 0 else 0
                except:
                    is_countable = 0

            rows.append({
                "filename": f,
                "value": value,
                "is_countable": is_countable,
                "path": full_path,
                "plate_type": f.split("_")[0],
                "plate_id": f.split("_")[1],
                "well": f.split("_")[2].split(".")[0],
            })

    dataset = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    dataset.to_csv(OUTPUT_CSV, index=False)

    print(f"Dataset saved to {OUTPUT_CSV}")
    print(f"Total wells: {len(dataset)}")
    print(f"Labeled wells: {dataset['value'].notna().sum()}")
    print(f"Countable wells: {dataset['is_countable'].sum()}")
    print(f"Uncountable wells: {(dataset['is_countable'] == 0).sum()}")



if __name__ == "__main__":
    build_dataset()
