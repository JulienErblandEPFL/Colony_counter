import os
import pandas as pd
from sklearn.model_selection import train_test_split

# CONFIG
LABEL_FILE = "data/global_counting_data.xlsx"
BASE_WELLS_DIR = "data/cropped_wells"
BASE_FULL_DATASET_CSV = "data/dataset.csv"
BASE_TRAIN_CSV = "data/training.csv"
BASE_TEST_CSV = "data/test.csv"
TEST_SIZE = 0.2  # 80/20 split

def get_base_filename(fname: str) -> str:
    """
    Converts augmented names like:
        IBV_plate1_A1_aug_0.jpg
        IBV_plate1_A1_aug_15.jpg
        IBV_plate1_A1_orig.jpg
    into the canonical label key:
        IBV_plate1_A1.jpg
    """
    name, ext = os.path.splitext(fname)

    # Split by '_'
    tokens = name.split("_")

    # Only keep the first 3 tokens:
    # 0: plate_type   (IBV)
    # 1: plate_id     (plate1)
    # 2: well         (A1)
    base = "_".join(tokens[:3]) + ext
    return base

def build_dataset(augmented: bool = False):
    """
    Build dataset CSV from cropped wells images.

    Parameters
    ----------
    augmented : bool
        If True, load from augmented dataset and produce *_augmented CSV outputs.
    """

    # ---- Paths depending on augmentation mode ----
    if augmented:
        wells_dir = "data/augmented_wells"
        full_dataset_csv = BASE_FULL_DATASET_CSV.replace(".csv", "_augmented.csv")
        train_csv = BASE_TRAIN_CSV.replace(".csv", "_augmented.csv")
        test_csv = BASE_TEST_CSV.replace(".csv", "_augmented.csv")
    else:
        wells_dir = BASE_WELLS_DIR
        full_dataset_csv = BASE_FULL_DATASET_CSV
        train_csv = BASE_TRAIN_CSV
        test_csv = BASE_TEST_CSV

    mode = "AUGMENTED DATASET" if augmented else "ORIGINAL DATASET"
    print(f"\n=== Building {mode} ===")
    print("Loading labels from:", LABEL_FILE)

    # ---- Load labels ----
    df = pd.read_excel(LABEL_FILE)

    # Expected columns: ["Name", "Value"]
    df["filename"] = df["Name"].astype(str) + ".jpg"
    label_dict = dict(zip(df["filename"], df["Value"]))

    rows = []

    # ---- Scan wells directory ----
    print("Scanning images in:", wells_dir)
    for root, _, files in os.walk(wells_dir):
        for f in files:
            if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            full_path = os.path.join(root, f)
            base_name = get_base_filename(f)
            value = label_dict.get(base_name, None)

            # Determine if well is countable
            if value is None:
                is_countable = 0
            else:
                try:
                    num_val = float(value)
                    is_countable = 1 if num_val >= 0 else 0
                except:
                    is_countable = 0

            # Extract plate info from base_name
            base_tokens = os.path.splitext(base_name)[0].split("_")

            rows.append({
                "filename": f,                # augmented filename
                "base_name": base_name,       # canonical label name
                "value": value,
                "is_countable": is_countable,
                "path": full_path,
                "plate_type": base_tokens[0] if len(base_tokens) > 0 else None,
                "plate_id": base_tokens[1] if len(base_tokens) > 1 else None,
                "well": base_tokens[2] if len(base_tokens) > 2 else None,
            })

    dataset = pd.DataFrame(rows)

    # ---- Save the full dataset ----
    os.makedirs(os.path.dirname(full_dataset_csv), exist_ok=True)
    dataset.to_csv(full_dataset_csv, index=False)

    print(f"\nDataset saved to {full_dataset_csv}")
    print(f"Total wells: {len(dataset)}")
    print(f"Labeled wells: {dataset['value'].notna().sum()}")
    print(f"Countable wells: {dataset['is_countable'].sum()}")
    print(f"Uncountable wells: {(dataset['is_countable'] == 0).sum()}")

    # ---- STRATIFIED SPLIT ----
    # Additional feature: wells with 0 colonies
    dataset["is_zero"] = (dataset["value"] == 0).astype(int)

    # Create strata labels
    dataset["strata"] = (
        dataset["is_countable"].astype(str)
        + "_"
        + dataset["is_zero"].astype(str)
    )

    print("\nDistribution per strata:")
    print(dataset["strata"].value_counts())

    train_df, test_df = train_test_split(
        dataset,
        test_size=TEST_SIZE,
        random_state=42,
        stratify=dataset["strata"],
    )

    # Save splits
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    print(f"\nTraining split saved to {train_csv}")
    print(f"Test split saved to {test_csv}")

    # Confirm distributions
    print("\n=== FINAL DISTRIBUTIONS CHECK ===")
    print("\nTrain strata:")
    print(train_df["strata"].value_counts(normalize=True))
    print("\nTest strata:")
    print(test_df["strata"].value_counts(normalize=True))


if __name__ == "__main__":
    # Example usage
    #build_dataset(augmented=False)
    build_dataset(augmented=True)  # when needed
