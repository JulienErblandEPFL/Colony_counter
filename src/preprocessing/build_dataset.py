import os
import pandas as pd
from sklearn.model_selection import train_test_split

# CONFIG
LABEL_FILE = "data/global_counting_data.xlsx"
WELLS_DIR = "data/cropped_wells"
FULL_DATASET_CSV = "data/dataset.csv"
TRAIN_CSV = "data/training.csv"
TEST_CSV = "data/test.csv"
TEST_SIZE = 0.2  # 80/20 split


def build_dataset():
    print("Loading labels from:", LABEL_FILE)
    df = pd.read_excel(LABEL_FILE)

    df["filename"] = df["Name"].astype(str) + ".jpg"
    label_dict = dict(zip(df["filename"], df["Value"]))

    rows = []

    print("Scanning images in:", WELLS_DIR)
    for root, _, files in os.walk(WELLS_DIR):
        for f in files:
            if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            full_path = os.path.join(root, f)
            value = label_dict.get(f, None)

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

    os.makedirs(os.path.dirname(FULL_DATASET_CSV), exist_ok=True)
    dataset.to_csv(FULL_DATASET_CSV, index=False)

    print(f"\nDataset saved to {FULL_DATASET_CSV}")
    print(f"Total wells: {len(dataset)}")
    print(f"Labeled wells: {dataset['value'].notna().sum()}")
    print(f"Countable wells: {dataset['is_countable'].sum()}")
    print(f"Uncountable wells: {(dataset['is_countable'] == 0).sum()}")

    # ---- STRATIFIED SPLIT ----
    dataset["is_zero"] = (dataset["value"] == 0).astype(int)

    # Create the stratum label: 'countable_zero', etc.
    dataset["strata"] = (
        dataset["is_countable"].astype(str)
        + "_" +
        dataset["is_zero"].astype(str)
    )

    print("\nDistribution per strata:")
    print(dataset["strata"].value_counts())

    train_df, test_df = train_test_split(
        dataset,
        test_size=TEST_SIZE,
        random_state=42,
        stratify=dataset["strata"]
    )

    # Save
    train_df.to_csv(TRAIN_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)

    print(f"\nTraining split saved to {TRAIN_CSV}")
    print(f"Test split saved to {TEST_CSV}")

    # Confirm distributions
    print("\n=== FINAL DISTRIBUTIONS CHECK ===")
    print("\nTrain strata:")
    print(train_df["strata"].value_counts(normalize=True))
    print("\nTest strata:")
    print(test_df["strata"].value_counts(normalize=True))


if __name__ == "__main__":
    build_dataset()
