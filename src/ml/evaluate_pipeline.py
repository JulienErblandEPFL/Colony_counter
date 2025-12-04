import sys
import os

# Find project root (folder containing "src/")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT)

import argparse
import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader

from src.ml.data.dataset import ColonyDataset
from src.ml.data.transforms import get_classifier_test_transforms,get_counter_test_transforms
from src.ml.models.CountabilityClassifier import CountabilityClassifier

# Adjust based on your setup:
from src.ml.models.ResNet34 import ResNet34Regressor
from src.ml.models.EfficientNet import EfficientNetB0Regressor


# ---------------------
# DEVICE PICKER
# ---------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------
# LOAD COUNTING MODELS
# ---------------------
def load_counter_model(name, weights_path):
    if name == "EfficientNet":
        model = EfficientNetB0Regressor(pretrained=False)
    elif name == "ResNet34":
        model = ResNet34Regressor(pretrained=True, dropout_p=0.5, freeze_backbone=False)
    else:
        raise ValueError(f"Unknown counting model '{name}'")

    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# ---------------------
# EVAL LOGIC
# ---------------------
def evaluate_full_pipeline(
    csv_path,
    classifier_weights,
    counter_model_name,
    counter_model_weights,
    threshold=0.5,
    output_csv="pipeline_results.csv",
):
    print("[INFO] Loading dataset:", csv_path)
    df = pd.read_csv(csv_path)

    # Dataset for classification ONLY
    ds = ColonyDataset(
        df=df,
        transform=get_classifier_test_transforms(),
        task="classify"
    )

    loader = DataLoader(ds, batch_size=32, shuffle=False)

    # Load classifier
    classifier = CountabilityClassifier(backbone="efficientnet_b0", pretrained=False)
    classifier.load_state_dict(torch.load(classifier_weights, map_location=device))
    classifier.to(device)
    classifier.eval()

    # Load counter
    counter = load_counter_model(counter_model_name, counter_model_weights)

    results = []

    print("[INFO] Starting evaluation...")

    with torch.no_grad():
        # Loop over dataset in batches
        for batch_imgs, batch_labels in loader:
            batch_imgs = batch_imgs.to(device)

            # -----------------------------------
            # 1. CLASSIFIER PREDICTION FOR BATCH
            # -----------------------------------
            logits = classifier(batch_imgs)                 # shape [B, 2]
            probs = F.softmax(logits, dim=1)                # convert to probabilities
            countable_scores = probs[:, 1]                  # probability of class=1 ("countable")
            countable_scores_np = countable_scores.cpu().numpy()

            # Binary prediction using threshold
            batch_preds = (countable_scores_np >= threshold).astype(int)


            # -----------------------------------
            # 2. PROCESS EACH IMAGE IN THE BATCH
            # -----------------------------------
            for batch_index, pred_is_countable in enumerate(batch_preds):

                # Use global index based on how many we've already saved
                global_idx = len(results)

                # Grab the matching row from the original dataframe
                df_row = ds.df.iloc[global_idx]

                record = {
                    "path": df_row["path"],
                    "true_is_countable": int(df_row["is_countable"]),
                    "pred_is_countable": int(pred_is_countable),
                    "confidence": float(countable_scores_np[batch_index]),
                }

                # ---------------------------------------------------------
                # 3. If classifier predicts COUNTABLE → run counter model
                # ---------------------------------------------------------
                if pred_is_countable == 1:
                    # Build a 1-row dataset to load the image again
                    # Using "classify" mode ensures NO filtering
                    single_df = ds.df.iloc[[global_idx]]

                    ds_img = ColonyDataset(
                        df=single_df,
                        transform=get_counter_test_transforms(),
                        task="classify"
                    )

                    img_single, _ = ds_img[0]                   # we ignore the label
                    img_single = img_single.unsqueeze(0).to(device)

                    # Round prediction to nearest integer and ensure non-negative
                    raw_pred = counter(img_single).item()
                    predicted_count = int(max(0, round(raw_pred)))
                    
                    record["predicted_count"] = predicted_count
                    record["true_value"] = df_row.get("value", None)

                else:
                    # Classifier rejected this well → no counting performed
                    record["predicted_count"] = -1 #-1 is the value associated to uncountable
                    record["true_value"] = df_row.get("value", None)

                # Save record
                results.append(record)

    filename = os.path.basename(csv_path).lower()  # e.g: test_augmented.csv

    if "aug" in filename or "augmented" in filename:
        output_csv = "pipeline_results_augmented.csv"
    else:
        output_csv = "pipeline_results.csv"

    # Save results
    df_out = pd.DataFrame(results)
    df_out.to_csv(output_csv, index=False)

    # ----------------
    # SUMMARY STATS
    # ----------------

    total = len(df_out)

    # Basic classifier stats
    pred_c = df_out["pred_is_countable"]
    true_c = df_out["true_is_countable"]

    uncountable_pred = (pred_c == 0).sum()
    countable_pred = (pred_c == 1).sum()

    classifier_accuracy = (pred_c == true_c).mean()

    # Precision / recall for "countable" class
    tp = ((pred_c == 1) & (true_c == 1)).sum()
    fp = ((pred_c == 1) & (true_c == 0)).sum()
    tn = ((pred_c == 0) & (true_c == 0)).sum()
    fn = ((pred_c == 0) & (true_c == 1)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0

    # --------------------------
    # Counter metrics
    # Only on wells that are:
    # 1. truly countable
    # 2. predicted countable
    # --------------------------
    valid_mask = (true_c == 1) & (pred_c == 1)

    valid_df = df_out[valid_mask].copy()

    counter_accuracy = None
    mae = None
    r2_score = None
    within_10pct = None

    if len(valid_df) > 0:
        # Extract true/pred counts
        y_true = valid_df["true_value"].astype(float)
        y_pred = valid_df["predicted_count"].astype(float)

        # --- MAE ---
        abs_error = (y_pred - y_true).abs()
        mae = abs_error.mean()

        # --- 10% margin ---
        pct_err = abs_error / y_true.replace(0, 1)  # avoid division by zero
        within_10pct = (pct_err <= 0.10).mean()

        counter_accuracy = within_10pct  # you already use this proxy

        # --- R² SCORE ---
        ss_res = ((y_true - y_pred) ** 2).sum()
        ss_tot = ((y_true - y_true.mean()) ** 2).sum()

        if ss_tot == 0:
            # All ground truth values identical → R² undefined
            r2_score = None
        else:
            r2_score = 1 - (ss_res / ss_tot)


    # -----------------------------------------------
    # Predictions on wells that are TRULY uncountable
    # but CLASSIFIER predicted as countable
    # -----------------------------------------------
    bad_mask = (true_c == 0) & (pred_c == 1)
    bad_predictions = df_out[bad_mask][["path", "predicted_count", "confidence"]]

    # Print
    print("\n========== FULL EVALUATION ==========")
    print(f"Total wells:                         {total}")
    print(f"Predicted COUNTABLE:                 {countable_pred}")
    print(f"Predicted UNCOUNTABLE:               {uncountable_pred}")
    print(f"\n--- CLASSIFIER METRICS ---")
    print(f"Classifier Accuracy:                 {classifier_accuracy:.3f}")
    print(f"Precision (countable):               {precision:.3f}")
    print(f"Recall    (countable):               {recall:.3f}")
    print(f"Confusion Matrix:")
    print(f"   TP: {tp} | FP: {fp}")
    print(f"   FN: {fn} | TN: {tn}")

    # ----------------------------------------
    # SHOW ACTUAL MISCLASSIFIED IMAGES
    # ----------------------------------------

    # Boolean masks
    fp_mask = (pred_c == 1) & (true_c == 0)   # FP: wrongly accepted
    fn_mask = (pred_c == 0) & (true_c == 1)   # FN: wrongly rejected

    fps = df_out[fp_mask].copy()
    fns = df_out[fn_mask].copy()

    print("\n--- MISCLASSIFIED WELLS ---")
    # False Positives
    print(f"\nFALSE POSITIVES (Predicted COUNTABLE but Truly UNCOUNTABLE): {len(fps)}")
    if len(fps) > 0:
        cols = ["path", "confidence", "predicted_count", "true_value"]
        print(fps[cols].head(20).to_string(index=False))
    else:
        print("None")

    # False Negatives
    print(f"\nFALSE NEGATIVES (Predicted UNCOUNTABLE but Truly COUNTABLE): {len(fns)}")
    if len(fns) > 0:
        cols = ["path", "confidence", "true_value"]
        print(fns[cols].head(20).to_string(index=False))
    else:
        print("None")
    
    print("\n--- COUNTER METRICS ---")
    if counter_accuracy is not None:
        print(f"MAE (true countable & predicted countable):         {mae:.3f}")
        print(f"R² score:                                           {r2_score:.3f}" if r2_score is not None else "R² score: Undefined (no variance)")
        print(f"% within 10% of true value:                        {within_10pct*100:.2f}%")
    else:
        print("No valid 'true countable & predicted countable' wells to evaluate counter model.")



    print(f"\nResults saved to: {output_csv}")



# ---------------------
# ENTRY POINT
# ---------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate full pipeline: classifier + counter")
    parser.add_argument("--csv", default="data/test.csv")
    parser.add_argument("--classifier_weights", default="countability_classifier.pth")
    parser.add_argument("--counter_model", default="EfficientNet", choices=["EfficientNet", "ResNet34"])
    parser.add_argument("--counter_weights", default="efficientnet_b0_colony.pth")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", default="pipeline_results.csv")
    args = parser.parse_args()

    evaluate_full_pipeline(
        csv_path=args.csv,
        classifier_weights=args.classifier_weights,
        counter_model_name=args.counter_model,
        counter_model_weights=args.counter_weights,
        threshold=args.threshold,
        output_csv=args.output,
    )
