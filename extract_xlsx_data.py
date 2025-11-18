import pandas as pd
import glob
import math

INPUT_FOLDER = "data/IBV/"
OUTPUT_FILE = "result.xlsx"

rows = []

# Parcours de tous les fichiers xlsx
for file in glob.glob(INPUT_FOLDER + "*.xlsx"):
    df = pd.read_excel(file, header=None)

    filename = file.split("/")[-1].replace(".xlsx", "")
    
    # Les plates commencent à la ligne 3 (index 2), puis tous les +6 lignes
    plate_index = 1
    start_row = 2  # car index 2 = ligne 3 en Excel

    while start_row < df.shape[0]:
        # Récupération de 3 lignes du plate actuel
        plate_block = df.iloc[start_row:start_row + 3]

        # Si bloc vide → stop
        if plate_block.isna().all().all():
            break

        # Pour chaque ligne (A/B/C)
        for i in range(3):
            label = str(plate_block.iloc[i, 4])  # Colonne E = label

            # Si label invalide → ignorer
            if label.lower() == "nan":
                continue

            # Colonnes A–D = 0–3
            for col_idx in range(4):
                value = plate_block.iloc[i, col_idx]

                # Ignorer x, NaN
                if pd.isna(value) or str(value).lower() == "x":
                    continue

                name = f"{filename}_Plate{plate_index}_{label}{col_idx+1}"

                rows.append({
                    "Name": name,
                    "Value": value
                })

        plate_index += 1
        start_row += 6   # next block

# Résultat final
result_df = pd.DataFrame(rows)
result_df.to_excel(OUTPUT_FILE, index=False)
