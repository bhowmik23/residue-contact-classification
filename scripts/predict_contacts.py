import pandas as pd
import xgboost as xgb
import numpy as np
import json
import os
import sys
from sklearn.preprocessing import LabelEncoder

# --- Get PDB name from command-line argument ---
if len(sys.argv) < 2:
    print("Usage: python3 predict_contacts.py <pdb_id>")
    sys.exit(1)

pdb_id = sys.argv[1]

# --- Load merged data ---
merged_path = f"output/{pdb_id}/{pdb_id}_merged.tsv"
df_merge = pd.read_csv(merged_path, sep="\t")

# --- Keep only columns relevant for prediction ---
features_list = [
    "s_ss8", "s_rsa", "s_phi", "s_psi",
    "s_a1", "s_a2", "s_a3", "s_a4", "s_a5",
    "s_3di_state", "s_3di_letter",
    "t_ss8", "t_rsa", "t_phi", "t_psi",
    "t_a1", "t_a2", "t_a3", "t_a4", "t_a5",
    "t_3di_state", "t_3di_letter"
]

# --- Subset data ---
X = df_merge.loc[:, features_list].copy()

# --- Clean and encode 3Di letters ---
X.loc[:, "s_3di_letter"] = X["s_3di_letter"].fillna("A")
X.loc[:, "t_3di_letter"] = X["t_3di_letter"].fillna("A")

unique_letters = pd.concat([
    X["s_3di_letter"],
    X["t_3di_letter"]
]).dropna().unique()

encoder_letters = LabelEncoder()
encoder_letters.fit(unique_letters)

X.loc[:, "s_3di_letter"] = encoder_letters.transform(X["s_3di_letter"]).astype(int)
X.loc[:, "t_3di_letter"] = encoder_letters.transform(X["t_3di_letter"]).astype(int)

# --- Clean and encode DSSP 8-state codes ---

# Replace missing or invalid symbols with "X"
X.loc[:, "s_ss8"] = X["s_ss8"].replace("-", "X").fillna("X")
X.loc[:, "t_ss8"] = X["t_ss8"].replace("-", "X").fillna("X")

# Optionally collapse unknown states to "X"
known_states = set(["H", "E", "G", "I", "B", "T", "S", "C", "X"])
X.loc[:, "s_ss8"] = X["s_ss8"].apply(lambda val: val if val in known_states else "X")
X.loc[:, "t_ss8"] = X["t_ss8"].apply(lambda val: val if val in known_states else "X")

unique_states = pd.concat([
    X["s_ss8"],
    X["t_ss8"]
]).dropna().unique()

ss8_encoder = LabelEncoder()
ss8_encoder.fit(unique_states)

X.loc[:, "s_ss8"] = ss8_encoder.transform(X["s_ss8"]).astype(int)
X.loc[:, "t_ss8"] = ss8_encoder.transform(X["t_ss8"]).astype(int)

# --- Ensure all other columns are numeric ---
for col in X.columns:
    if X[col].dtype == object:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

# --- Load model config ---
with open("model/config.json") as f:
    config = json.load(f)

labels_map = config["labels_map"]

# --- Load ensemble models ---
models = []
for i in range(config["n_classes"]):
    booster = xgb.Booster()
    booster.load_model(f"model/trained-ensemble/xgboost-ovr-ensemble_class_{i}.ubj")
    models.append(booster)

# --- Prepare data for prediction ---
dtest = xgb.DMatrix(X)

# --- Predict with ensemble ---
probas = []
for model in models:
    preds = model.predict(dtest)
    probas.append(preds)

probas = np.vstack(probas).T
predicted_classes = np.argmax(probas, axis=1)

# Map back to string labels
reverse_map = {v: k for k, v in labels_map.items()}
predicted_labels = [reverse_map[i] for i in predicted_classes]

# Add predicted label and score to dataframe
df_merge["Predicted_label"] = predicted_labels
scores = probas[np.arange(len(probas)), predicted_classes]
df_merge["Predicted_score"] = scores.round(6)

# Only keep relevant output columns
output_cols = [
    'pdb_id',
    's_ch','s_resi','s_ins','s_resn',
    's_ss8','s_rsa','s_phi','s_psi',
    's_a1','s_a2','s_a3','s_a4','s_a5',
    's_3di_state','s_3di_letter',
    't_ch','t_resi','t_ins','t_resn',
    't_ss8','t_rsa','t_phi','t_psi',
    't_a1','t_a2','t_a3','t_a4','t_a5',
    't_3di_state','t_3di_letter',
    'Predicted_label','Predicted_score'
]

df_merge_filtered = df_merge.loc[:, output_cols]

# Save result
output_path = f"output/{pdb_id}/{pdb_id}_predictions.tsv"
df_merge_filtered.to_csv(output_path, sep="\t", index=False)
print(f"Predictions saved to {output_path}")


