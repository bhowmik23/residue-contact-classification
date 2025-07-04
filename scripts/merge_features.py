import os
import sys
import pandas as pd

# Read the PDB ID from the command-line
pdb_id = sys.argv[1]

# Paths to input files
features_file = f"output/{pdb_id}/{pdb_id}_features.tsv"
three_di_file = f"output/{pdb_id}/{pdb_id}_3di.tsv"

# Load input files
df_contacts = pd.read_csv(features_file, sep="\t")
df_3di = pd.read_csv(three_di_file, sep="\t")

# Prepare source residue 3Di data
df_3di_src = df_3di.rename(columns={
    "ch": "s_ch",
    "resi": "s_resi",
    "ins": "s_ins",
    "resn": "s_resn",
    "3di_state": "s_3di_state",
    "3di_letter": "s_3di_letter"
})

df_merge = df_contacts.merge(
    df_3di_src[["pdb_id", "s_ch", "s_resi", "s_ins", "s_resn", "s_3di_state", "s_3di_letter"]],
    on=["pdb_id", "s_ch", "s_resi", "s_ins", "s_resn"],
    how="left"
)

# Prepare target residue 3Di data
df_3di_tgt = df_3di.rename(columns={
    "ch": "t_ch",
    "resi": "t_resi",
    "ins": "t_ins",
    "resn": "t_resn",
    "3di_state": "t_3di_state",
    "3di_letter": "t_3di_letter"
})

df_merge = df_merge.merge(
    df_3di_tgt[["pdb_id", "t_ch", "t_resi", "t_ins", "t_resn", "t_3di_state", "t_3di_letter"]],
    on=["pdb_id", "t_ch", "t_resi", "t_ins", "t_resn"],
    how="left"
)

# Save merged file into same protein folder
out_dir = f"output/{pdb_id}/"
os.makedirs(out_dir, exist_ok=True)
merged_file = os.path.join(out_dir, f"{pdb_id}_merged.tsv")

df_merge.to_csv(merged_file, sep="\t", index=False)
print(f"Merged file saved to {merged_file}")
