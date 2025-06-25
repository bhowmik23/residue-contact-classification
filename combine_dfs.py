"""
REQUIRES:
- 'features_ring' directory containing .tsv files of PDB structure features

OUTPUTS:
- A single 'data.csv' file of the combined .tsv files
"""

# SETUP
# --- Libraries ---
import gc
import os
import pandas as pd
from pathlib import Path

# --- Paths ---
# Directories
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path().resolve()
SOURCE_DATA_DIR = BASE_DIR / 'features_ring'
DATA_DIR = BASE_DIR / 'data'
DATA_DIR.mkdir(exist_ok=True)

# Files
OUTPUT_FILE = DATA_DIR / 'data.csv'

# IMPORT DATA
dfs = []
for filename in os.listdir(SOURCE_DATA_DIR):
    dfs.append(pd.read_csv(f'{SOURCE_DATA_DIR}/{filename}', sep='\t'))
df = pd.concat(dfs)

# CLEAN DATA
# --- Create "Unclassified" class ---
df['Interaction'] = df['Interaction'].fillna('Unclassified')

# --- Remove NA's ---
df.dropna(inplace=True)

# EXPORT DATA
df.to_csv(OUTPUT_FILE, index=False)

# CLEAR MEMORY
del dfs
gc.collect()