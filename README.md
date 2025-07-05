# RING Classification of Residue-Residue Contacts in Protein Structures

This repository contains the implementation of a supervised predictor for residue–residue contact types in protein structures. The pipeline processes a PDB/mmCIF file, extracts structural features, and predicts RING interaction classes using an XGBoost OvR ensemble.

Authors: [Anahita Soltantouyeh](https://github.com/anahita-soltan), [Mikael Poli](https://github.com/mikaelpoli), [Biddut Bhowmik](https://github.com/bhowmik23) – Structural Bioinformatics Course – UNIPD – A.Y. 2024/25

## Project Overview

Protein structures are stabilized by various residue-residue interactions, including hydrogen bonds, van der Waals contacts, disulfide bridges, salt bridges, and $\pi-\pi$ interactions. Accurately classifying these contacts is essential for understanding protein folding, stability, and function.
Traditionally, software like RING ([Del Conte et al., 2024](https://academic.oup.com/nar/article/52/W1/W306/7660079)) identifies and classifies residue-residue contacts in protein structures using geometric and physicochemical rules. In this project, we experiment with a machine learning-based approach to predict the type of contacts without relying directly on geometric thresholds.

For full implementation details, we recommend reading the [report](https://github.com/mikaelpoli/residue-contact-classification/blob/main/report.pdf).

## Essential Repo Structure
[root.]
  - 3di_model/
  - data/
  - figures/
  - model/
  - output/
  - scripts/
  - supplementary/
  - src/
main.ipynb
combine_dfs.py

### Component Descriptions
* `combine_dfs.py`: Merges a list of .tsv feature files (one per PDB structure) into a single dataframe. This unified dataset can be passed to `main.ipynb` for preprocessing and model training.

* `main.ipynb`: Jupyter notebook for data preprocessing and model training. It uses datasets located in the `data/ directory (resampled training, validation, and test sets). To retrain the model, run the Setup and Model: XGBoost sections.

* `src/`: Contains utility functions used throughout the pipeline, including feature processing and model handling.

* `model/`: Stores the trained XGBoost ensemble, its configuration file, and evaluation metrics on the test set.

* `supplementary/`: Includes an alternative single multiclass XGBoost model trained for comparative analysis.

* `3di_model/`, `scripts/`, `output/`: Contain code and outputs related to the contact type prediction software.

## Running the Pipeline
Run from the terminal:

```
python3 scripts/run_pipeline.py <path_to_cif_file>
```
Example:


```
python3 scripts/run_pipeline.py pdb_files/6vw9.cif
```

## Output
All results are saved in:

`output/<pdb_id>/`
#### Files include:
    * `<pdb_id>_features.tsv` → residue features
    * `<pdb_id>_3di.tsv` → 3Di states
    * `<pdb_id>_merged.tsv` → merged feature table
    * `<pdb_id>_predictions.tsv` → final predictions
    
#### Each contact prediction includes:
    * Source & target residues
    * Structural and physico-chemical features
    * Predicted contact type
    * Prediction score
    
### Scripts Overview
#### scripts/calc_features.py
* Loads a PDB/mmCIF structure.
* Computes residue-level features:
	* Secondary structure (DSSP 8-state)
	* Relative solvent accessibility
	* Half-sphere exposure values
	* Ramachandran angles and regions
	* Atchley scale physicochemical properties
* Detects residue pairs in contact based on a distance threshold.
* Outputs a TSV file with all contacts and features.

#### scripts/calc_3di.py
* Loads the same PDB/mmCIF structure.
* Computes geometric descriptors for residues.
* Encodes local 3D environments into the FoldSeek 3Di alphabet.
* Outputs per-residue 3Di state and letter codes.

#### scripts/merge_features.py
* Joins the outputs of `calc_features.py` and `calc_3di.py`.
* Adds 3Di state and letter for both source and target residues in contact pairs.
* Produces a unified TSV feature table. 

#### scripts/predict_contacts.py
* Loads the merged features TSV.
* Filters and encodes categorical features for the ML model.
* Runs an ensemble of XGBoost classifiers.
* Outputs contact-level predictions:
	* RING contact class
	* Probability score

#### scripts/run_pipeline.py
* Orchestrates the full pipeline:
  1. Runs feature extraction
  2. Runs 3Di calculation
  3. Merges features
  4. Predicts contact types
* Ensures correct file naming and folder structure.

### Configuration

Adjust configuration.json if you need to change:
* DSSP executable path
* Ramachandran file path
* Atchley scale file path
* Distance thresholds for contact detection

Example snippet:

```json
{
  "dssp_file": "/usr/local/bin/mkdssp",
  "rama_file": "ramachandran.dat",
  "atchley_file": "atchley.tsv",
  "distance_threshold": 3.5,
  "sequence_separation": 3
}
```

### Requirements
* Python 3.9+
* Biopython
* pandas
* numpy
* scikit-learn
* XGBoost
* Torch (for FoldSeek encoder)
* DSSP installed and accessible from the configured path
