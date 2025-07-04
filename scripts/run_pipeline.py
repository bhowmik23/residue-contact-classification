import subprocess
import sys
import os

def run_command(command, cwd=None):
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, cwd=cwd)
    if result.returncode != 0:
        print("Error running command:", command)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_pipeline.py <pdb_file.cif>")
        sys.exit(1)

    pdb_file = sys.argv[1]
    pdb_id = os.path.splitext(os.path.basename(pdb_file))[0]

    # Create output folder for this protein
    out_dir = f"output/{pdb_id}/"
    os.makedirs(out_dir, exist_ok=True)

    # Step 1: Run calc_features.py
    run_command(f"python3 scripts/calc_features.py {pdb_file} -out_dir {out_dir}")

    # Step 2: Run calc_3di.py
    run_command(f"python3 scripts/calc_3di.py {pdb_file} -out_dir {out_dir}")

    # Step 3: Run merge_features.py
    run_command(f"python3 scripts/merge_features.py {pdb_id}")

    # Step 4: Run predict_contacts.py
    run_command(f"python3 scripts/predict_contacts.py {pdb_id}")

    print(f" Pipeline completed for {pdb_file}")
