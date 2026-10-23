import os
import subprocess


base_dir = "results2/"  #


selected_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

for dir in selected_dirs:
    dir_path = os.path.join(base_dir, dir)
    print(f"Dang xu ly: {dir_path}")
    cmd = ["python", "evaluateSeparation.py", "--results_dir", dir_path]
    subprocess.run(cmd)