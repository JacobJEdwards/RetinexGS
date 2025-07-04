import shutil
from pathlib import Path
import subprocess

input_dir = Path("../../LOM_full")
output_dir = Path("../../result")

for d in input_dir.iterdir():
    if not d.is_dir():
        continue

    print(f"Processing {d.name}...")
    cmd = [
        "python",
        "simple_trainer.py",
        "--data_dir", str(d),
        "--result_dir", str(output_dir / d.name),
    ]

    print("Running command:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    shutil.rmtree(str(output_dir / d.name / "tb"))

print("All datasets processed.")





