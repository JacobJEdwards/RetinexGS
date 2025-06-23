from pathlib import Path
import subprocess

input_dir = Path("../../NeRF_360")
output_dir = Path("../../result")

for d in input_dir.iterdir():
    if not d.is_dir():
        continue
    
    print(f"Processing {d.name}...")
    cmd = [
        "python",
        "simple_trainer.py",
        "default "
        "--data_dir", str(d),
        "--result_dir", str(output_dir / d.name),
    ]
    
    print("Running command:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    
print("All datasets processed.")
    

    
    
    
    