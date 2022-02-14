import subprocess
from glob import glob
import os

os.makedirs("data/patterns/", exist_ok=True)

for file in glob("data/NY-PATTERNS*.zip"):
    print("Unzipping", file)
    base_name = os.path.basename(file).replace(".zip", "")
    os.makedirs(os.path.join("data", "patterns", base_name), exist_ok=True)
    subprocess.call(["unzip", file, "-d", os.path.join("data", "patterns", base_name)])
    subprocess.call(
        ["gzip", "-d", os.path.join("data", "patterns", base_name, "patterns.csv.gz")]
    )
    print("Unzipped", file)
