import kagglehub
from pathlib import Path
import shutil

source = Path(kagglehub.dataset_download("dhoogla/ctu13"))
destination = Path("data/raw")
destination.mkdir(parents=True, exist_ok=True)

for file in source.rglob("*.parquet"):
    shutil.copy2(file, destination / file.name)
    print(f"Downloaded: {file.name}")