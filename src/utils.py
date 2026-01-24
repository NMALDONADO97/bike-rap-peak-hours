from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path
import requests

UCI_BIKE_ZIP_URL = "https://archive.ics.uci.edu/static/public/275/bike%2Bsharing%2Bdataset.zip"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest: Path, timeout: int = 60) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 128):
                if chunk:
                    f.write(chunk)


def ensure_ucibike_data(raw_dir: Path, level: str = "hour") -> Path:
    """Ensure the dataset exists locally. Returns path to day.csv or hour.csv."""
    raw_dir.mkdir(parents=True, exist_ok=True)

    zip_path = raw_dir / "bike_sharing_dataset.zip"
    if not zip_path.exists():
        print(f"Downloading dataset from UCI -> {zip_path}")
        download_file(UCI_BIKE_ZIP_URL, zip_path)
    else:
        print(f"Dataset ZIP already exists -> {zip_path}")

    target_name = "day.csv" if level == "day" else "hour.csv"
    target_path = raw_dir / target_name

    if not target_path.exists():
        print(f"Extracting {target_name} from ZIP...")
        with zipfile.ZipFile(zip_path, "r") as z:
            candidates = [n for n in z.namelist() if n.endswith(f"/{target_name}") or n.endswith(target_name)]
            if not candidates:
                raise FileNotFoundError(
                    f"{target_name} not found inside {zip_path}. Contents: {z.namelist()[:10]}..."
                )
            member = candidates[0]
            with z.open(member) as src, target_path.open("wb") as dst:
                dst.write(src.read())
    else:
        print(f"{target_name} already extracted -> {target_path}")

    return target_path
