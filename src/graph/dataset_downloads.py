from __future__ import annotations

import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path


DGL_FRAUD_DATASETS = {
    "amazon": {
        "file_name": "Amazon.mat",
        "url": "https://data.dgl.ai/dataset/FraudAmazon.zip",
    },
    "yelpchi": {
        "file_name": "YelpChi.mat",
        "url": "https://data.dgl.ai/dataset/FraudYelp.zip",
    },
    "yelp": {
        "file_name": "YelpChi.mat",
        "url": "https://data.dgl.ai/dataset/FraudYelp.zip",
    },
}


def _format_mb(num_bytes: int) -> str:
    return f"{num_bytes / (1024 * 1024):.1f} MB"


def _render_progress(downloaded: int, total: int | None) -> None:
    if total and total > 0:
        width = 28
        fraction = min(downloaded / total, 1.0)
        filled = int(width * fraction)
        bar = "#" * filled + "-" * (width - filled)
        message = (
            f"\r[download] [{bar}] {fraction * 100:5.1f}% "
            f"({_format_mb(downloaded)} / {_format_mb(total)})"
        )
    else:
        message = f"\r[download] {_format_mb(downloaded)}"
    sys.stdout.write(message)
    sys.stdout.flush()


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] Fetching {url}")
    with urllib.request.urlopen(url, timeout=120) as response:
        total_header = response.headers.get("Content-Length")
        total = int(total_header) if total_header and total_header.isdigit() else None
        downloaded = 0
        with destination.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                _render_progress(downloaded, total)
    sys.stdout.write("\n")
    sys.stdout.flush()


def _extract_mat_from_zip(zip_path: Path, expected_name: str, destination: Path) -> None:
    with zipfile.ZipFile(zip_path) as archive:
        candidates = [
            member
            for member in archive.namelist()
            if Path(member).name.lower() == expected_name.lower()
        ]
        if not candidates:
            raise FileNotFoundError(f"{expected_name} was not found in {zip_path.name}.")
        member = candidates[0]
        destination.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(member) as source, destination.open("wb") as target:
            shutil.copyfileobj(source, target)


def download_dgl_fraud_dataset(dataset_key: str, data_dir: str | Path = "data") -> Path:
    """Download a DGL fraud .mat dataset into data_dir if it is missing."""

    key = dataset_key.lower()
    if key not in DGL_FRAUD_DATASETS:
        raise KeyError(f"Unknown DGL fraud dataset: {dataset_key}")

    spec = DGL_FRAUD_DATASETS[key]
    data_dir = Path(data_dir)
    destination = data_dir / spec["file_name"]
    if destination.exists():
        return destination

    with tempfile.TemporaryDirectory(prefix="agentgad_dataset_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        zip_path = tmpdir_path / f"{key}.zip"
        extracted_path = tmpdir_path / spec["file_name"]
        _download_file(spec["url"], zip_path)
        _extract_mat_from_zip(zip_path, spec["file_name"], extracted_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(extracted_path), destination)

    return destination


def ensure_default_fraud_datasets(data_dir: str | Path = "data") -> dict[str, Path]:
    """Ensure Amazon.mat and YelpChi.mat are present in data_dir."""

    return {
        "amazon": download_dgl_fraud_dataset("amazon", data_dir=data_dir),
        "yelpchi": download_dgl_fraud_dataset("yelpchi", data_dir=data_dir),
    }


def maybe_download_known_graph_dataset(dataset_path: str | Path) -> Path:
    path = Path(dataset_path)
    if path.exists():
        return path

    file_name = path.name.lower()
    if file_name == "amazon.mat":
        return download_dgl_fraud_dataset("amazon", data_dir=path.parent)
    if file_name == "yelpchi.mat":
        return download_dgl_fraud_dataset("yelpchi", data_dir=path.parent)
    return path
