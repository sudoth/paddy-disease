import shutil
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path

from paddy_disease.config import DataConfig


@dataclass(frozen=True)
class DatasetPaths:
    raw_dir: Path
    train_images: Path
    test_images: Path
    train_csv: Path
    sample_submission_csv: Path


def get_dataset_paths(raw_dir: Path) -> DatasetPaths:
    raw_dir = raw_dir.resolve()
    return DatasetPaths(
        raw_dir=raw_dir,
        train_images=raw_dir / "train_images",
        test_images=raw_dir / "test_images",
        train_csv=raw_dir / "train.csv",
        sample_submission_csv=raw_dir / "sample_submission.csv",
    )


def dataset_is_present(paths: DatasetPaths) -> bool:
    return (
        paths.train_images.exists()
        and paths.test_images.exists()
        and paths.train_csv.exists()
        and paths.sample_submission_csv.exists()
    )


def download_data(data_cfg: DataConfig) -> None:
    """
    Download Kaggle competition data and put it into data/raw/.

    Requires Kaggle API token in your environment.
    And you must have accepted the competition rules on Kaggle website once.
    """
    paths = get_dataset_paths(Path(data_cfg.raw_dir))
    paths.raw_dir.mkdir(parents=True, exist_ok=True)

    if dataset_is_present(paths):
        print(f"Data already present in {paths.raw_dir}.")
        return

    tmp_dir = paths.raw_dir.parent / "_tmp_kaggle"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading dataset from Kaggle")
    subprocess.run(
        [
            "kaggle",
            "competitions",
            "download",
            "-c",
            "paddy-disease-classification",
            "-p",
            str(tmp_dir),
            "--force",
        ],
        check=True,
    )

    zips = sorted(tmp_dir.glob("*.zip"))
    if not zips:
        raise FileNotFoundError(f"No .zip files downloaded into {tmp_dir}")
    zip_path = zips[0]
    print(f"Extracting: {zip_path.name}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmp_dir)

    expected = {
        "train_images": tmp_dir / "train_images",
        "test_images": tmp_dir / "test_images",
        "train.csv": tmp_dir / "train.csv",
        "sample_submission.csv": tmp_dir / "sample_submission.csv",
    }
    missing = [k for k, p in expected.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {missing} in {tmp_dir}")

    print(f"Moving files in {paths.raw_dir}")

    # remove old data if exists
    if paths.train_images.exists():
        shutil.rmtree(paths.train_images)
    if paths.test_images.exists():
        shutil.rmtree(paths.test_images)
    if paths.train_csv.exists():
        paths.train_csv.unlink()
    if paths.sample_submission_csv.exists():
        paths.sample_submission_csv.unlink()

    shutil.move(str(expected["train_images"]), str(paths.train_images))
    shutil.move(str(expected["test_images"]), str(paths.test_images))
    shutil.move(str(expected["train.csv"]), str(paths.train_csv))
    shutil.move(str(expected["sample_submission.csv"]), str(paths.sample_submission_csv))

    shutil.rmtree(tmp_dir)
    print(f"Download complete. Data is in {paths.raw_dir}.")
