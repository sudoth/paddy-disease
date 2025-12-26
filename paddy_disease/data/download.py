import shutil
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetPaths:
    """
    Class for dataset path.
    """

    raw_dir: Path
    train_images: Path
    test_images: Path
    train_csv: Path
    sample_submission_csv: Path


def get_dataset_paths(repo_root: Path) -> DatasetPaths:
    """
    Dataset path extraction function.

    :param repo_root: Description
    :type repo_root: Path
    :return: Description
    :rtype: DatasetPaths
    """
    raw_dir = repo_root / "data" / "raw"
    return DatasetPaths(
        raw_dir=raw_dir,
        train_images=raw_dir / "train_images",
        test_images=raw_dir / "test_images",
        train_csv=raw_dir / "train.csv",
        sample_submission_csv=raw_dir / "sample_submission.csv",
    )


def dataset_is_present(paths: DatasetPaths) -> bool:
    """
    Cheks if dataset is already exists.

    :param paths: Description
    :type paths: DatasetPaths
    :return: Description
    :rtype: bool
    """
    return (
        paths.train_images.exists()
        and paths.test_images.exists()
        and paths.train_csv.exists()
        and paths.sample_submission_csv.exists()
    )


def download_data(repo_root: Path) -> None:
    """
    Download Kaggle competition data and put it into data/raw/.

    Requires Kaggle API token in your enviroment.
    And you must have accepted the competition rules on Kaggle website once.
    """
    paths = get_dataset_paths(repo_root)
    paths.raw_dir.mkdir(parents=True, exist_ok=True)

    if dataset_is_present(paths):
        print("Data already present in data/raw/.")
        return

    tmp_dir = repo_root / "data" / "_tmp_kaggle"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading dataset from Kaggle")
    # --force to re-download if partial file exists
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
    zip_path = zips[0]
    print(f"Extracting: {zip_path.name}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmp_dir)

    # In the zip: train_images/, test_images/, train.csv, sample_submission.csv
    expected = {
        "train_images": tmp_dir / "train_images",
        "test_images": tmp_dir / "test_images",
        "train.csv": tmp_dir / "train.csv",
        "sample_submission.csv": tmp_dir / "sample_submission.csv",
    }

    print("Moving files into data/raw/ ...")
    # remove partial old data if exists
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
    print("Download complete. Data is in data/raw/.")
