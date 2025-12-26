import subprocess
from pathlib import Path

from paddy_disease.data.download import dataset_is_present, download_data, get_dataset_paths


def repo_root() -> Path:
    """
    Function that returns repo root.

    :return: Description
    :rtype: Path
    """
    return Path(__file__).parents[2]


def try_dvc_pull(root: Path) -> bool:
    """
    Returns True if dvc pull ran successfully, False otherwise.
    """
    try:
        subprocess.run(["dvc", "pull"], cwd=root, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, PermissionError):
        return False


def ensure_data() -> None:
    """
    Tries to download via DVC in train/infer
    If local storage can't provide data, download from Kaggle
    """
    root = repo_root()
    paths = get_dataset_paths(root)

    if dataset_is_present(paths):
        return

    print("Data not found. Trying dvc pull.")
    _ = try_dvc_pull(root)

    # check again after pull
    if dataset_is_present(paths):
        print("Data restored via DVC.")
        return

    print("DVC pull did not restore data. Going back to download from Kaggle")
    download_data(root)
