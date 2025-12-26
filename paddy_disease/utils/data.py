import subprocess
from pathlib import Path

from paddy_disease.config import DataConfig
from paddy_disease.data.download import download_data


def _try_dvc_pull() -> bool:
    try:
        subprocess.run(["dvc", "pull"], check=True)
        return True
    except Exception:
        return False


def ensure_data(data_cfg: DataConfig) -> None:
    raw_dir = Path(data_cfg.raw_dir)

    # если данные есть — ничего не делаем
    if (raw_dir / "train_images").exists() and (raw_dir / "train.csv").exists():
        return

    # 1) пробуем dvc pull
    print("Data not found. Trying dvc pull.")
    ok = _try_dvc_pull()
    if ok and (raw_dir / "train_images").exists():
        print("Data restored via DVC.")
        return

    # 2) иначе качаем с Kaggle
    print("DVC pull did not restore data. Going back to download from Kaggle")
    download_data(data_cfg)
