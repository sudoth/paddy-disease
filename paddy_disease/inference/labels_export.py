import json
from pathlib import Path

from paddy_disease.data.labels import build_label_mapping, read_train_csv
from paddy_disease.utils.data import ensure_data


def export_labels(out_path: Path = Path("models/labels.json")) -> None:
    ensure_data()
    raw_dir = Path("data/raw")

    df = read_train_csv(raw_dir / "train.csv")
    mapping = build_label_mapping(df)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(mapping.idx_to_label, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved labels in {out_path}")
