from pathlib import Path

from paddy_disease.training.train import TrainConfig, train_main


def main() -> None:
    # smoke test трейнинга
    cfg = TrainConfig(raw_dir=Path("data/raw"), epochs=1, batch_size=8, num_workers=0)
    train_main(cfg)


if __name__ == "__main__":
    main()
