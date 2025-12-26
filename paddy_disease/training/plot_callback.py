from pathlib import Path

import matplotlib.pyplot as plt
import pytorch_lightning as pl


class SavePlotsCallback(pl.Callback):
    def __init__(self, out_dir: Path) -> None:
        super().__init__()
        self.out_dir = out_dir
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
        }

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        metrics = trainer.callback_metrics
        if "train_loss" in metrics:
            self.history["train_loss"].append(float(metrics["train_loss"].cpu()))
        if "val_loss" in metrics:
            self.history["val_loss"].append(float(metrics["val_loss"].cpu()))
        if "val_acc" in metrics:
            self.history["val_acc"].append(float(metrics["val_acc"].cpu()))

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)

        for key, values in self.history.items():
            if not values:
                continue
            plt.figure()
            plt.plot(values)
            plt.title(key)
            plt.xlabel("epoch")
            plt.ylabel(key)
            out_path = self.out_dir / f"{key.replace('/', '_')}.png"
            plt.savefig(out_path)
            plt.close()
