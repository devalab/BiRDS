from argparse import ArgumentParser
import torch
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader, SubsetRandomSampler
import pytorch_lightning as pl
from pytorch_lightning import Trainer

from datasets import collate_fn, Kalasanty
from metrics import batch_loss, batch_metrics
from models import ResNet as model


class Net(pl.LightningModule):
    def __init__(self, hparams, dataset, criterion):
        super().__init__()
        self.hparams = hparams
        self._dataset = dataset
        self._folds = dataset.custom_cv()
        self.train_indices, self.valid_indices = next(self._folds)
        self._criterion = criterion
        self.feat_vec_len = dataset[0][0].shape[0]
        self._model = model(self.feat_vec_len, hparams)

    def forward(self, X, lengths):
        return self._model(X, lengths)

    def training_step(self, batch, batch_idx):
        X, y, lengths = batch
        y_pred = self(X, lengths)
        loss = batch_loss(y_pred, y, lengths, criterion=self._criterion)
        logs = {"train_loss": loss}
        return {"loss": loss, "log": logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def train_dataloader(self):
        return DataLoader(
            self._dataset,
            batch_size=1,
            sampler=SubsetRandomSampler(self.train_indices),
            collate_fn=collate_fn,
        )

    def validation_step(self, batch, batch_idx):
        X, y, lengths = batch
        y_pred = self(X, lengths)
        metrics = batch_metrics(y_pred, y, lengths)
        metrics["val_loss"] = batch_loss(y_pred, y, lengths, criterion=self._criterion)
        return metrics

    def validation_epoch_end(self, outputs):
        for key in outputs[0]:
            avg_metrics = {key: torch.stack([el[key] for el in outputs]).mean()}
        logs = avg_metrics
        return {"val_loss": avg_metrics["val_loss"], "log": logs}

    def val_dataloader(self):
        return DataLoader(
            self._dataset,
            batch_size=1,
            sampler=SubsetRandomSampler(self.valid_indices),
            collate_fn=collate_fn,
        )


def main(hparams) -> None:
    """
    Main training routine
    :param hparams:
    """
    torch.manual_seed(hparams.seed)
    dataset = Kalasanty(precompute_class_weights=True)
    criterion = BCEWithLogitsLoss(pos_weight=torch.Tensor(dataset.pos_weight))

    net = Net(hparams, dataset, criterion)

    trainer = Trainer.from_argparse_args(
        hparams,
        logger=True,
        # checkpoint_callback=True,
        # early_stop_callback=True,
        # fast_dev_run=True,
        gpus=1,
        default_root_dir="outputs/",
        auto_lr_find=True,
    )
    trainer.fit(net)


if __name__ == "__main__":
    parser = ArgumentParser(description="Binding Site Predictor", add_help=True)
    # add PROGRAM level args
    parser.add_argument("--seed", type=int, default=42, help="Training seed.")
    parser.add_argument(
        "--batch_size", default=1, type=int, help="Batch size to be used."
    )
    parser.add_argument("--learning_rate", default=0.01, type=float)

    # add model specific args
    parser = model.add_model_specific_args(parser)

    # add all the available trainer options to argparse
    # ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    parser = Trainer.add_argparse_args(parser)

    hparams = parser.parse_args()

    main(hparams)
