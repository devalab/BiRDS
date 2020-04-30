import os
from argparse import ArgumentParser
from collections import OrderedDict

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader, SubsetRandomSampler

from datasets import Kalasanty, collate_fn
from metrics import batch_loss, batch_metrics


class Net(pl.LightningModule):
    def __init__(self, hparams, model_name, dataset):
        super().__init__()
        self.hparams = hparams
        self._dataset = dataset
        self._folds = dataset.custom_cv()
        self.train_indices, self.valid_indices = next(self._folds)
        self.train_criterion = BCEWithLogitsLoss(
            pos_weight=torch.Tensor(dataset.pos_weight)
        )
        self.val_criterion = BCEWithLogitsLoss()
        self.feat_vec_len = dataset[0][0].shape[0]
        self._model = model_name(self.feat_vec_len, hparams)

    def forward(self, X, lengths):
        return self._model(X, lengths)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, verbose=True
        )
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(
            self._dataset,
            batch_size=1,
            sampler=SubsetRandomSampler(self.train_indices),
            collate_fn=collate_fn,
        )

    def training_step(self, batch, batch_idx):
        X, y, lengths = batch
        y_pred = self(X, lengths)
        loss = batch_loss(y_pred, y, lengths, criterion=self.train_criterion)
        return {"loss": loss, "log": {"loss": loss}}

    def training_epoch_end(self, outputs):
        avg_loss = {"train_loss": torch.stack([el["loss"] for el in outputs]).mean()}
        return {**avg_loss, "progress_bar": avg_loss, "log": avg_loss}

    def val_dataloader(self):
        return DataLoader(
            self._dataset,
            batch_size=1,
            sampler=SubsetRandomSampler(self.valid_indices),
            collate_fn=collate_fn,
        )

    def validation_step(self, batch, batch_idx):
        X, y, lengths = batch
        y_pred = self(X, lengths)
        metrics = OrderedDict(
            {"val_loss": batch_loss(y_pred, y, lengths, criterion=self.val_criterion)}
        )
        for key, val in batch_metrics(y_pred, y, lengths).items():
            metrics["val_" + key] = val
        return metrics

    def validation_epoch_end(self, outputs):
        avg_metrics = OrderedDict(
            {key: torch.stack([el[key] for el in outputs]).mean() for key in outputs[0]}
        )
        return {**avg_metrics, "progress_bar": avg_metrics, "log": avg_metrics}


def main(hparams, model_name):
    torch.manual_seed(hparams.seed)
    dataset = Kalasanty(precompute_class_weights=True)
    logger = TensorBoardLogger(save_dir=os.getenv("HOME"), name="logs")
    net = Net(hparams, model_name, dataset)
    trainer = Trainer.from_argparse_args(
        hparams,
        logger=logger,
        checkpoint_callback=False,
        row_log_interval=100,
        log_save_interval=100,
        val_check_interval=0.5,
        progress_bar_refresh_rate=25,
        gpus=1,
        profiler=True,
        # default_root_dir=os.getenv("HOME"),
        max_epochs=50,
        # fast_dev_run=True,
        # overfit_pct=0.02,
    )
    trainer.fit(net)


if __name__ == "__main__":
    parser = ArgumentParser(description="Binding Site Predictor", add_help=True)
    parser.add_argument("--seed", type=int, default=42, help="Training seed.")
    parser.add_argument(
        "--model",
        type=str,
        default="resnet",
        choices=["resnet", "bilstm", "bigru", "stackedconv", "stackednn", "unet"],
    )
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--learning_rate", default=0.01, type=float)

    # add model specific args
    tmp_args = parser.parse_known_args()[0]
    print(tmp_args)
    if tmp_args.model == "resnet":
        from models import ResNet as model_name
    elif tmp_args.model == "bilstm":
        from models import BiLSTM as model_name
    elif tmp_args.model == "bigru":
        from models import BiGRU as model_name
    elif tmp_args.model == "stackedconv":
        from models import StackedConv as model_name
    elif tmp_args.model == "stackednn":
        from models import StackedNN as model_name
    elif tmp_args.model == "unet":
        from models import UNet as model_name
    parser = model_name.add_model_specific_args(parser)

    # add all the available trainer options to argparse

    hparams = parser.parse_args()

    main(hparams, model_name)
