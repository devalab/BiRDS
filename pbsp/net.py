from collections import OrderedDict

import pytorch_lightning as pl
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset

from pbsp.datasets import scPDB
from pbsp.metrics import (
    batch_loss,
    batch_metrics,
    make_figure,
    pl_weighted_loss,
    weighted_bce_loss,
    weighted_focal_loss,
)
from pbsp.models import BiLSTM, Detector, ResNet

SMOOTH = 1e-6


class Net(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.num_cpus = 10
        if hparams.gpus != 0:
            self.pin_memory = True
        else:
            self.pin_memory = False
        self.train_ds = scPDB(hparams)
        if hparams.run_tests:
            self.test_ds = scPDB(hparams, test=True)

        if hparams.embedding_model == "bilstm":
            self.model_class = BiLSTM
        else:
            self.model_class = ResNet
        self.embedding_model = self.model_class(self.train_ds.input_size, hparams)
        dummy = torch.ones((1, self.train_ds.input_size, 10))
        self.embedding_dim = self.embedding_model(dummy, dummy.shape[2]).shape[1]

        self.detector = Detector(self.embedding_dim, hparams)
        if hparams.loss == "focal":
            self.loss_func = weighted_focal_loss
        elif hparams.loss == "pl":
            self.loss_func = pl_weighted_loss
        else:
            self.loss_func = weighted_bce_loss

    def forward(self, X, lengths):
        # [Batch, hidden_sizes[-1], Max_length]
        output = self.embedding_model(X, lengths)

        # [Batch, hidden_sizes[-1], Max_length] -> [Batch, Max_length]
        output = output.transpose(1, 2)
        output = self.detector(output)

        return output

    # def on_after_backward(self):
    #     # example to inspect gradient information in tensorboard
    #     if self.trainer.global_step % 200 == 0:  # don't make the tf file huge
    #         params = self.state_dict()
    #         for k, v in params.items():
    #             grads = v
    #             name = k
    #             self.logger.experiment.add_histogram(
    #                 tag=name, values=grads, global_step=self.trainer.global_step
    #             )

    def train_dataloader(self):
        return DataLoader(
            Subset(self.train_ds, self.train_ds.train_indices),
            # self.train_ds,
            batch_size=self.hparams.batch_size,
            collate_fn=self.train_ds.collate_fn,
            num_workers=self.num_cpus,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            Subset(self.train_ds, self.train_ds.valid_indices),
            # self.test_ds,
            batch_size=self.hparams.batch_size,
            collate_fn=self.train_ds.collate_fn,
            num_workers=self.num_cpus,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.hparams.batch_size,
            collate_fn=self.train_ds.collate_fn,
            num_workers=self.num_cpus,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.net_lr)
        scheduler = {
            "scheduler": ReduceLROnPlateau(
                optimizer, mode="max", patience=4, verbose=True
            ),
            "monitor": "v_mcc",
        }
        return [optimizer], [scheduler]

    # learning rate warm-up
    def optimizer_step(self, curr_epoch, batch_idx, optim, opt_idx, *args, **kwargs):
        # warm up lr
        warm_up_steps = float(16384 // self.hparams.batch_size)
        if self.trainer.global_step < warm_up_steps:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / warm_up_steps)
            for pg in optim.param_groups:
                pg["lr"] = lr_scale * self.hparams.net_lr

        optim.step()
        optim.zero_grad()

    def training_step(self, batch, batch_idx):
        data, meta = batch
        y_pred = self(data["feature"], meta["length"])
        loss = batch_loss(
            y_pred,
            data["label"],
            meta["length"],
            self.loss_func,
            pos_weight=self.train_ds.pos_weight,
        )
        return {"loss": loss, "log": {"loss": loss}}

    def val_test_step(self, batch, batch_idx, prefix):
        data, meta = batch
        y_pred = self(data["feature"], meta["length"])
        metrics = OrderedDict(
            {
                prefix
                + "loss": batch_loss(
                    y_pred,
                    data["label"],
                    meta["length"],
                    self.loss_func,
                    pos_weight=[1.0],
                )
            }
        )
        for key, val in batch_metrics(
            y_pred,
            data,
            meta,
            # logger=self.logger,
            # epoch=self.current_epoch,
        ).items():
            if key.startswith("f_"):
                metrics["f_" + prefix + key[2:]] = val
            else:
                metrics[prefix + key] = val
        return metrics

    def validation_step(self, batch, batch_idx):
        return self.val_test_step(batch, batch_idx, "v_")

    def validation_epoch_end(self, outputs):
        avg_metrics = OrderedDict(
            {
                key: torch.stack([el[key] for el in outputs]).mean()
                for key in outputs[0]
                if not key.startswith("f_")
            }
        )
        figure_metrics = OrderedDict(
            {
                key[2:]: torch.stack(sum([el[key] for el in outputs], [])).cpu().numpy()
                for key in outputs[0]
                if key.startswith("f_")
            }
        )
        for key, val in figure_metrics.items():
            self.logger.experiment.add_figure(
                key, make_figure(key[2:], val), self.current_epoch
            )
        return {**avg_metrics, "progress_bar": avg_metrics, "log": avg_metrics}

    def test_step(self, batch, batch_idx):
        return self.val_test_step(batch, batch_idx, "t_")

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)

    @staticmethod
    def add_class_specific_args(parser):
        parser = Detector.add_class_specific_args(parser)
        parser.add_argument(
            "--embedding-model",
            default="resnet",
            type=str,
            choices=["resnet", "bilstm"],
            help="Model to be used for generating embeddings. Default: %(default)s",
        )
        parser.add_argument(
            "--net-lr",
            default=0.01,
            type=float,
            help="Main Net Learning Rate. Default: %(default)f",
        )
        parser.add_argument(
            "--dropout",
            type=float,
            default=0.2,
            help="Dropout to be applied between layers. Default: %(default)f",
        )
        parser.add_argument(
            "--loss",
            type=str,
            default="bce",
            choices=["bce", "focal"],
            help="Loss function to use. Default: %(default)s",
        )
        return parser
