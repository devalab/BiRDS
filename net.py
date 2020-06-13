from collections import OrderedDict

import pytorch_lightning as pl
import torch
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset

from datasets import scPDB
from metrics import (
    batch_loss,
    batch_metrics,
    detector_margin_loss,
    generator_mse_loss,
    generator_pos_loss,
    pl_weighted_loss,
    weighted_bce_loss,
    weighted_focal_loss,
)
from models import BiLSTM, CGenerator, Detector, ResNet

SMOOTH = 1e-6


class Net(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.num_cpus = 10
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

    def on_after_backward(self):
        # example to inspect gradient information in tensorboard
        if self.trainer.global_step % 200 == 0:  # don't make the tf file huge
            params = self.state_dict()
            for k, v in params.items():
                grads = v
                name = k
                self.logger.experiment.add_histogram(
                    tag=name, values=grads, global_step=self.trainer.global_step
                )

    def train_dataloader(self):
        return DataLoader(
            Subset(self.train_ds, self.train_ds.train_indices),
            # self.train_ds,
            batch_size=self.hparams.batch_size,
            collate_fn=self.train_ds.collate_fn,
            num_workers=self.num_cpus,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            Subset(self.train_ds, self.train_ds.valid_indices),
            # self.test_ds,
            batch_size=self.hparams.batch_size,
            collate_fn=self.train_ds.collate_fn,
            num_workers=self.num_cpus,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.hparams.batch_size,
            collate_fn=self.train_ds.collate_fn,
            num_workers=self.num_cpus,
            shuffle=False,
        )

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.net_lr)
        scheduler = {
            "scheduler": ReduceLROnPlateau(
                optimizer, mode="max", patience=4, verbose=True
            ),
            "monitor": "val_mcc",
        }
        return [optimizer], [scheduler]

    # learning rate warm-up
    def optimizer_step(self, curr_epoch, batch_nb, optim, optim_idx, soc=None):
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
                    y_pred, data["label"], meta["length"], self.loss_func
                )
            }
        )
        for key, val in batch_metrics(
            y_pred,
            data["label"],
            meta["length"],
            logger=self.logger,
            epoch=self.current_epoch,
        ).items():
            metrics[prefix + key] = val
        return metrics

    def validation_step(self, batch, batch_idx):
        return self.val_test_step(batch, batch_idx, "val_")

    def validation_epoch_end(self, outputs):
        avg_metrics = OrderedDict(
            {key: torch.stack([el[key] for el in outputs]).mean() for key in outputs[0]}
        )
        return {**avg_metrics, "progress_bar": avg_metrics, "log": avg_metrics}

    def test_step(self, batch, batch_idx):
        return self.val_test_step(batch, batch_idx, "test_")

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
            default=0.08,
            type=float,
            help="Main Net Learning Rate. Default: %(default)f",
        )
        parser.add_argument(
            "--dropout",
            type=float,
            default=0.2,
            help="Dropout to be applied between layers. Default: %(default)f",
        )
        parser.add_argument("--loss", type=str, default="bce", choices=["bce", "focal"])
        return parser


class CGAN(pl.LightningModule):
    def __init__(self, hparams, net):
        super().__init__()
        self.hparams = hparams
        self.net = net
        self.generator = CGenerator(self.net.embedding_dim, hparams)

    def forward(self, X, lengths):
        # [Batch, hidden_sizes[-1], Max_length]
        output = self.net.embedding_model(X, lengths)
        # Return the embeddings as [Batch, Max_length, hidden_sizes[-1]]
        return output.transpose(1, 2)

    def on_after_backward(self):
        return self.net.on_after_backward()

    def train_dataloader(self):
        return self.net.train_dataloader()

    def val_dataloader(self):
        return self.net.val_dataloader()

    def test_dataloader(self):
        return self.net.test_dataloader()

    def configure_optimizers(self):
        if self.hparams.generator_opt == "SGD":
            opt_g = SGD(self.generator.parameters(), lr=self.hparams.cgan_lr)
        else:
            opt_g = Adam(self.generator.parameters(), lr=self.hparams.cgan_lr)

        if self.hparams.detector_opt == "SGD":
            opt_d = SGD(self.net.detector.parameters(), lr=self.hparams.cgan_lr)
        else:
            opt_d = Adam(self.net.detector.parameters(), lr=self.hparams.cgan_lr)
        return [opt_g, opt_d]

    def training_step(self, batch, batch_idx, optimizer_idx):
        data, meta = batch
        embed = self(data["feature"], meta["length"])
        mask = []
        for i, yt in enumerate(data["label"]):
            mask.append(torch.eq(yt[: meta["length"][i]], 0))

        # Train Generator
        if optimizer_idx == 0:
            generated_embed = self.generator(embed)
            y_fake_pred = self.net.detector(generated_embed)
            g_pos_loss = batch_loss(
                y_fake_pred, data["label"], meta["length"], generator_pos_loss
            )
            if self.hparams.use_mse_loss:
                g_mse_loss = batch_loss(
                    generated_embed,
                    embed,
                    meta["length"],
                    generator_mse_loss,
                    mask=mask,
                )
                g_loss = g_pos_loss + self.hparams.generator_lambda * g_mse_loss
                log_dict = {
                    "g_loss": g_loss,
                    "g_pos_loss": g_pos_loss,
                    "g_mse_loss": g_mse_loss,
                }
            else:
                g_loss = g_pos_loss
                log_dict = {"g_loss": g_loss}
            tqdm_dict = {"g_loss": g_loss}
            return OrderedDict(
                {"loss": g_loss, "progress_bar": tqdm_dict, "log": log_dict}
            )

        # Train detector
        if optimizer_idx == 1:
            y_pred = self.net.detector(embed)
            generated_embed = self.generator(embed).detach()
            net_loss = batch_loss(
                y_pred,
                data["label"],
                meta["length"],
                self.net.loss_func,
                pos_weight=self.net.train_ds.pos_weight,
            )
            if self.hparams.use_margin_loss:
                y_fake_pred = self.net.detector(generated_embed)
                mar_loss = batch_loss(
                    y_fake_pred, data["label"], meta["length"], detector_margin_loss
                )
                d_loss = net_loss + self.hparams.detector_lambda * mar_loss
                log_dict = {
                    "net_loss": net_loss,
                    "mar_loss": mar_loss,
                    "d_loss": d_loss,
                }
            else:
                d_loss = net_loss
                log_dict = {"d_loss": d_loss}
            tqdm_dict = {"d_loss": d_loss}
            return OrderedDict(
                {"loss": d_loss, "progress_bar": tqdm_dict, "log": log_dict}
            )

    def validation_step(self, batch, batch_idx):
        return self.net.val_test_step(batch, batch_idx, "val_")

    def validation_epoch_end(self, outputs):
        return self.net.validation_epoch_end(outputs)

    def test_step(self, batch, batch_idx):
        return self.net.val_test_step(batch, batch_idx, "test_")

    def test_epoch_end(self, outputs):
        return self.net.validation_epoch_end(outputs)

    @staticmethod
    def add_class_specific_args(parser):
        parser.add_argument(
            "--cgan-epochs",
            metavar="EPOCHS",
            default=50,
            type=int,
            help="Default: %(default)d",
        )
        parser.add_argument(
            "--cgan-lr",
            default=0.0002,
            type=float,
            help="CGAN Learning Rate. Default: %(default)f",
        )
        parser = CGenerator.add_class_specific_args(parser)
        parser.add_argument(
            "--generator-opt",
            metavar="OPTIMIZER",
            default="Adam",
            type=str,
            choices=["Adam", "SGD"],
            help="Optimizer to be used by generator",
        )
        parser.add_argument(
            "--detector-opt",
            metavar="OPTIMIZER",
            default="Adam",
            choices=["Adam", "SGD"],
            type=str,
            help="Optimizer to be used by detector",
        )
        parser.add_argument(
            "--generator-lambda", default=1.0, type=float, help="Default: %(default)f"
        )
        parser.add_argument(
            "--detector-lambda", type=float, default=1.0, help="Default: %(default)f"
        )
        parser.add_argument(
            "--no-mse-loss",
            dest="use_mse_loss",
            action="store_false",
            help="Default: %(default)s",
        )
        parser.set_defaults(use_mse_loss=True)
        parser.add_argument(
            "--no-margin-loss",
            dest="use_margin_loss",
            action="store_false",
            help="Default: %(default)s",
        )
        parser.set_defaults(use_margin_loss=True)
        return parser
