from collections import OrderedDict

import pytorch_lightning as pl
import torch
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset

from datasets import Chen, Kalasanty, KalasantyDict, collate_fn, collate_fn_dict
from metrics import (
    batch_loss,
    batch_metrics,
    detector_margin_loss,
    generator_mse_loss,
    generator_pos_loss,
    weighted_bce_loss,
    weighted_focal_loss,
)
from models import BiLSTM, CGenerator, Detector, ResNet

SMOOTH = 1e-6


class Net(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.train_ds = Kalasanty(precompute_class_weights=True, fixed_length=False)
        self.collate_fn = collate_fn
        if hparams.testing:
            self.test_ds = Chen()

        if hparams.embedding_model == "bilstm":
            self.model_class = BiLSTM
        else:
            self.model_class = ResNet
        self.embedding_model = self.model_class(self.train_ds.feat_vec_len, hparams)
        dummy = torch.ones((1, self.train_ds.feat_vec_len, 100))
        self.embedding_dim = self.embedding_model(dummy, dummy.shape[2]).shape[1]

        self.detector = Detector(self.embedding_dim, hparams)
        if hparams.loss == "focal":
            self.loss_func = weighted_focal_loss
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
            Subset(self.train_ds, self.train_ds.train_indices[0]),
            batch_size=self.hparams.batch_size,
            collate_fn=self.collate_fn,
            num_workers=10,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            Subset(self.train_ds, self.train_ds.valid_indices[0]),
            batch_size=self.hparams.batch_size,
            collate_fn=self.collate_fn,
            num_workers=10,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.hparams.batch_size,
            collate_fn=self.collate_fn,
            num_workers=10,
            shuffle=False,
        )

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.net_lr)
        scheduler = {
            "scheduler": ReduceLROnPlateau(
                optimizer, mode="max", patience=5, verbose=True
            ),
            "monitor": "val_mcc",
        }
        return [optimizer], [scheduler]

    # learning rate warm-up
    def optimizer_step(self, curr_epoch, batch_nb, optim, optim_idx, soc=None):
        # warm up lr
        if self.trainer.global_step < 500:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / 500.0)
            for pg in optim.param_groups:
                pg["lr"] = lr_scale * self.hparams.net_lr

        optim.step()
        optim.zero_grad()

    def training_step(self, batch, batch_idx):
        X, y, lengths = batch
        y_pred = self(X, lengths)
        loss = batch_loss(
            y_pred, y, lengths, self.loss_func, pos_weight=self.train_ds.pos_weight
        )
        return {"loss": loss, "log": {"loss": loss}}

    def val_test_step(self, batch, batch_idx, prefix):
        X, y, lengths = batch
        y_pred = self(X, lengths)
        metrics = OrderedDict(
            {prefix + "loss": batch_loss(y_pred, y, lengths, self.loss_func)}
        )
        for key, val in batch_metrics(y_pred, y, lengths).items():
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
    def add_model_specific_args(parser):
        parser = Detector.add_model_specific_args(parser)
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
        parser.add_argument("--loss", type=str, default="bce", choices=["bce", "focal"])
        return parser


class NetDict(Net):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.train_ds = KalasantyDict(
            precompute_class_weights=True,
            fixed_length=False,
            use_dist_map=hparams.use_dist_map,
            use_pl_dist=hparams.use_pl_dist,
        )
        self.collate_fn = collate_fn_dict
        if hparams.use_dist_map:
            self.dist_map_model = self.model_class(512, hparams)
            self.detector = Detector(2 * self.embedding_dim, hparams)
        if hparams.use_pl_dist:
            self.loss_func = self.pl_weighted_loss

    def forward(self, X, lengths):
        # [Batch, hidden_sizes[-1], Max_length]
        output = self.embedding_model(X["X"], lengths)
        if self.hparams.use_dist_map:
            out2 = self.dist_map_model(X["dist_map"], lengths)
            output = torch.cat((output, out2), 1)
            # [Batch, 2 * hidden_sizes[-1], Max_length] -> [Batch, Max_length]

        # [Batch, hidden_sizes[-1], Max_length] -> [Batch, Max_length]
        output = output.transpose(1, 2)
        output = self.detector(output)

        return output

    def pl_weighted_loss(
        self, y_pred, y_true, batch_idx, lengths, pl_dist=None, **kwargs
    ):
        y_pred = torch.clamp(torch.sigmoid(y_pred), SMOOTH, 1.0 - SMOOTH)
        if self.training:
            assert pl_dist is not None
            tmp = pl_dist[batch_idx, : lengths[batch_idx]]
            # if lengths[batch_idx] < 75:
            #     print(tmp)
            #     print(y_true)
            #     exit(1)
            loss = -(15 * y_true * torch.log(y_pred)) - (
                (1 - y_true) * torch.log(1.0 - y_pred) * (tmp / 10.0)
            )
        else:
            loss = -(y_true * torch.log(y_pred)) - (
                (1 - y_true) * torch.log(1.0 - y_pred)
            )
        return torch.mean(loss)

    def training_step(self, batch, batch_idx):
        X, y, lengths = batch
        y_pred = self(X, lengths)
        if self.hparams.use_pl_dist:
            loss = batch_loss(
                y_pred, y["y"], lengths, self.loss_func, pl_dist=y["pl_dist"]
            )
        else:
            loss = batch_loss(y_pred, y["y"], lengths, self.loss_func)
        return {"loss": loss, "log": {"loss": loss}}

    def val_test_step(self, batch, batch_idx, prefix):
        X, y, lengths = batch
        y_pred = self(X, lengths)
        metrics = OrderedDict(
            {prefix + "loss": batch_loss(y_pred, y["y"], lengths, self.loss_func)}
        )
        for key, val in batch_metrics(y_pred, y["y"], lengths).items():
            metrics[prefix + key] = val
        return metrics

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument(
            "--use-dist-map",
            dest="use_dist_map",
            action="store_true",
            help="Default: %(default)s",
        )
        parser.set_defaults(use_dist_map=False)
        parser.add_argument(
            "--use-pl-dist",
            dest="use_pl_dist",
            action="store_true",
            help="Default: %(default)s",
        )
        parser.set_defaults(use_pl_dist=False)
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
        X, y, lengths = batch
        mask = []
        for i, yt in enumerate(y):
            mask.append(torch.eq(yt[: lengths[i]], 0))

        # Train Generator
        if optimizer_idx == 0:
            self.embed = self(X, lengths)
            generated_embed = self.generator(self.embed)
            y_fake_pred = self.net.detector(generated_embed)
            g_pos_loss = batch_loss(y_fake_pred, y, lengths, generator_pos_loss)
            if self.hparams.use_mse_loss:
                g_mse_loss = batch_loss(
                    generated_embed, self.embed, lengths, generator_mse_loss, mask=mask
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
            y_pred = self.net.detector(self.embed.detach())
            generated_embed = self.generator(self.embed)
            net_loss = batch_loss(
                y_pred,
                y,
                lengths,
                self.net.loss_func,
                pos_weight=self.net.train_ds.pos_weight,
            )
            if self.hparams.use_margin_loss:
                y_fake_pred = self.net.detector(generated_embed)
                mar_loss = batch_loss(y_fake_pred, y, lengths, detector_margin_loss)
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
    def add_model_specific_args(parser):
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
        parser = CGenerator.add_model_specific_args(parser)
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
