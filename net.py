from collections import OrderedDict

import pytorch_lightning as pl
import torch
from torch.nn.functional import binary_cross_entropy_with_logits, mse_loss
from torch.utils.data import DataLoader, SubsetRandomSampler

from datasets import Chen, Kalasanty, collate_fn
from metrics import batch_loss, batch_metrics
from models import CGenerator, Detector

SMOOTH = 1e-6


class Net(pl.LightningModule):
    def __init__(self, hparams, model_class):
        super().__init__()
        self.hparams = hparams
        self.train_ds = Kalasanty(precompute_class_weights=True, fixed_length=False)
        if hparams.testing:
            self.test_ds = Chen()
        self.embedding_model = model_class(self.train_ds.feat_vec_len, hparams)
        dummy = torch.ones((1, self.train_ds.feat_vec_len, 100))
        self.embedding_dim = self.embedding_model(dummy, dummy.shape[2]).shape[1]

        self.detector = Detector(self.embedding_dim, hparams)
        if hparams.loss == "focal":
            self.loss_func = self.focal_loss
        else:
            self.loss_func = self.weighted_loss
        if hparams.reduction == "sum":
            self.reduction = torch.sum
        else:
            self.reduction = torch.mean

    def forward(self, X, lengths):
        # [Batch, hidden_sizes[-1], Max_length]
        output = self.embedding_model(X, lengths)

        # [Batch, hidden_sizes[-1], Max_length] -> [Batch, Max_length]
        output = output.transpose(1, 2)
        output = self.detector(output)

        return output

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            sampler=SubsetRandomSampler(self.train_ds.train_indices[0]),
            collate_fn=collate_fn,
            num_workers=10,
        )

    def val_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            sampler=SubsetRandomSampler(self.train_ds.valid_indices[0]),
            collate_fn=collate_fn,
            num_workers=10,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.hparams.batch_size,
            collate_fn=collate_fn,
            num_workers=10,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.net_lr)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", patience=5, verbose=True
            ),
            "monitor": "val_mcc",
        }
        return [optimizer], [scheduler]

    # learning rate warm-up
    def optimizer_step(
        self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure=None
    ):
        # warm up lr
        if self.trainer.global_step < 500:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / 500.0)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams.net_lr

        # update params
        optimizer.step()
        optimizer.zero_grad()

    def focal_loss(self, y_pred, y_true, gamma=2.0, alpha=0.25, **kwargs):
        y_pred = torch.sigmoid(y_pred)
        pos = torch.where(torch.eq(y_true, 1), y_pred, torch.ones_like(y_pred))
        neg = torch.where(torch.eq(y_true, 0), y_pred, torch.zeros_like(y_pred))

        pos = torch.clamp(pos, SMOOTH, 1.0 - SMOOTH)
        neg = torch.clamp(neg, SMOOTH, 1.0 - SMOOTH)

        loss = -(alpha * torch.pow(1.0 - pos, gamma) * torch.log(pos)) - (
            (1.0 - alpha) * torch.pow(neg, gamma) * torch.log(1.0 - neg)
        )
        return self.reduction(loss)

    def weighted_loss(self, output, target, **kwargs):
        if self.training:
            pos_weight = self.train_ds.pos_weight
        else:
            pos_weight = [1]
        pos_weight = torch.Tensor(pos_weight).type(target[0].type())
        return binary_cross_entropy_with_logits(
            output, target, pos_weight=pos_weight, reduction=self.hparams.reduction
        )

    def training_step(self, batch, batch_idx):
        X, y, lengths = batch
        y_pred = self(X, lengths)
        loss = batch_loss(y_pred, y, lengths, self.loss_func)
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
        self.train_ds = net.train_ds
        if hparams.testing:
            self.test_ds = net.test_ds
        self.generator = CGenerator(self.net.embedding_dim, hparams)

    def forward(self, X, lengths):
        # [Batch, hidden_sizes[-1], Max_length]
        output = self.net.embedding_model(X, lengths)
        # Return the embeddings as [Batch, Max_length, hidden_sizes[-1]]
        return output.transpose(1, 2)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            sampler=SubsetRandomSampler(self.train_ds.train_indices[0]),
            collate_fn=collate_fn,
            num_workers=10,
        )

    def val_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            sampler=SubsetRandomSampler(self.train_ds.valid_indices[0]),
            collate_fn=collate_fn,
            num_workers=10,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.hparams.batch_size,
            collate_fn=collate_fn,
            num_workers=10,
        )

    def configure_optimizers(self):
        if self.hparams.generator_opt == "SGD":
            opt_g = torch.optim.SGD(
                self.generator.parameters(), lr=self.hparams.cgan_lr
            )
        else:
            opt_g = torch.optim.Adam(
                self.generator.parameters(), lr=self.hparams.cgan_lr
            )

        if self.hparams.detector_opt == "SGD":
            opt_d = torch.optim.SGD(
                self.net.detector.parameters(), lr=self.hparams.cgan_lr
            )
        else:
            opt_d = torch.optim.Adam(
                self.net.detector.parameters(), lr=self.hparams.cgan_lr
            )
        return [opt_g, opt_d]

    def margin_loss(self, yp, yt, batch_idx):
        yp = torch.sigmoid(yp)
        yp = torch.masked_select(yp, self.mask[batch_idx])
        yp = torch.clamp(yp, SMOOTH, 1.0 - SMOOTH)
        loss = -(yp * torch.log(yp)) - ((1 - yp) * torch.log(1 - yp))
        return self.net.reduction(loss)

    def g_pos_loss(self, yp, yt, batch_idx):
        yp = torch.sigmoid(yp)
        yp = torch.masked_select(yp, self.mask[batch_idx])
        yp = torch.clamp(yp, SMOOTH, 1.0 - SMOOTH)
        loss = -torch.log(yp)
        return self.net.reduction(loss)

    def g_mse_loss(self, generated_embed, embed, batch_idx):
        embed = torch.masked_select(embed, self.mask[batch_idx].unsqueeze(1))
        generated_embed = torch.masked_select(
            generated_embed, self.mask[batch_idx].unsqueeze(1)
        )
        return mse_loss(generated_embed, embed, reduction=self.hparams.reduction)

    def training_step(self, batch, batch_idx, optimizer_idx):
        X, y, lengths = batch
        self.mask = []
        for i, yt in enumerate(y):
            self.mask.append(torch.eq(yt[: lengths[i]], 0))

        # Train Generator
        if optimizer_idx == 0:
            self.embed = self(X, lengths)
            generated_embed = self.generator(self.embed)
            y_fake_pred = self.net.detector(generated_embed)
            g_pos_loss = batch_loss(y_fake_pred, y, lengths, self.g_pos_loss)
            g_mse_loss = batch_loss(
                generated_embed, self.embed, lengths, self.g_mse_loss
            )
            g_loss = 0.05 * g_pos_loss + g_mse_loss
            tqdm_dict = {"g_loss": g_loss}
            log_dict = {
                "g_pos_loss": g_pos_loss,
                "g_mse_loss": g_mse_loss,
                "g_loss": g_loss,
            }
            return OrderedDict(
                {"loss": g_loss, "progress_bar": tqdm_dict, "log": log_dict}
            )

        # Train detector
        if optimizer_idx == 1:
            y_pred = self.net.detector(self.embed.detach())
            generated_embed = self.generator(self.embed)
            y_fake_pred = self.net.detector(generated_embed)
            net_loss = batch_loss(y_pred, y, lengths, self.net.loss_func)
            mar_loss = batch_loss(y_fake_pred, y, lengths, self.margin_loss)
            d_loss = net_loss + 0.05 * mar_loss
            tqdm_dict = {"d_loss": d_loss}
            log_dict = {"net_loss": net_loss, "mar_loss": mar_loss, "d_loss": d_loss}
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
        return parser
