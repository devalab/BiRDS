from argparse import ArgumentParser
from collections import OrderedDict

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler

from datasets import Chen, Kalasanty, collate_fn
from metrics import batch_loss, batch_metrics


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

        self.fc_layers = torch.nn.ModuleList([])
        feat_vec_len = self.embedding_dim
        for unit in hparams.num_units:
            self.fc_layers.append(torch.nn.Linear(feat_vec_len, unit))
            self.fc_layers.append(torch.nn.ReLU())
            self.fc_layers.append(torch.nn.Dropout(hparams.dropout))
            feat_vec_len = unit
        self.fc_layers.append(torch.nn.Linear(hparams.num_units[-1], 1))

    def forward(self, X, lengths):
        # [Batch, hidden_sizes[-1], Max_length]
        output = self.embedding_model(X, lengths)

        # [Batch, hidden_sizes[-1], Max_length] -> [Batch * Max_length, 1]
        output = output.transpose(1, 2).contiguous().view(-1, self.embedding_dim)
        for layer in self.fc_layers:
            output = layer(output)

        # [Batch * Max_length, 1] -> [Batch, Max_length]
        output = output.view(-1, lengths[0])
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
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
                pg["lr"] = lr_scale * self.hparams.learning_rate

        # update params
        optimizer.step()
        optimizer.zero_grad()

    def training_step(self, batch, batch_idx):
        X, y, lengths = batch
        y_pred = self(X, lengths)
        loss = batch_loss(y_pred, y, lengths, self.train_ds.pos_weight)
        return {"loss": loss, "log": {"loss": loss}}

    def val_test_step(self, batch, batch_idx, prefix):
        X, y, lengths = batch
        y_pred = self(X, lengths)
        metrics = OrderedDict({prefix + "loss": batch_loss(y_pred, y, lengths)})
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
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--num_units",
            nargs="+",
            type=int,
            default=[8],
            help="Number of units that the fully connected layer has. Default: 8",
        )
        parser.add_argument(
            "--dropout",
            type=float,
            default=0.2,
            help="Dropout between the fully connected layers. Default 0.2",
        )
        return parser
