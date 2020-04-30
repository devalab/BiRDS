import torch
from collections import OrderedDict
from torch.nn.functional import binary_cross_entropy
from torch.utils.data import DataLoader, SubsetRandomSampler

from datasets import collate_fn, fl_collate_fn
from metrics import batch_loss, batch_metrics
import pytorch_lightning as pl


class Net(pl.LightningModule):
    def __init__(self, hparams, model_name, dataset):
        super().__init__()
        self.hparams = hparams
        self._dataset = dataset
        self._folds = dataset.custom_cv()
        self.train_indices, self.valid_indices = next(self._folds)
        self.feat_vec_len = dataset[0][0].shape[0]
        self._model = model_name(self.feat_vec_len, hparams)

    def forward(self, X, lengths):
        return self._model(X, lengths)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return [optimizer]

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
        loss = batch_loss(y_pred, y, lengths, self._dataset.pos_weight)
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
        metrics = OrderedDict({"val_loss": batch_loss(y_pred, y, lengths)})
        for key, val in batch_metrics(y_pred, y, lengths).items():
            metrics["val_" + key] = val
        return metrics

    def validation_epoch_end(self, outputs):
        avg_metrics = OrderedDict(
            {key: torch.stack([el[key] for el in outputs]).mean() for key in outputs[0]}
        )
        return {**avg_metrics, "progress_bar": avg_metrics, "log": avg_metrics}


from models import Generator, Discriminator


class GAN(pl.LightningModule):
    def __init__(self, hparams, dataset):
        super().__init__()
        self.hparams = hparams
        self._dataset = dataset
        self._folds = dataset.custom_cv()
        self.train_indices, self.valid_indices = next(self._folds)
        self.feat_vec_len = dataset[0][0].shape[0]
        self.generator = Generator(self.feat_vec_len, hparams)
        self.discriminator = Discriminator()

    def forward(self, X, lengths):
        return self.generator(X)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            self.generator.parameters(), lr=self.hparams.learning_rate
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.hparams.learning_rate
        )
        return opt_g, opt_d

    def train_dataloader(self):
        return DataLoader(
            self._dataset,
            batch_size=1,
            sampler=SubsetRandomSampler(self.train_indices),
            collate_fn=fl_collate_fn,
        )

    def adversarial_loss(self, y_pred, y):
        return binary_cross_entropy(y_pred, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        X, y, lengths = batch
        batch_size = y.shape[0]
        # train generator
        if optimizer_idx == 0:
            self.y_pred = self(X, lengths)
            valid = torch.ones(batch_size, 1).type(y[0][0].type())
            real_loss = batch_loss(self.y_pred, y, lengths, self._dataset.pos_weight)
            # adversarial loss is binary cross-entropy
            fake_loss = self.adversarial_loss(self.discriminator(self.y_pred), valid)
            g_loss = (real_loss + fake_loss) / 2
            tqdm_dict = {"g_loss": g_loss}
            output = OrderedDict(
                {"loss": g_loss, "progress_bar": tqdm_dict, "log": tqdm_dict}
            )
            return output

        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples
            # how well can it label as real?
            valid = torch.ones(batch_size, 1).type(y[0][0].type())
            real_loss = self.adversarial_loss(self.discriminator(y), valid)
            # how well can it label as fake?
            fake = torch.zeros(batch_size, 1).type(y[0][0].type())
            fake_loss = self.adversarial_loss(
                self.discriminator(self.y_pred.detach()), fake
            )
            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            tqdm_dict = {"d_loss": d_loss}
            output = OrderedDict(
                {"loss": d_loss, "progress_bar": tqdm_dict, "log": tqdm_dict}
            )
            return output

    def training_epoch_end(self, outputs):
        avg_loss = {"train_loss": torch.stack([el["loss"] for el in outputs]).mean()}
        return {**avg_loss, "progress_bar": avg_loss, "log": avg_loss}

    def val_dataloader(self):
        return DataLoader(
            self._dataset,
            batch_size=1,
            sampler=SubsetRandomSampler(self.valid_indices),
            collate_fn=fl_collate_fn,
        )

    def validation_step(self, batch, batch_idx):
        X, y, lengths = batch
        y_pred = self(X, lengths)
        metrics = OrderedDict({"val_loss": batch_loss(y_pred, y, lengths)})
        for key, val in batch_metrics(y_pred, y, lengths).items():
            metrics["val_" + key] = val
        return metrics

    def validation_epoch_end(self, outputs):
        avg_metrics = OrderedDict(
            {key: torch.stack([el[key] for el in outputs]).mean() for key in outputs[0]}
        )
        return {**avg_metrics, "progress_bar": avg_metrics, "log": avg_metrics}
