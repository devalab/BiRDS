from argparse import Namespace
from collections import OrderedDict

import pytorch_lightning as pl
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import (
    F1,
    ROC,
    Accuracy,
    ConfusionMatrix,
    IoU,
    MatthewsCorrcoef,
    MetricCollection,
    Precision,
    PrecisionRecallCurve,
    Recall,
)
from torchmetrics.functional import auc

from birds.metrics import DCC, batch_work, make_figure, weighted_bce_loss, weighted_focal_loss
from birds.models import Detector, ResNet

SMOOTH = 1e-6


class Net(pl.LightningModule):
    def __init__(self, hparams, input_size, pos_weight=None):
        super().__init__()
        if type(hparams) is dict:
            hparams = Namespace(**hparams)
        # torch.autograd.set_detect_anomaly(True)
        hparams.input_size = input_size
        hparams.pos_weight = pos_weight
        self.save_hyperparameters(hparams)

        metrics = MetricCollection(
            [
                MatthewsCorrcoef(2, threshold=hparams.threshold),
                Accuracy(threshold=hparams.threshold),
                F1(threshold=hparams.threshold),
                IoU(2, threshold=hparams.threshold),
                Precision(threshold=hparams.threshold),
                Recall(threshold=hparams.threshold),
            ]
        )
        self.valid_metrics = metrics.clone(prefix="v_")
        self.test_metrics = metrics.clone(prefix="t_")

        figure_metrics = MetricCollection(
            [
                ConfusionMatrix(2, threshold=hparams.threshold),
                ROC(),
                PrecisionRecallCurve(),
            ]
        )
        self.valid_figure_metrics = figure_metrics.clone(prefix="v_")
        self.test_figure_metrics = figure_metrics.clone(prefix="t_")

        self.model_class = ResNet
        self.embedding_model = self.model_class(input_size, hparams)
        dummy = torch.ones((1, input_size, 10))
        self.embedding_dim = self.embedding_model(dummy, dummy.shape[2], segment_label=dummy[:, 0, :].int()).shape[1]

        self.detector = Detector(self.embedding_dim, hparams)
        if hparams.loss == "focal":
            self.loss_func = weighted_focal_loss
        else:
            self.loss_func = weighted_bce_loss

    def forward(self, X, lengths, **kwargs):
        # [Batch, hidden_sizes[-1], Max_length]
        output = self.embedding_model(X, lengths, **kwargs)

        # [Batch, hidden_sizes[-1], Max_length] -> [Batch, Max_length]
        output = output.transpose(1, 2)
        output = self.detector(output, **kwargs)

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

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = {
            "scheduler": ReduceLROnPlateau(optimizer, mode="max", patience=4, verbose=True),
            "monitor": "v_MatthewsCorrcoef",
        }
        return [optimizer], [scheduler]

    # learning rate warm-up
    def optimizer_step(self, curr_epoch, batch_idx, optim, opt_idx, optimizer_closure, *args, **kwargs):
        # warm up lr
        warm_up_steps = float(17000 // self.hparams.batch_size)
        if self.trainer.global_step < warm_up_steps:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / warm_up_steps)
            for pg in optim.param_groups:
                pg["lr"] = lr_scale * self.hparams.lr

        optim.step(closure=optimizer_closure)

    def training_step(self, batch, batch_idx):
        data, meta = batch
        y_pred = self(data["feature"], meta["length"], segment_label=data["segment_label"])
        y_pred, y_true = batch_work(y_pred, data["label"], meta["length"])
        return self.loss_func(y_pred, y_true, pos_weight=self.hparams.pos_weight)

    def val_test_step(self, batch, calc_metrics, calc_figure_metrics):
        data, meta = batch
        y_pred = self(data["feature"], meta["length"], segment_label=data["segment_label"])
        y_preds, y_true = batch_work(y_pred, data["label"], meta["length"])
        y_pred = torch.sigmoid(y_preds)
        calc_metrics.update(y_pred, y_true.int())
        calc_figure_metrics.update(y_pred, y_true.int())
        return {
            "f_"
            + calc_figure_metrics.prefix
            + "dcc": DCC((y_pred >= self.hparams.threshold).bool(), y_true.bool(), data, meta),
            calc_metrics.prefix + "loss": self.loss_func(y_preds, y_true, pos_weight=[1.0]),
        }

    def val_test_epoch_end(self, outputs, calc_metrics, calc_figure_metrics):
        metrics = OrderedDict(
            {key: torch.stack([el[key] for el in outputs]).mean() for key in outputs[0] if not key.startswith("f_")}
        )
        metrics.update(calc_metrics.compute())
        calc_metrics.reset()
        for key, val in metrics.items():
            self.try_log(key, val, len(outputs))

        figure_metrics = OrderedDict(
            {key[2:]: torch.stack(sum([el[key] for el in outputs], [])) for key in outputs[0] if key.startswith("f_")}
        )
        figure_metrics.update(calc_figure_metrics.compute())
        calc_figure_metrics.reset()
        for key, val in figure_metrics.items():
            self.logger.experiment.add_figure(key, make_figure(key, val), self.current_epoch)
            if key[2:] == "ROC":
                self.try_log("v_auroc", auc(val[0], val[1], reorder=True), len(outputs))
            if key[2:] == "PrecisionRecallCurve":
                self.try_log("v_auprc", auc(val[0], val[1], reorder=True), len(outputs))

    def validation_step(self, batch, batch_idx):
        return self.val_test_step(batch, self.valid_metrics, self.valid_figure_metrics)

    def validation_epoch_end(self, outputs):
        self.val_test_epoch_end(outputs, self.valid_metrics, self.valid_figure_metrics)

    def test_step(self, batch, batch_idx):
        return self.val_test_step(batch, self.test_metrics, self.test_figure_metrics)

    def test_epoch_end(self, outputs):
        self.val_test_epoch_end(outputs, self.test_metrics, self.test_figure_metrics)

    def try_log(self, key, value, batch_size):
        try:
            self.log(key, value, prog_bar=True, batch_size=batch_size)
        except Exception as e:
            print(e)
            return

    @staticmethod
    def add_class_specific_args(parser):
        parser = Detector.add_class_specific_args(parser)
        parser.add_argument(
            "--embedding-model",
            default="resnet",
            type=str,
            choices=["resnet"],
            help="Model to be used for generating embeddings. Default: %(default)s",
        )
        parser.add_argument(
            "--lr",
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
        parser.add_argument(
            "--threshold",
            type=float,
            default=0.5,
            help="Threshold to use for binary classification. Default: %(default)f",
        )
        return parser
