import os
from argparse import ArgumentParser

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TestTubeLogger

from net import GAN, Net  # noqa: F401


class MyModelCheckpoint(ModelCheckpoint):
    def format_checkpoint_name(self, epoch, metrics, ver=None):
        if self.filename == "{epoch}":
            self.filename = "{epoch}-{val_mcc:.3f}"
        return super().format_checkpoint_name(epoch, metrics, ver)


def main(hparams, model_class):
    torch.manual_seed(hparams.seed)
    # Logging to HOME so that all experiments are available for viewing on Cluster
    save_dir = os.getenv("HOME")
    logger = TestTubeLogger(save_dir=save_dir, name="logs", create_git_tag=True)
    checkpoint_callback = MyModelCheckpoint(
        monitor="val_mcc", verbose=True, save_top_k=3, mode="max",
    )
    net = Net(hparams, model_class)
    # net = GAN(hparams, dataset)
    bs = hparams.batch_size
    row_log_interval = 64 / bs
    log_save_interval = 256 / bs
    progress_bar_refresh_rate = 64 / bs
    accumulate_grad_batches = {5: max(1, 16 // bs), 10: 64 // bs}
    gradient_clip_val = 0
    trainer = Trainer.from_argparse_args(
        hparams,
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        row_log_interval=row_log_interval,
        log_save_interval=log_save_interval,
        val_check_interval=0.5,
        progress_bar_refresh_rate=progress_bar_refresh_rate,
        profiler=True,
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_val=gradient_clip_val,
        # fast_dev_run=True,
        # overfit_pct=0.01,
    )
    trainer.fit(net)


if __name__ == "__main__":
    parser = ArgumentParser(description="Binding Site Predictor", add_help=True)
    parser.add_argument("--seed", default=42, type=int, help="Training seed.")
    parser.add_argument(
        "--model",
        default="resnet",
        type=str,
        choices=["resnet", "bilstm", "bigru", "stackedconv", "stackednn", "unet"],
    )
    parser.add_argument("--gpus", default=1, type=int)
    parser.add_argument("--max_epochs", default=200, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--learning_rate", default=0.01, type=float)

    # add model specific args
    tmp_args = parser.parse_known_args()[0]
    if tmp_args.model == "resnet":
        from models import ResNet as model_class
    elif tmp_args.model == "bilstm":
        from models import BiLSTM as model_class
    elif tmp_args.model == "bigru":
        from models import BiGRU as model_class
    elif tmp_args.model == "stackedconv":
        from models import StackedConv as model_class
    elif tmp_args.model == "stackednn":
        from models import StackedNN as model_class
    elif tmp_args.model == "unet":
        from models import UNet as model_class
    parser = model_class.add_model_specific_args(parser)

    hparams = parser.parse_args()
    print(hparams)
    main(hparams, model_class)
