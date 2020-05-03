import os
from argparse import ArgumentParser

import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from net import Net, GAN  # noqa: F401

from datasets import Kalasanty  # noqa: F401


def main(hparams, model_name):
    torch.manual_seed(hparams.seed)
    dataset = Kalasanty(precompute_class_weights=True, fixed_length=True)
    logger = TensorBoardLogger(save_dir=os.getenv("HOME"), name="logs")
    net = Net(hparams, model_name, dataset)
    # net = GAN(hparams, dataset)
    trainer = Trainer.from_argparse_args(
        hparams,
        logger=logger,
        checkpoint_callback=True,
        row_log_interval=10,
        log_save_interval=10,
        val_check_interval=0.5,
        progress_bar_refresh_rate=1,
        gpus=1,
        profiler=True,
        # default_root_dir=os.getenv("HOME"),
        max_epochs=5,
        # fast_dev_run=True,
        overfit_pct=0.01,
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
    parser.add_argument("--batch_size", default=32, type=int)
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
