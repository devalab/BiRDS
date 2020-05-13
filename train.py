import os
from argparse import ArgumentParser

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TestTubeLogger

from models import BiLSTM, ResNet
from net import CGAN, Net


class MyModelCheckpoint(ModelCheckpoint):
    def format_checkpoint_name(self, epoch, metrics, ver=None):
        if self.filename == "{epoch}":
            self.filename = "{epoch}-{val_mcc:.3f}-{val_acc:.3f}-{val_f1:.3f}"
        return super().format_checkpoint_name(epoch, metrics, ver)


def main(hparams):
    torch.manual_seed(hparams.seed)
    # Logging to HOME so that all experiments are available for viewing on Cluster
    save_dir = os.getenv("HOME")
    logger = TestTubeLogger(save_dir=save_dir, name="logs", create_git_tag=True)
    checkpoint_callback = MyModelCheckpoint(
        monitor="val_mcc", verbose=True, save_top_k=3, mode="max",
    )
    bs = hparams.batch_size
    row_log_interval = 64 / bs
    log_save_interval = 256 / bs
    if hparams.progress_bar_refresh_rate is None:
        hparams.progress_bar_refresh_rate = 64 / bs
    accumulate_grad_batches = {5: max(1, 16 // bs), 10: 64 // bs}
    gradient_clip_val = 0
    trainer = Trainer.from_argparse_args(
        hparams,
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        row_log_interval=row_log_interval,
        log_save_interval=log_save_interval,
        val_check_interval=0.5,
        profiler=True,
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_val=gradient_clip_val,
        # fast_dev_run=True,
        # overfit_pct=0.05,
    )

    if hparams.embedding_model == "bilstm":
        model_class = BiLSTM
    else:
        model_class = ResNet

    if hparams.load_cpkt is None:
        net = Net(hparams, model_class)
    else:
        net = Net.load_checkpoint(hparams.load_cpkt, model_class=model_class)

    trainer.fit(net)

    # TODO Load the best model here

    if hparams.cgan:
        cgan = CGAN(hparams, net)
        trainer.max_epochs = hparams.max_epochs + hparams.cgan_epochs
        trainer.fit(cgan)

    if hparams.testing:
        trainer.test()


if __name__ == "__main__":
    parser = ArgumentParser(description="Binding Site Predictor", add_help=True)

    # Trainer Args
    trainer_group = parser.add_argument_group("Trainer")
    trainer_group.add_argument(
        "--seed", default=42, type=int, help="Training seed. Default: %(default)d"
    )
    trainer_group.add_argument(
        "--gpus", default=1, type=int, help="Default: %(default)d"
    )
    trainer_group.add_argument(
        "--max-epochs",
        metavar="EPOCHS",
        default=30,
        type=int,
        help="Main Net epochs. Default: %(default)d",
    )
    trainer_group.add_argument(
        "--no-progress-bar",
        dest="progress_bar_refresh_rate",
        action="store_const",
        const=0,
    )
    trainer_group.add_argument(
        "--testing", dest="testing", action="store_true", help="Default: %(default)s"
    )
    trainer_group.add_argument("--no-testing", dest="testing", action="store_false")
    trainer_group.set_defaults(testing=False)
    trainer_group.add_argument(
        "--cgan", dest="cgan", action="store_true", help="Default: %(default)s"
    )
    trainer_group.add_argument("--no-cgan", dest="cgan", action="store_false")
    trainer_group.set_defaults(cgan=True)

    # Lightning Module Args
    net_group = parser.add_argument_group("Net")
    net_group.add_argument(
        "--load-cpkt",
        metavar="PATH",
        default=None,
        type=str,
        help="Load model from file path provided",
    )
    net_group.add_argument(
        "--batch-size",
        metavar="SIZE",
        default=32,
        type=int,
        help="Default: %(default)d",
    )
    net_group.add_argument(
        "--net-lr",
        default=0.01,
        type=float,
        help="Main Net Learning Rate. Default: %(default)f",
    )
    net_group.add_argument(
        "--reduction",
        default="mean",
        type=str,
        choices=["mean", "sum"],
        help="The type of reduction to use in the loss function. Default: %(default)s",
    )
    net_group = Net.add_model_specific_args(net_group)
    net_group.add_argument(
        "--embedding-model",
        default="resnet",
        type=str,
        choices=["resnet", "bilstm"],
        help="Model to be used for generating embeddings. Default: %(default)s",
    )

    # ResNet Args
    resnet_group = parser.add_argument_group("ResNet")
    resnet_group = ResNet.add_model_specific_args(resnet_group)

    # BiLSTM Args
    # bilstm_group = parser.add_argument_group("BiLSTM")
    # bilstm_group = BiLSTM.add_model_specific_args(bilstm_group)

    # CGAN Arguments
    cgan_group = parser.add_argument_group("CGAN")
    cgan_group = CGAN.add_model_specific_args(cgan_group)

    # Parse as hyperparameters
    hparams = parser.parse_args()
    print(hparams)
    main(hparams)
