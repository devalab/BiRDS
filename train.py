import os
from argparse import ArgumentParser, Namespace

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TestTubeLogger

from datasets import Kalasanty
from models import BiLSTM, ResNet  # noqa: F401
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
    if hparams.progress_bar_refresh_rate is None:
        hparams.progress_bar_refresh_rate = 64 // bs
    const_params = {
        "max_epochs": hparams.net_epochs,
        "row_log_interval": 64 // bs,
        "log_save_interval": 256 // bs,
        "gradient_clip_val": 0,
    }
    hparams = Namespace(**vars(hparams), **const_params)
    if not hparams.resume_from_checkpoint:
        accumulate_grad_batches = {5: max(1, 16 // bs), 10: 64 // bs}
    else:
        accumulate_grad_batches = 1
    print(hparams)
    trainer = Trainer.from_argparse_args(
        hparams,
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        # val_check_interval=0.5,
        profiler=True,
        accumulate_grad_batches=accumulate_grad_batches,
        # track_grad_norm=2,
        # fast_dev_run=True,
        # overfit_pct=0.05,
    )

    if hparams.load_from_checkpoint is None:
        net = Net(hparams)
    else:
        net = Net.load_from_checkpoint(hparams.load_from_checkpoint)

    trainer.fit(net)

    # TODO Load the best model here

    if hparams.use_cgan:
        cgan = CGAN(hparams, net)
        trainer.max_epochs = hparams.net_epochs + hparams.cgan_epochs
        trainer.fit(cgan)

    if hparams.run_tests:
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
        "--batch-size",
        metavar="SIZE",
        default=32,
        type=int,
        help="Default: %(default)d",
    )
    trainer_group.add_argument(
        "--net-epochs",
        metavar="EPOCHS",
        default=50,
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
        "--test",
        dest="run_tests",
        action="store_true",
        help="Run tests on model. Default: %(default)s",
    )
    trainer_group.add_argument("--no-test", dest="run_tests", action="store_false")
    trainer_group.set_defaults(run_tests=False)
    trainer_group.add_argument(
        "--cgan",
        dest="use_cgan",
        action="store_true",
        help="Train a Complementary GAN after main net training. Default: %(default)s",
    )
    trainer_group.add_argument("--no-cgan", dest="use_cgan", action="store_false")
    trainer_group.set_defaults(use_cgan=False)
    trainer_group.add_argument(
        "--load-from-checkpoint",
        metavar="PATH",
        default=None,
        type=str,
        help="Load model from file path provided",
    )
    trainer_group.add_argument(
        "--resume-from-checkpoint",
        metavar="PATH",
        default=None,
        type=str,
        help="Resume trainer from the specified checkpoint provided",
    )

    # Dataset Args
    dataset_group = parser.add_argument_group("Dataset")
    dataset_group = Kalasanty.add_class_specific_args(dataset_group)

    # Lightning Module Args
    net_group = parser.add_argument_group("Net")
    net_group = Net.add_class_specific_args(net_group)

    # ResNet Args
    resnet_group = parser.add_argument_group("ResNet")
    resnet_group = ResNet.add_class_specific_args(resnet_group)

    # BiLSTM Args
    # bilstm_group = parser.add_argument_group("BiLSTM")
    # bilstm_group = BiLSTM.add_class_specific_args(bilstm_group)

    # CGAN Arguments
    cgan_group = parser.add_argument_group("CGAN")
    cgan_group = CGAN.add_class_specific_args(cgan_group)

    # Parse as hyperparameters
    hparams = parser.parse_args()
    main(hparams)
