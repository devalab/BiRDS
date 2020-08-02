import os
from argparse import ArgumentParser, Namespace

import pytorch_lightning as pl

from pbsp.datasets import scPDB
from pbsp.models import ResNet
from pbsp.net import Net


class MyModelCheckpoint(pl.callbacks.ModelCheckpoint):
    def format_checkpoint_name(self, epoch, metrics, ver=None):
        if self.filename == "{epoch}":
            self.filename = "{epoch}-{v_mcc:.3f}-{v_acc:.3f}-{v_loss:.3f}"
        return super().format_checkpoint_name(epoch, metrics, ver)


def main(hparams):
    pl.seed_everything(hparams.seed)
    logger = pl.loggers.TestTubeLogger(
        save_dir=hparams.weights_save_path, name=hparams.exp_name, create_git_tag=True,
    )
    checkpoint_callback = MyModelCheckpoint(
        monitor="v_mcc", verbose=True, save_top_k=3, mode="max",
    )
    bs = hparams.batch_size
    if hparams.progress_bar_refresh_rate is None:
        hparams.progress_bar_refresh_rate = 64 // bs
    const_params = {
        "row_log_interval": 64 // bs,
        "log_save_interval": 256 // bs,
        "gradient_clip_val": 0,
    }
    hparams = Namespace(**vars(hparams), **const_params)
    if not hparams.resume_from_checkpoint:
        accumulate_grad_batches = {5: max(1, 16 // bs), 10: 64 // bs}
        # accumulate_grad_batches = 1
    else:
        accumulate_grad_batches = 1
    print(hparams)
    trainer = pl.Trainer.from_argparse_args(
        hparams,
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        val_check_interval=0.5,
        # num_sanity_val_steps=60,
        profiler=True,
        accumulate_grad_batches=accumulate_grad_batches,
        deterministic=True,
        weights_summary="full",
        # track_grad_norm=2,
        # fast_dev_run=True,
        # overfit_pct=0.05,
    )
    net = Net(hparams)
    trainer.fit(net)

    if hparams.run_tests:
        trainer.test(ckpt_path="best")


def parse_arguments():
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
        "--weights-save-path",
        metavar="DIR",
        default="~/logs",
        type=str,
        help="Default directory to store the logs and weights. Defalut: %(default)s",
    )
    trainer_group.add_argument(
        "--exp-name",
        default="experiment_0",
        type=str,
        help="Name of the experiment. Each experiment can have multiple versions inside it. Default: %(default)ss",
    )
    trainer_group.add_argument(
        "--net-epochs",
        metavar="EPOCHS",
        default=100,
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
    trainer_group.set_defaults(run_tests=True)
    trainer_group.add_argument(
        "--cgan",
        dest="use_cgan",
        action="store_true",
        help="Train a Complementary GAN after main net training. Default: %(default)s",
    )
    trainer_group.add_argument("--no-cgan", dest="use_cgan", action="store_false")
    trainer_group.set_defaults(use_cgan=False)
    trainer_group.add_argument(
        "--resume-from-checkpoint",
        metavar="PATH",
        default=None,
        type=str,
        help="Resume trainer from the specified checkpoint provided",
    )

    # Dataset Args
    dataset_group = parser.add_argument_group("Dataset")
    dataset_group = scPDB.add_class_specific_args(dataset_group)

    # Lightning Module Args
    net_group = parser.add_argument_group("Net")
    net_group = Net.add_class_specific_args(net_group)

    # ResNet Args
    resnet_group = parser.add_argument_group("ResNet")
    resnet_group = ResNet.add_class_specific_args(resnet_group)

    # Parse as hyperparameters
    hparams = parser.parse_args()
    hparams.weights_save_path = os.path.abspath(
        os.path.expanduser(hparams.weights_save_path)
    )
    hparams.data_dir = os.path.abspath(os.path.expanduser(hparams.data_dir))
    return hparams


if __name__ == "__main__":
    main(parse_arguments())
