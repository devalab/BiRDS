import os
from argparse import ArgumentParser, Namespace

import pytorch_lightning as pl
from birds.bert import BERT

from birds.datasets import Birds, scPDB
from birds.models import ResNet
from birds.net import Net


def main(hparams):
    pl.seed_everything(hparams.seed)
    logger = pl.loggers.TensorBoardLogger(save_dir=hparams.weights_save_path, name=hparams.exp_name)
    callbacks = [
        pl.callbacks.ModelSummary(),
        pl.callbacks.ModelCheckpoint(
            monitor="v_MatthewsCorrcoef",
            verbose=True,
            save_top_k=3,
            mode="max",
            filename="{epoch:d}-{v_MatthewsCorrcoef:.3f}-{v_Accuracy:.3f}-{v_F1:.3f}",
        ),
        pl.callbacks.StochasticWeightAveraging(),
    ]
    if hparams.enable_progress_bar:
        callbacks.append(pl.callbacks.RichProgressBar(refresh_rate_per_second=2))
    const_params = {
        "max_epochs": hparams.net_epochs,
        "row_log_interval": 2,
        "log_save_interval": 8,
    }
    hparams = Namespace(**vars(hparams), **const_params)
    print(hparams)

    datamodule = Birds(hparams)
    trainer = pl.Trainer.from_argparse_args(
        hparams,
        logger=logger,
        callbacks=callbacks,
        val_check_interval=0.5,
        precision=16,
        profiler="simple",
        accumulate_grad_batches=16,
        # deterministic=True,
        # track_grad_norm=2,
        # fast_dev_run=True,
        # overfit_pct=0.05,
    )
    net = Net(hparams, datamodule.input_size, datamodule.pos_weight)

    trainer.fit(net, ckpt_path=hparams.ckpt_path, datamodule=datamodule)

    if hparams.run_tests:
        trainer.test(ckpt_path="best")


def parse_arguments():
    parser = ArgumentParser(description="Binding Site Predictor", add_help=True)

    # Trainer Args
    trainer_group = parser.add_argument_group("Trainer")
    trainer_group.add_argument("--seed", default=42, type=int, help="Training seed. Default: %(default)d")
    trainer_group.add_argument("--gpus", default=1, type=int, help="Default: %(default)d")
    trainer_group.add_argument(
        "--batch-size",
        metavar="SIZE",
        default=3,
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
        dest="enable_progress_bar",
        action="store_false",
    )
    trainer_group.set_defaults(enable_progress_bar=True)
    trainer_group.add_argument(
        "--test",
        dest="run_tests",
        action="store_true",
        help="Run tests on model. Default: %(default)s",
    )
    trainer_group.add_argument("--no-test", dest="run_tests", action="store_false")
    trainer_group.set_defaults(run_tests=True)
    trainer_group.add_argument(
        "--ckpt-path",
        metavar="PATH",
        default=None,
        type=str,
        help="Resume trainer from the specified checkpoint provided",
    )

    # LightningDataModule Args
    data_group = parser.add_argument_group("LightningDataModule")
    data_group = Birds.add_class_specific_args(data_group)

    # scPDB Dataset Args
    dataset_group = parser.add_argument_group("scPDB Dataset")
    dataset_group = scPDB.add_class_specific_args(dataset_group)

    # LightningModule Args
    net_group = parser.add_argument_group("LightningModule")
    net_group = Net.add_class_specific_args(net_group)

    # ResNet Args
    resnet_group = parser.add_argument_group("ResNet")
    resnet_group = ResNet.add_class_specific_args(resnet_group)

    # Bert Args
    bert_group = parser.add_argument_group("Bert")
    bert_group = BERT.add_class_specific_args(bert_group)

    # Parse as hyperparameters
    hparams = parser.parse_args()
    hparams.weights_save_path = os.path.abspath(os.path.expanduser(hparams.weights_save_path))
    hparams.data_dir = os.path.abspath(os.path.expanduser(hparams.data_dir))
    return hparams


if __name__ == "__main__":
    main(parse_arguments())
