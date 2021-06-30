import pytorch_lightning as pl
from functools import partial
from pytorch_lightning.callbacks import Callback
from ray import tune
from ray.tune import CLIReporter
from birds.train import parse_arguments, MyModelCheckpoint
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining, FIFOScheduler
from argparse import Namespace
from birds.net import Net


class TuneReportCallback(Callback):
    def on_validation_end(self, trainer, pl_module):
        tune.report(
            v_mcc=trainer.callback_metrics["v_mcc"].item(),
            v_acc=trainer.callback_metrics["v_acc"].item(),
            v_loss=trainer.callback_metrics["v_loss"].item(),
        )


def main(config, checkpoint=None, hparams=None):
    pl.seed_everything(hparams.seed)
    hparams.weights_save_path = tune.get_trial_dir()
    logger = pl.loggers.TestTubeLogger(
        save_dir=hparams.weights_save_path, name="", version="."
    )
    checkpoint_callback = MyModelCheckpoint(
        monitor="v_mcc", verbose=True, save_top_k=3, mode="max",
    )
    bs = hparams.batch_size
    hparams.progress_bar_refresh_rate = 0
    const_params = {
        "row_log_interval": 64 // bs,
        "log_save_interval": 256 // bs,
        "gradient_clip_val": 0,
    }
    hparams = vars(hparams)
    hparams.update(config)
    hparams = Namespace(**hparams, **const_params)
    accumulate_grad_batches = {5: max(1, 16 // bs), 10: 64 // bs}
    print(hparams)
    if checkpoint:
        trainer = pl.Trainer(resume_from_checkpoint=checkpoint)
    else:
        trainer = pl.Trainer.from_argparse_args(
            hparams,
            logger=logger,
            checkpoint_callback=checkpoint_callback,
            val_check_interval=0.5,
            # num_sanity_val_steps=60,
            profiler=True,
            accumulate_grad_batches=accumulate_grad_batches,
            deterministic=True,
            callbacks=[TuneReportCallback()],
            # track_grad_norm=2,
            # fast_dev_run=True,
            # overfit_pct=0.05,
        )
    net = Net(hparams)
    trainer.fit(net)
    if hparams.run_tests:
        trainer.test(ckpt_path="best")


def tune_net(config, scheduler, hparams):
    reporter = CLIReporter(
        parameter_columns=list(config.keys()),
        metric_columns=["v_mcc", "v_acc", "v_loss", "training_iteration"],
    )

    tune.run(
        partial(main, hparams=hparams),
        resources_per_trial={"cpu": 10, "gpu": 1},
        config=config,
        num_samples=1,
        scheduler=scheduler,
        # local_dir="./results",
        progress_reporter=reporter,
        # queue_trials=True,
        name="tune_nla",
    )


if __name__ == "__main__":
    hparams = parse_arguments()
    config = {
        "kernel_sizes": tune.choice([[3, 3], [5, 5], [7, 7]]),
        "layers": tune.choice(
            [[2, 2, 2, 2, 2], [2, 3, 4, 3, 2], [1, 2, 2, 2, 1], [2, 4, 8, 4, 2]]
        ),
        "hidden_sizes": tune.choice([[128, 64, 32, 16], [128, 256, 128, 64]]),
        # "net_lr": tune.loguniform(1e-4, 1e-1),
        # "net_lr": 0.01,
    }
    # scheduler = PopulationBasedTraining(
    #     time_attr="training_iteration",
    #     metric="v_mcc",
    #     mode="max",
    #     perturbation_interval=8,
    #     hyperparam_mutations={"lr": lambda: tune.loguniform(1e-4, 1e-1).func(None)},
    # )
    scheduler = ASHAScheduler(
        metric="v_mcc",
        mode="max",
        max_t=hparams.max_epochs,
        grace_period=10,
        reduction_factor=2,
    )
    # scheduler = FIFOScheduler()
    tune_net(config, scheduler, hparams)
