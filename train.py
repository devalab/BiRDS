from datetime import datetime
from os import path

import torch
from fire import Fire
from skorch.callbacks import LRScheduler, ProgressBar
from skorch.cli import parse_args
from skorch.dataset import CVSplit

# from skorch.helper import predefined_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import cross_validate, cross_val_score, GridSearchCV

from callbacks import IOU, MyCheckpoint, MyEpochScoring, MCC
from constants import DEVICE, PROJECT_FOLDER
from dataloader import (
    # DeepCSeqSite,
    Kalasanty,
    # PDBbindRefined,
    TupleDataset,
    collate_fn,
    feat_vec_len,
)
from models.resnet_1d import ResNet  # noqa: F401
from models.transformer import Transformer  # noqa: F401
from models.unet_1d import UNet  # noqa: F401
from net import Net
from utils import copy_code

NET_DEFAULTS = {
    "optimizer": torch.optim.Adam,
    "lr": 0.01,
    "max_epochs": 2,
    "batch_size": 4,
    "train_split": CVSplit(cv=0.1, random_state=42),
    "warm_start": False,
    "verbose": 1,
    "device": DEVICE,
    "iterator_train__collate_fn": collate_fn,
    "iterator_valid__collate_fn": collate_fn,
    "dataset": TupleDataset,
}


def initialize_net(**kwargs):
    net_name = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    net_location = path.join(PROJECT_FOLDER, "outputs", net_name)
    callbacks = [
        ProgressBar(),
        MyEpochScoring(scoring=IOU, lower_is_better=False),
        MyEpochScoring(scoring=MCC, lower_is_better=False),
        MyCheckpoint(dirname=path.join(net_location, "latest"), monitor=None),
        LRScheduler(
            policy=ReduceLROnPlateau, monitor="valid_loss", patience=5, verbose=1
        ),
    ]
    monitors = ["best_valid_loss", "IOU_best", "MCC_best"]
    for monitor in monitors:
        callbacks.append(
            MyCheckpoint(
                dirname=path.join(net_location, monitor),
                monitor=lambda net: net.history[-1, monitor],
            )
        )
    net = Net(
        module=ResNet,
        module__feat_vec_len=feat_vec_len,
        # module__resnet_layer="resnet6",
        # module__num_units=64,
        # module__dropout=0.2,
        criterion=torch.nn.BCEWithLogitsLoss,
        callbacks=callbacks,
    )
    parsed_args = parse_args(kwargs, defaults=NET_DEFAULTS)
    net = parsed_args(net)
    net.name = net_name
    net.location = net_location
    return net


def main(**kwargs):
    dataset = Kalasanty()

    net = initialize_net(train_split=CVSplit(cv=dataset.custom_cv()), **kwargs)
    copy_code("./", path.join(net.location, "code"))
    net.fit(dataset, y=None)

    # Grid Search has to be done without validation monitors or score monitors and
    # LRScheduler needs to have a different way of reducing

    # net = initialize_net(train_split=None, **kwargs)
    # copy_code("./", path.join(net.location, "code"))
    # params = [
    #     {"lr": [0.1, 2, 0.01]},
    # ]
    # gs = GridSearchCV(net, params, cv=dataset.custom_cv(), verbose=5)
    # gs.fit(dataset, y=None)
    # print("Results of grid search:")
    # print("Best parameter configuration:", gs.best_params_)
    # print("Achieved MCC score:", gs.best_score_)

    # print("Saving best model to '{}'.".format(net.location))
    # gs.best_estimator_.save_params(f_params=net.location)


if __name__ == "__main__":
    Fire(main)
