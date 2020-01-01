from os import path
from datetime import datetime

import torch
from fire import Fire
from skorch.callbacks import LRScheduler, ProgressBar
from skorch.cli import parse_args
from skorch.dataset import CVSplit
from torch.optim.lr_scheduler import ReduceLROnPlateau

from callbacks import IOU, MyCheckpoint, MyEpochScoring
from constants import DEVICE, PROJECT_FOLDER
from dataloader import PDBbind, PDBbind_collate_fn
from models.resnet_1d import ResNet  # noqa: F401
from models.transformer import Transformer  # noqa: F401
from net import Net
from utils import copy_code

NET_DEFAULTS = {
    "optimizer": torch.optim.Adam,
    "lr": 0.01,
    "max_epochs": 100,
    "batch_size": 1,
    "train_split": CVSplit(10, random_state=42),
    "warm_start": False,
    "verbose": 1,
    "device": DEVICE,
    "iterator_train__collate_fn": PDBbind_collate_fn,
    "iterator_valid__collate_fn": PDBbind_collate_fn,
}


def initialize_net(**kwargs):
    net_name = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    net_location = path.join(PROJECT_FOLDER, "outputs", net_name)
    callbacks = [
        ProgressBar(),
        MyEpochScoring(scoring=IOU, lower_is_better=False),
        MyCheckpoint(dirname=path.join(net_location, "latest"), monitor=None),
        LRScheduler(
            policy=ReduceLROnPlateau, monitor="valid_loss", patience=5, verbose=1
        ),
    ]
    monitors = ["best_valid_loss", "IOU_best"]
    for monitor in monitors:
        callbacks.append(
            MyCheckpoint(
                dirname=path.join(net_location, monitor),
                monitor=lambda net: net.history[-1, monitor],
            )
        )
    net = Net(
        module=ResNet,
        module__resnet_layer="resnet6",
        module__num_units=64,
        module__dropout=0.2,
        criterion=torch.nn.BCEWithLogitsLoss,
        callbacks=callbacks,
    )
    parsed_args = parse_args(kwargs, defaults=NET_DEFAULTS)
    net = parsed_args(net)
    net.name = net_name
    net.location = net_location
    return net


def initialize_dataset():
    data_needed = ["pdb_id", "sequence", "length", "labels"]
    dataset = PDBbind(
        path.join(PROJECT_FOLDER, "data/PDBbind/preprocessed/unique2"), data_needed
    )
    return dataset


def main(**kwargs):
    dataset = initialize_dataset()
    net = initialize_net(**kwargs)
    copy_code("./", path.join(net.location, "code"))
    net.fit(dataset, y=None)


if __name__ == "__main__":
    Fire(main)
