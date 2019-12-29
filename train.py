import sys
import torch
from net import Net
from models.resnet_1d import ResNet
from skorch.callbacks import ProgressBar, LRScheduler
from callbacks import IOU, MyEpochScoring, MyCheckpoint
from skorch.dataset import CVSplit
from torch.optim.lr_scheduler import ReduceLROnPlateau

from constants import DEVICE, PROJECT_FOLDER
from datetime import datetime
from data import PDBbind_collate_fn, PDBbind
from utils import copy_code

torch.manual_seed(42)


def initialize_net():
    net_name = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    net_location = PROJECT_FOLDER + "outputs/" + net_name + "/"
    callbacks = [
        ProgressBar(),
        MyEpochScoring(scoring=IOU, lower_is_better=False),
        MyCheckpoint(
            dirname=net_location + "best_valid_loss/", monitor="valid_loss_best"
        ),
        MyCheckpoint(dirname=net_location + "latest/", monitor=None),
        # MyCheckpoint(
        #     dirname=net_location + "best_iou/",
        #     monitor=lambda net: net.history[-1, "iou_best"],
        # ),
        LRScheduler(policy=ReduceLROnPlateau, monitor="valid_loss", patience=5),
    ]
    net = Net(
        module=ResNet,
        module__resnet_layer="resnet6",
        module__num_units=64,
        module__dropout=0.2,
        criterion=torch.nn.BCEWithLogitsLoss,
        optimizer=torch.optim.Adam,
        lr=0.01,
        max_epochs=100,
        batch_size=1,
        train_split=CVSplit(10, random_state=42),
        callbacks=callbacks,
        warm_start=True,
        verbose=1,
        device=DEVICE,
        iterator_train__collate_fn=PDBbind_collate_fn,
        iterator_valid__collate_fn=PDBbind_collate_fn,
    )
    net.name = net_name
    net.location = net_location
    return net


def initialize_dataset():
    data_needed = ["pdb_id", "sequence", "length", "labels"]
    dataset = PDBbind(PROJECT_FOLDER + "data/PDBbind/preprocessed/unique/", data_needed)
    return dataset


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Please provide a description for the model")
        print("Eg: python main.py 'Resnet-18 with batch size 4'")
        exit(1)

    # TODO Log the description

    net = initialize_net()
    dataset = initialize_dataset()
    copy_code("./", net.location + "code/")

    net.fit(dataset, y=None)
