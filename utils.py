from os import path, makedirs, system

import torch

from constants import DEVICE


def copy_code(from_location, to_location):
    # Copy .sh and .py files
    if not path.exists(to_location):
        makedirs(to_location)
    system(
        "rsync -mar --exclude='outputs' --include='*/' "
        + "--include='*\.py' --include='*\.sh' --exclude='*' "
        + from_location
        + " "
        + to_location
    )


def load_net(net_path):
    from train import initialize_net

    net = initialize_net()
    if DEVICE == torch.device("cpu"):
        net.load_state_dict(torch.load(net_path, map_location={"cuda:0": "cpu"}))
    else:
        net.load_state_dict(torch.load(net_path))
    return net
