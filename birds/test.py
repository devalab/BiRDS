import os
from argparse import ArgumentParser
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from pytorch_lightning.utilities.apply_func import move_data_to_device
from scipy.ndimage.filters import gaussian_filter1d
from tqdm import tqdm

from birds.metrics import *
from birds.net import Net


def validate(net):
    print("Running model " + net.train_ds.fold + " on its validation set")
    dcc = []
    cm = defaultdict(int)
    fnl_metrics = {}
    for idx, batch in tqdm(enumerate(net.val_dataloader())):
        batch = move_data_to_device(batch, net.device)
        metrics = net.validation_step(batch, idx)
        dcc += [el.item() for el in metrics["f_v_dcc"]]
        cm["tn"] += metrics["f_v_cm"][0][0]
        cm["fp"] += metrics["f_v_cm"][0][1]
        cm["fn"] += metrics["f_v_cm"][0][2]
        cm["tp"] += metrics["f_v_cm"][0][3]
    fnl_metrics["cm"] = cm
    fnl_metrics["dcc"] = dcc
    return fnl_metrics


def test(nets, threshold=5):
    print("Running models on the test set")
    test_dl = nets[0].test_dataloader()
    device = nets[0].device
    dcc = []
    cm = defaultdict(int)
    fnl_metrics = {}
    for idx, batch in enumerate(test_dl):
        data, meta = move_data_to_device(batch, device)
        y_preds = []
        for i, net in enumerate(nets):
            y_pred = (torch.sigmoid(net(data["feature"], meta["length"])) > 0.5).bool()
            y_preds.append(y_pred)
        y_pred = (torch.stack(y_preds).sum(dim=0) >= threshold).bool()
        metrics = batch_metrics(y_pred, data, meta, False, None)
        dcc += [el.item() for el in metrics["f_dcc"]]
        cm["tn"] += metrics["f_cm"][0][0]
        cm["fp"] += metrics["f_cm"][0][1]
        cm["fn"] += metrics["f_cm"][0][2]
        cm["tp"] += metrics["f_cm"][0][3]
    fnl_metrics["cm"] = cm
    fnl_metrics["dcc"] = dcc
    return fnl_metrics


def dcc_figure(values):
    values = np.nan_to_num(values, nan=20000, posinf=20000, neginf=20000)
    figure = plt.figure(figsize=(12, 12))
    x = np.linspace(0, 20, 1000)
    y = []
    for value in values:
        y.append(np.array([(value <= el).sum() / len(value) * 100 for el in x]))
    colours = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
    ]
    for i, colour in enumerate(colours):
        y_new = gaussian_filter1d(y[i], sigma=5)
        plt.plot(x, y_new, colour, label="Fold " + str(i + 1))
    plt.legend()
    plt.xticks(np.arange(0, 21, 1))
    plt.yticks(np.arange(0, 101, 5))
    plt.title("Distance to the center of the binding site")
    plt.ylabel("Success Rate")
    plt.xlabel("Distance to binding site")
    plt.show()
    return figure


def get_best_ckpt(folder):
    tmp = [el[:-5].split("-") for el in sorted(os.listdir(folder))]
    tmp = sorted(
        tmp,
        key=lambda x: (
            float(x[1].split("=")[1]),
            float(x[2].split("=")[1]),
            float(x[3].split("=")[1]),
        ),
        reverse=True,
    )
    return os.path.join(folder, "-".join(tmp[0]) + ".ckpt")


def load_nets_frozen(hparams):
    nets = []
    for i in range(10):
        print("Loading model for fold " + str(i))
        ckpt = get_best_ckpt(
            os.path.join(hparams.ckpt_dir, "fold_" + str(i), "checkpoints")
        )
        if i == 0:
            test = True
        else:
            test = False
        net = Net.load_from_checkpoint(
            ckpt,
            data_dir=hparams.data_dir,
            gpus=hparams.gpus,
            run_tests=(not hparams.validate and test),
        )
        if hparams.gpus != 0:
            net = net.cuda()
        nets.append(net)
        nets[i].freeze()
        nets[i].eval()
        print()
    return nets


def print_metrics(cm):
    print("-------------------------")
    print("IOU: ", IOU(cm).item())
    print("MCC: ", MCC(cm).item())
    print("ACCURACY: ", ACCURACY(cm).item())
    print("RECALL: ", RECALL(cm).item())
    print("PRECISION: ", PRECISION(cm).item())
    print("F1: ", F1(cm).item())
    print("-------------------------")


def main(hparams):
    nets = load_nets_frozen(hparams)

    if hparams.validate:
        metrics = []
        for i, net in enumerate(nets):
            metric = validate(net)
            metrics.append(metric)
            print("Fold " + str(i) + " metrics")
            print_metrics(cm)
    else:
        metrics = test(nets)

    dcc_figure([metric["dcc"] for metric in metrics])
    cm = defaultdict(int)
    for metric in metrics:
        for key, val in metric["cm"].items():
            cm[key] += val
    print("Aggregated metrics")
    print_metrics(cm)


def parse_arguments():
    parser = ArgumentParser(description="Binding Site Predictor", add_help=True)
    parser.add_argument(
        "--ckpt_dir",
        default="../model",
        type=str,
        help="Checkpoint directory containing checkpoints of all 10 folds. Default: %(default)s",
    )
    parser.add_argument(
        "--data-dir",
        default="../data",
        type=str,
        help="Location of data directory. Default: %(default)s",
    )
    parser.add_argument(
        "--gpus",
        default=1,
        type=int,
        help="Number of gpus to use for computation. Default: %(default)d",
    )
    parser.add_argument(
        "--validate",
        dest="validate",
        action="store_true",
        help="Run the models on their validation sets. Default: %(default)s",
    )
    parser.set_defaults(validate=False)
    hparams = parser.parse_args()
    hparams.data_dir = os.path.abspath(os.path.expanduser(hparams.data_dir))
    hparams.ckpt_dir = os.path.abspath(os.path.expanduser(hparams.ckpt_dir))
    return hparams


if __name__ == "__main__":
    main(parse_arguments())
