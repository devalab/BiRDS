import os
from argparse import ArgumentParser
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_lightning.utilities.apply_func import move_data_to_device
from scipy.ndimage.filters import gaussian_filter1d
from sklearn.metrics._plot.precision_recall_curve import PrecisionRecallDisplay
from torchmetrics.functional.classification.auc import auc
from tqdm import tqdm

from birds.datasets import Birds
from birds.metrics import DCC, batch_work, confusion_matrix_figure
from birds.net import Net

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
plt.rcParams.update({"font.size": 20})


def dcc_figure(values, labels=None):
    len_values = len(values)
    values = np.nan_to_num(values, nan=20000, posinf=20000, neginf=20000)
    figure = plt.figure(figsize=(12, 12))
    x = np.linspace(0, 20, 1000)
    y = []
    for value in values:
        y.append(np.array([(value <= el).sum() / len(value) * 100 for el in x]))
    for i, colour in enumerate(colours[:len_values]):
        if labels:
            label = labels[i]
        else:
            label = "Fold " + str(i + 1)
        y_new = gaussian_filter1d(y[i], sigma=5)
        plt.plot(x, y_new, colour, label=label)
    plt.legend()
    plt.xticks(np.arange(0, 21, 1))
    plt.yticks(np.arange(0, 101, 5))
    plt.title("Distance to the center of the binding site")
    plt.ylabel("Success Rate")
    plt.xlabel("Distance to binding site")
    return figure


def roc_figure(fprs, tprs, areas, labels=None):
    len_values = len(fprs)
    figure = plt.figure(figsize=(12, 12))
    lw = 2
    for i, colour in enumerate(colours[:len_values]):
        if labels:
            label = labels[i]
        else:
            label = "Fold " + str(i + 1)
        tpr = gaussian_filter1d(tprs[i], sigma=5)
        plt.plot(
            fprs[i],
            tpr,
            colour,
            lw=lw,
            label="%s (area = %0.2f)" % (label, areas[i]),
        )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.01])
    plt.ylim([0.0, 1.01])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristics Curve")
    plt.legend(loc="lower right")
    return figure


def pr_figure(precisions, recalls, areas, labels=None):
    len_values = len(precisions)
    figure, ax = plt.subplots(figsize=(12, 12))

    f_scores = np.linspace(0.2, 0.8, num=4)
    _, f_labels = [], []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
        plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

    for i, colour in enumerate(colours[:len_values]):
        if labels:
            label = labels[i]
        else:
            label = "Fold " + str(i + 1)
        display = PrecisionRecallDisplay(recall=recalls[i], precision=precisions[i])
        display.plot(ax=ax, name="%s (area = %0.2f)" % (label, areas[i]), color=colour)

    # add the legend for the iso-f1 curves
    handles, f_labels = display.ax_.get_legend_handles_labels()
    handles.extend([l])
    f_labels.extend(["Iso-F1 curves"])
    # set the legend and the axes
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(handles=handles, labels=f_labels, loc="best")
    ax.set_title("Precision-Recall Curve")
    return figure


def make_figure(key, values, labels=None):
    if key[2:] == "ConfusionMatrix":
        cm = [[0, 0], [0, 0]]
        for value in values:
            cm[0][0] += value[0][0].numpy()
            cm[0][1] += value[0][1].numpy()
            cm[1][0] += value[1][0].numpy()
            cm[1][1] += value[1][1].numpy()
        return confusion_matrix_figure(np.array(cm), ["NBR", "BR"])
    elif key[2:] == "dcc":
        values = [[el.numpy() for el in value] for value in values]
        return dcc_figure(values, labels)
    elif key[2:] == "ROC":
        areas = [auc(value[0], value[1], reorder=True).detach().cpu().numpy() for value in values]
        return roc_figure([value[0].numpy() for value in values], [value[1].numpy() for value in values], areas, labels)
    elif key[2:] == "PrecisionRecallCurve":
        areas = [auc(value[0], value[1], reorder=True) for value in values]
        return pr_figure([value[0].numpy() for value in values], [value[1].numpy() for value in values], areas, labels)
    else:
        return plt.figure(figsize=(12, 12))


def validate(net, datamodule):
    print("Running model " + datamodule.train_ds.fold + " on its validation set")
    dcc = []
    for idx, batch in tqdm(enumerate(datamodule.val_dataloader())):
        batch = move_data_to_device(batch, net.device)
        dcc += move_data_to_device(net.validation_step(batch, idx)["f_v_dcc"], "cpu")
    metrics = move_data_to_device(net.valid_metrics.compute(), "cpu")

    figure_metrics = move_data_to_device(net.valid_figure_metrics.compute(), "cpu")
    figure_metrics["v_dcc"] = dcc
    return metrics, figure_metrics


def predict(nets, data, meta):
    y_preds = []
    for i, net in enumerate(nets):
        y_pred = torch.sigmoid(net(data["feature"], meta["length"], segment_label=data["segment_label"]))
        y_preds.append(y_pred)
    return torch.stack(y_preds).sum(dim=0) / len(nets)


def test(nets, datamodule):
    print("Running models on the test set")
    test_dl = datamodule.test_dataloader()
    device = nets[0].device
    dcc = []
    thresh = sum([net.hparams.threshold for net in nets]) / len(nets)
    for idx, batch in tqdm(enumerate(test_dl)):
        data, meta = move_data_to_device(batch, device)
        y_pred = predict(nets, data, meta)
        y_pred, y_true = batch_work(y_pred, data["label"], meta["length"])
        nets[0].test_metrics.update(y_pred, y_true.int())
        nets[0].test_figure_metrics.update(y_pred, y_true.int())
        dcc += move_data_to_device(DCC((y_pred >= thresh).bool(), y_true.bool(), data, meta), "cpu")
    metrics = move_data_to_device(nets[0].test_metrics.compute(), "cpu")
    figure_metrics = move_data_to_device(nets[0].test_figure_metrics.compute(), "cpu")
    figure_metrics["t_dcc"] = dcc
    return [metrics], [figure_metrics]


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


def load_nets_frozen(hparams, validate_one=False):
    nets = []
    test = False
    if not hasattr(hparams, "predict"):
        hparams.predict = False
    if not hasattr(hparams, "use_ohe"):
        hparams.use_ohe = True
    for i in range(10):
        print("Loading model for fold " + str(i))
        ckpt = get_best_ckpt(os.path.join(hparams.ckpt_dir, "fold_" + str(i), "checkpoints"))
        if i == 0:
            if not hparams.predict:
                test = True
        else:
            if validate_one:
                hparams.validate = False
            test = False
            hparams.predict = False
        net = Net.load_from_checkpoint(
            ckpt,
            data_dir=hparams.data_dir,
            gpus=hparams.gpus,
            run_tests=(not hparams.validate and test),
            load_train_ds=hparams.validate,
            predict=hparams.predict,
            use_ohe=hparams.use_ohe,
            input_size=47,
        )
        if hparams.gpus != 0:
            net = net.cuda()
        nets.append(net)
        nets[i].freeze()
        nets[i].eval()
        print()
    return nets


def print_metrics(metric):
    print("-------------------------")
    for k, v in metric.items():
        if type(v) == list:
            print(k + ":" + str((sum(v) / len(v)).item()))
        else:
            print(k + ": " + str(v.item()))
    print("-------------------------")


def main(hparams):
    print(hparams)
    nets = load_nets_frozen(hparams)

    if hparams.validate:
        metrics = []
        figure_metrics = []
        for i, net in enumerate(nets):
            datamodule = Birds(net.hparams)
            metric, figure_metric = validate(net, datamodule)
            metrics.append(metric)
            figure_metrics.append(figure_metric)
            print("Fold " + str(i) + " metrics")
            print_metrics(metric)
    else:
        datamodule = Birds(nets[0].hparams)
        metrics, figure_metrics = test(nets, datamodule)

    fnl_metrics = defaultdict(list)
    {fnl_metrics[key].append(val) for metric in metrics for key, val in metric.items()}
    fnl_figure_metrics = defaultdict(list)
    {fnl_figure_metrics[key].append(val) for metric in figure_metrics for key, val in metric.items()}

    print("Aggregated metrics")
    print_metrics(fnl_metrics)
    if not hparams.validate:
        label = ["Test (Full)"]
    else:
        label = None
    for key, value in fnl_figure_metrics.items():
        make_figure(key, value, label)
    plt.show(block=True)


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
