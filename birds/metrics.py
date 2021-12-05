import itertools

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import PrecisionRecallDisplay
from torch.nn.functional import binary_cross_entropy_with_logits
from torchmetrics.functional import auc

SMOOTH = 1e-6
plt.rcParams.update({"font.size": 18})


def confusion_matrix_figure(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
    """
    cm = cm.astype("int")
    figure = plt.figure(figsize=(8, 8))
    # Normalize the confusion matrix.
    cmap = plt.cm.viridis
    normalized = np.around(cm.astype("float") / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    plt.imshow(normalized, interpolation="nearest", cmap=cmap)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Use red text if squares are dark; otherwise black.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = cmap(1.0) if normalized[i, j] < 0.5 else cmap(0.0)
        plt.text(
            j,
            i,
            str(normalized[i, j]) + "\n(" + str(cm[i, j]) + ")",
            horizontalalignment="center",
            verticalalignment="center",
            color=color,
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    return figure


def dcc_figure(values):
    values = np.nan_to_num(values, nan=20000, posinf=20000, neginf=20000)
    figure = plt.figure(figsize=(8, 8))
    x = np.linspace(0, 20, 1000)
    y = np.array([(values <= el).sum() / len(values) * 100 for el in x])
    plt.plot(x, y)
    plt.xticks(np.arange(0, 21, 1))
    plt.yticks(np.arange(0, 101, 5))
    plt.title("Distance to the center of the binding site")
    plt.ylabel("Success Rate")
    plt.xlabel("Distance to binding site")
    return figure


def roc_figure(fpr, tpr, area):
    figure = plt.figure(figsize=(8, 8))
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % area,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.01])
    plt.ylim([0.0, 1.01])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve")
    plt.legend(loc="lower right")
    return figure


def pr_figure(precision, recall, area):
    figure, ax = plt.subplots(figsize=(8, 8))

    f_scores = np.linspace(0.2, 0.8, num=4)
    _, labels = [], []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
        plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

    display = PrecisionRecallDisplay(recall=recall, precision=precision)
    display.plot(ax=ax, name="PR curve (area = %0.2f)" % area, color="gold")

    # add the legend for the iso-f1 curves
    handles, labels = display.ax_.get_legend_handles_labels()
    handles.extend([l])
    labels.extend(["Iso-F1 curves"])
    # set the legend and the axes
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(handles=handles, labels=labels, loc="best")
    ax.set_title("Precision-Recall curve")

    return figure


def make_figure(key, values):
    if key[2:] == "ConfusionMatrix":
        return confusion_matrix_figure(values.detach().cpu().numpy(), ["NBR", "BR"])
    elif key[2:] == "dcc":
        return dcc_figure(values.detach().cpu().numpy())
    elif key[2:] == "ROC":
        area = auc(values[0], values[1], reorder=True)
        return roc_figure(values[0].detach().cpu().numpy(), values[1].detach().cpu().numpy(), area)
    elif key[2:] == "PrecisionRecallCurve":
        area = auc(values[0], values[1], reorder=True)
        return pr_figure(values[0].detach().cpu().numpy(), values[1].detach().cpu().numpy(), area)
    else:
        return plt.figure(figsize=(8, 8))


def DCC(y_pred, y_true, data, meta):
    idx = 0
    dcc = []
    for i, length in enumerate(meta["length"]):
        coords = data["coords"][i, ..., :length]
        true_pocket = coords[:, y_true[idx : idx + length]]
        true_pocket = true_pocket[:, ~torch.isinf(true_pocket[0])]
        pred_pocket = coords[:, y_pred[idx : idx + length]]
        pred_pocket = pred_pocket[:, ~torch.isinf(pred_pocket[0])]
        if len(pred_pocket) == 0:
            dcc += [torch.tensor(100.0).float().to(y_true.device)]
            idx += length
            continue
        true_centroid = torch.mean(true_pocket, dim=1)
        pred_centroid = torch.mean(pred_pocket, dim=1)
        dcc += [torch.norm(true_centroid - pred_centroid)]
        idx += length
    return dcc


def weighted_focal_loss(y_pred, y_true, gamma=2.0, pos_weight=[1], **kwargs):
    pos_weight = torch.Tensor(pos_weight).to(y_true.device)
    y_pred = torch.clamp(torch.sigmoid(y_pred), SMOOTH, 1.0 - SMOOTH)
    loss = -(pos_weight * y_true * torch.pow(1.0 - y_pred, gamma) * torch.log(y_pred)) - (
        (1 - y_true) * torch.pow(y_pred, gamma) * torch.log(1.0 - y_pred)
    )
    return torch.mean(loss)


def weighted_bce_loss(y_pred, y_true, pos_weight, **kwargs):
    if pos_weight is None:
        pos_weight = y_true.sum()
        pos_weight = [(len(y_true) - pos_weight) / pos_weight]
    pos_weight = torch.Tensor(pos_weight).to(y_true.device)
    return binary_cross_entropy_with_logits(y_pred, y_true, pos_weight=pos_weight, reduction="mean")


def batch_work(y_preds, y_trues, lengths):
    batch_size = len(lengths)
    indices = []
    for i in range(batch_size):
        indices += [i * lengths[0] + el for el in range(lengths[i])]
    indices = torch.tensor(indices).to(y_trues.device)
    return y_preds.take(indices), y_trues.take(indices)
