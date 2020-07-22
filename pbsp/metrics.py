import itertools
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy_with_logits, mse_loss
from torch_cluster import fps, grid_cluster

# Utils

SMOOTH = 1e-6


def confusion_matrix_figure(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype("float") / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use red text if squares are dark; otherwise black.
    threshold = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "red" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

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


def make_figure(name, values):
    if name == "cm":
        cm = [[0, 0], [0, 0]]
        for value in values:
            cm[0][0] += value[0]
            cm[0][1] += value[1]
            cm[1][0] += value[2]
            cm[1][1] += value[3]
        return confusion_matrix_figure(np.array(cm), ["NBR", "BR"])
    if name == "dcc":
        return dcc_figure(values)


# Validation Metrics


def CM(y_pred, y_true):
    # Get Confusion Matrix
    cm = {}
    cm["tp"] = (y_true * y_pred).sum().float() + SMOOTH
    cm["tn"] = (~y_true * ~y_pred).sum().float() + SMOOTH
    cm["fp"] = (~y_true * y_pred).sum().float() + SMOOTH
    cm["fn"] = (y_true * ~y_pred).sum().float() + SMOOTH
    return cm


def IOU(cm):
    return cm["tp"] / (cm["tp"] + cm["fp"] + cm["fn"])


def MCC(cm):
    return (cm["tp"] * cm["tn"] - cm["fp"] * cm["fn"]) / torch.sqrt(
        (cm["tp"] + cm["fp"])
        * (cm["tp"] + cm["fn"])
        * (cm["tn"] + cm["fp"])
        * (cm["tn"] + cm["fn"])
    )


def ACCURACY(cm):
    return (cm["tp"] + cm["tn"]) / (cm["tp"] + cm["tn"] + cm["fp"] + cm["fn"])


def RECALL(cm):
    return cm["tp"] / (cm["tp"] + cm["fn"])


def PRECISION(cm):
    return cm["tp"] / (cm["tp"] + cm["fp"])


def F1(cm):
    return (2 * cm["tp"]) / (2 * cm["tp"] + cm["fp"] + cm["fn"])


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


def GRID(y_pred, y_true, data, meta):
    idx = 0
    out = 0
    grid_size = torch.Tensor([40.0, 40.0, 40.0]).to(y_true.device)
    for i, length in enumerate(meta["length"]):
        coords = data["coords"][i, ..., :length]
        true_pocket = coords[:, y_true[idx : idx + length]]
        true_pocket = true_pocket[:, ~torch.isinf(true_pocket[0])].t()
        true_centroid = torch.mean(true_pocket, dim=0)
        pred_pocket = coords[:, y_pred[idx : idx + length]]
        pred_pocket = pred_pocket[:, ~torch.isinf(pred_pocket[0])].t()
        if len(pred_pocket) == 0:
            idx += length
            continue
        pred_labels = grid_cluster(pred_pocket, grid_size)
        max_cluster = []
        for val in pred_labels.unique():
            cluster = (pred_labels == val).nonzero()
            if len(max_cluster) < len(cluster):
                max_cluster = cluster
        pred_centroid = torch.mean(pred_pocket[max_cluster], dim=0)
        # tmp = true_labels.unique()
        # if len(tmp) != 1:
        #     print(meta["pisc"][i], len(tmp))
        return torch.norm(true_centroid - pred_centroid)
        if torch.norm(true_centroid - pred_centroid) < 8.0:
            out += 1
        idx += length
    return torch.tensor(float(out) / len(meta["length"])).float().to(y_true.device)


def FPS(y_pred, y_true, data, meta):
    idx = 0
    out = 0
    for i, length in enumerate(meta["length"]):
        coords = data["coords"][i, ..., :length]
        true_pocket = coords[:, y_true[idx : idx + length]]
        true_pocket = true_pocket[:, ~torch.isinf(true_pocket[0])].t()
        pred_pocket = coords[:, y_pred[idx : idx + length]]
        pred_pocket = pred_pocket[:, ~torch.isinf(pred_pocket[0])].t()
        if len(pred_pocket) == 0:
            idx += length
            continue
        true_fp = fps(true_pocket, ratio=0.5, random_start=False)
        true_neighbours = true_pocket[
            torch.ones_like(true_pocket[:, 0]).bool().scatter_(0, true_fp, 0.0)
        ]
        true_centroid = torch.mean(true_neighbours, dim=0)
        pred_fp = fps(pred_pocket, ratio=0.5, random_start=False)
        pred_neighbours = pred_pocket[
            torch.ones_like(pred_pocket[:, 0]).bool().scatter_(0, pred_fp, 0.0)
        ]
        pred_centroid = torch.mean(pred_neighbours, dim=0)
        return torch.norm(true_centroid - pred_centroid)
        if torch.norm(true_centroid - pred_centroid) < 8.0:
            out += 1
        idx += length
    return torch.tensor(out / len(meta["length"])).to(y_true.device)


# Loss Functions


def weighted_focal_loss(y_pred, y_true, gamma=2.0, pos_weight=[1], **kwargs):
    pos_weight = torch.Tensor(pos_weight).to(y_true.device)
    y_pred = torch.clamp(torch.sigmoid(y_pred), SMOOTH, 1.0 - SMOOTH)
    loss = -(
        pos_weight * y_true * torch.pow(1.0 - y_pred, gamma) * torch.log(y_pred)
    ) - ((1 - y_true) * torch.pow(y_pred, gamma) * torch.log(1.0 - y_pred))
    return torch.mean(loss)


def pl_weighted_loss(y_pred, y_true, batch_idx, lengths, pl_dist=None, **kwargs):
    y_pred = torch.clamp(torch.sigmoid(y_pred), SMOOTH, 1.0 - SMOOTH)
    if pl_dist is not None:
        tmp = pl_dist[batch_idx, : lengths[batch_idx]]
        loss = -(15 * y_true * torch.log(y_pred)) - (
            (1 - y_true) * torch.log(1.0 - y_pred) * (tmp / 10.0)
        )
    else:
        loss = -(y_true * torch.log(y_pred)) - ((1 - y_true) * torch.log(1.0 - y_pred))
    return torch.mean(loss)


def weighted_bce_loss(y_pred, y_true, pos_weight, **kwargs):
    if pos_weight is None:
        pos_weight = y_true.sum()
        pos_weight = [(len(y_true) - pos_weight) / pos_weight]
    pos_weight = torch.Tensor(pos_weight).to(y_true.device)
    return binary_cross_entropy_with_logits(
        y_pred, y_true, pos_weight=pos_weight, reduction="mean"
    )


def detector_margin_loss(y_pred, y_true, **kwargs):
    y_pred = torch.clamp(torch.sigmoid(y_pred), SMOOTH, 1.0 - SMOOTH)
    loss = (1 - y_true) * (
        -(y_pred * torch.log(y_pred)) - ((1 - y_pred) * torch.log(1 - y_pred))
    )
    return torch.mean(loss)


def generator_pos_loss(y_pred, y_true, **kwargs):
    y_pred = torch.clamp(torch.sigmoid(y_pred), SMOOTH, 1.0 - SMOOTH)
    loss = (1 - y_true) * (-torch.log(y_pred))
    return torch.mean(loss)


def generator_mse_loss(generated_embed, embed, mask, batch_idx, **kwargs):
    embed = torch.masked_select(embed, mask[batch_idx].unsqueeze(1))
    generated_embed = torch.masked_select(generated_embed, mask[batch_idx].unsqueeze(1))
    return mse_loss(generated_embed, embed, reduction="mean")


def batch_work(y_preds, y_trues, lengths):
    batch_size = len(lengths)
    indices = []
    for i in range(batch_size):
        indices += [i * lengths[0] + el for el in range(lengths[i])]
    indices = torch.tensor(indices).to(y_trues.device)
    return y_preds.take(indices), y_trues.take(indices)


def batch_metrics(y_preds, data, meta, is_logits=True, threshold=0.5):
    y_preds, y_trues = batch_work(y_preds, data["label"], meta["length"])
    if is_logits:
        y_preds = torch.sigmoid(y_preds)
    y_preds = (y_preds > threshold).bool()
    y_trues = y_trues.bool()
    # top_cluster(y_preds, y_trues, data, meta)
    # return {}
    cm = CM(y_preds, y_trues)
    metrics = OrderedDict(
        {
            "mcc": MCC(cm),
            "acc": ACCURACY(cm),
            "fps": FPS(y_preds, y_trues, data, meta),
            "grid": GRID(y_preds, y_trues, data, meta),
            "iou": IOU(cm),
            "precision": PRECISION(cm),
            "recall": RECALL(cm),
            "f1": F1(cm),
            "f_cm": [torch.stack([cm["tn"], cm["fp"], cm["fn"], cm["tp"]])],
            "f_dcc": DCC(y_preds, y_trues, data, meta),
        }
    )
    return metrics


def batch_loss(y_preds, y_trues, lengths, loss_func, **kwargs):
    y_preds, y_trues = batch_work(y_preds, y_trues, lengths)
    loss = loss_func(y_preds, y_trues, **kwargs)
    return loss
