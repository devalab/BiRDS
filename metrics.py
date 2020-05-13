from collections import OrderedDict

import torch
from torch.nn.functional import binary_cross_entropy_with_logits

SMOOTH = 1e-6


def CM(output, target):
    # Get Confusion Matrix
    cm = {}
    cm["tp"] = (target * output).sum().float() + SMOOTH
    cm["tn"] = (~target * ~output).sum().float() + SMOOTH
    cm["fp"] = (~target * output).sum().float() + SMOOTH
    cm["fn"] = (target * ~output).sum().float() + SMOOTH
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


def batch_loss(outputs, targets, lengths, loss_func, reduction="mean", **kwargs):
    batch_size = len(lengths)
    loss = 0.0
    for i in range(batch_size):
        loss += loss_func(
            outputs[i, : lengths[i]], targets[i, : lengths[i]], batch_idx=i, **kwargs
        )
    if reduction == "mean":
        loss /= batch_size
    return loss


def batch_metrics(outputs, targets, lengths, reduction="mean"):
    batch_size = len(lengths)
    for i in range(batch_size):
        output = (torch.sigmoid(outputs[i, : lengths[i]]) > 0.5).bool()
        target = targets[i, : lengths[i]].bool()
        cm = CM(output, target)
        metrics = OrderedDict(
            {
                "mcc": MCC(cm),
                "acc": ACCURACY(cm),
                "iou": IOU(cm),
                "precision": PRECISION(cm),
                "recall": RECALL(cm),
                "f1": F1(cm),
            }
        )
        if i == 0:
            running_metrics = metrics
            continue
        for key in metrics:
            running_metrics[key] += metrics[key]
    if reduction == "mean":
        for key in running_metrics:
            running_metrics[key] /= batch_size
    return running_metrics
