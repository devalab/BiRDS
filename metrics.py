from collections import OrderedDict

import torch
import numpy as np
import matplotlib.pyplot as plt
import itertools
from torch.nn.functional import binary_cross_entropy_with_logits, mse_loss

SMOOTH = 1e-6


def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
    """
    for key in cm:
        cm[key] = cm[key].cpu().numpy()
    cm = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype("float") / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    return figure


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


def weighted_focal_loss(y_pred, y_true, gamma=2.0, pos_weight=[1], **kwargs):
    pos_weight = torch.Tensor(pos_weight).type(y_true[0].type())
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


def weighted_bce_loss(y_pred, y_true, pos_weight=[1], **kwargs):
    pos_weight = torch.Tensor(pos_weight).type(y_true[0].type())
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


def batch_metrics(
    y_preds, y_trues, lengths, is_logits=True, threshold=0.5, logger=None, epoch=0
):
    y_preds, y_trues = batch_work(y_preds, y_trues, lengths)
    if is_logits:
        y_preds = torch.sigmoid(y_preds)
    y_preds = (y_preds > threshold).bool()
    y_trues = y_trues.bool()
    cm = CM(y_preds, y_trues)
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
    if logger:
        figure = plot_confusion_matrix(cm, ["NBR", "BR"])
        logger.experiment.add_figure("Confusion Matrix", figure, global_step=epoch)
    return metrics


def batch_loss(y_preds, y_trues, lengths, loss_func, **kwargs):
    y_preds, y_trues = batch_work(y_preds, y_trues, lengths)
    loss = loss_func(y_preds, y_trues, **kwargs)
    return loss
