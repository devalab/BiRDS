from collections import OrderedDict
from torch.nn.functional import binary_cross_entropy_with_logits, mse_loss
import torch

SMOOTH = 1e-6


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


def batch_metrics(y_preds, y_trues, lengths):
    batch_size = len(lengths)
    for i in range(batch_size):
        y_pred = (torch.sigmoid(y_preds[i, : lengths[i]]) > 0.5).bool()
        y_true = y_trues[i, : lengths[i]].bool()
        cm = CM(y_pred, y_true)
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
    for key in running_metrics:
        running_metrics[key] /= batch_size
    return running_metrics


def weighted_focal_loss(y_pred, y_true, gamma=2.0, pos_weight=[1], **kwargs):
    pos_weight = torch.Tensor(pos_weight).type(y_true[0].type())
    y_pred = torch.clamp(torch.sigmoid(y_pred), SMOOTH, 1.0 - SMOOTH)
    loss = -(
        pos_weight * y_true * torch.pow(1.0 - y_pred, gamma) * torch.log(y_pred)
    ) - ((1 - y_true) * torch.pow(y_pred, gamma) * torch.log(1.0 - y_pred))
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


def batch_loss(y_preds, y_trues, lengths, loss_func, **kwargs):
    batch_size = len(lengths)
    loss = 0.0
    for i in range(batch_size):
        loss += loss_func(
            y_preds[i, : lengths[i]],
            y_trues[i, : lengths[i]],
            batch_idx=i,
            lengths=lengths,
            **kwargs
        )
    loss /= batch_size
    return loss
