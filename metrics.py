import torch
from torch.nn.functional import binary_cross_entropy_with_logits

SMOOTH = 1e-6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def IOU(output, target):
    intersection = (output & target).float().sum()
    union = (output | target).float().sum()
    return (intersection) / (union)


def CM(output, target):
    # Get Confusion Matrix
    cm = {}
    cm["tp"] = (target * output).sum().float() + SMOOTH
    cm["tn"] = (~target * ~output).sum().float() + SMOOTH
    cm["fp"] = (~target * output).sum().float() + SMOOTH
    cm["fn"] = (target * ~output).sum().float() + SMOOTH
    return cm


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
    precision = PRECISION(cm)
    recall = RECALL(cm)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def LOSS(output, target, pos_weight=None, criterion=None):
    if criterion is not None:
        return criterion(output, target)
    if pos_weight is None:
        ones = (target == 1).float().sum()
        zeros = len(target) - ones
        pos_weight = torch.Tensor([(zeros + SMOOTH) / (ones + SMOOTH)], device=device)
    return binary_cross_entropy_with_logits(output, target, pos_weight=pos_weight)


def batch_loss(
    outputs, targets, lengths, pos_weight=None, criterion=None, reduction="mean"
):
    batch_size = len(lengths)
    loss = 0.0
    for i in range(batch_size):
        loss += LOSS(
            outputs[i, : lengths[i]], targets[i, : lengths[i]], pos_weight, criterion
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
        metrics = {
            "iou": IOU(output, target).item(),
            "mcc": MCC(cm).item(),
            "precision": PRECISION(cm).item(),
            "recall": RECALL(cm).item(),
            "f1": F1(cm).item(),
            "acc": ACCURACY(cm).item(),
        }
        if i == 0:
            running_metrics = metrics
            continue
        for key in metrics:
            running_metrics[key] += metrics[key]
    if reduction == "mean":
        for key in running_metrics:
            running_metrics[key] /= batch_size
    return running_metrics
