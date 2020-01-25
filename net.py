import skorch
import torch
from torch.nn.functional import binary_cross_entropy_with_logits

from constants import DEVICE
from callbacks import MCC

# from skorch.utils import to_numpy


class Net(skorch.NeuralNet):
    def __init__(self, module, criterion, name=None, location=None, *args, **kwargs):
        super(Net, self).__init__(module, criterion, *args, **kwargs)
        self.name = name
        self.location = location

    def on_epoch_begin(self, net, dataset_train=None, dataset_valid=None, **kwargs):
        super(Net, self).on_epoch_begin(net, dataset_train, dataset_valid, **kwargs)
        if dataset_train and hasattr(dataset_train, "curr_fold"):
            dataset_train.curr_fold = (dataset_train.curr_fold + 1) % 10
        if dataset_valid and hasattr(dataset_valid, "curr_fold"):
            dataset_valid.curr_fold = (dataset_valid.curr_fold + 1) % 10

    def get_loss(self, y_pred, y_true, X, *args, **kwargs):
        lengths = X["lengths"]
        batch_size = len(lengths)
        # Use a variable criterion function
        ones = 0.0
        zeros = 0.0
        for i in range(batch_size):
            one = (y_true[i] == 1).float().sum()
            zeros += lengths[i] - one
            ones += one
        pos_weight = torch.zeros(1, device=DEVICE, dtype=torch.float32)
        # To avoid division by 0 which should not occur. Should check dataset
        pos_weight[0] = (zeros + 1) / (ones + 1)
        loss = 0
        for i in range(batch_size):
            loss += binary_cross_entropy_with_logits(
                y_pred[i, : lengths[i]], y_true[i, : lengths[i]], pos_weight=pos_weight
            )
        loss /= batch_size
        return loss

    def predict_proba(self, X):
        y_proba = []
        for yp in self.forward_iter(X, training=False):
            y_proba.append(yp)
        return y_proba

    def score(self, X, y=None):
        return MCC(self, X, y)
