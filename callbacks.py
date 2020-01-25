from os import path

import torch
from sklearn.metrics import jaccard_score, matthews_corrcoef
from skorch.callbacks import Checkpoint, EpochScoring
from skorch.utils import noop

SMOOTH = 1e-6


# Make sure scoring outputs a float number and not tensor
# Or else it won't be logged
def IOU(net, X, y):
    # y will be None
    iterator = net.get_iterator(X, training=False)
    iou = 0.0
    data_size = 0
    for data in iterator:
        Xi, yt = data
        yp = net.evaluation_step(Xi, training=False)
        yp = torch.sigmoid(yp)
        yp = (yp > 0.5).bool().cpu()
        yt = yt.bool().cpu()
        lengths = Xi["lengths"]
        batch_size = len(lengths)
        data_size += batch_size
        for i in range(batch_size):
            iou += jaccard_score(yt[i, : lengths[i]], yp[i, : lengths[i]])
    return iou / data_size


def MCC(net, X, y):
    # y will be None
    iterator = net.get_iterator(X, training=False)
    mcc = 0.0
    data_size = 0
    for data in iterator:
        Xi, yt = data
        yp = net.evaluation_step(Xi, training=False)
        yp = torch.sigmoid(yp)
        yp = (yp > 0.5).bool().cpu()
        yt = yt.bool().cpu()
        lengths = Xi["lengths"]
        batch_size = len(lengths)
        data_size += batch_size
        for i in range(batch_size):
            mcc += matthews_corrcoef(yt[i, : lengths[i]], yp[i, : lengths[i]])
    return mcc / data_size


class MyEpochScoring(EpochScoring):
    def get_test_data(self, dataset_train, dataset_valid):
        # Problem with getting the test data in original code
        dataset = dataset_train if self.on_train else dataset_valid

        if self.use_caching:
            X_test = dataset
            y_pred = self.y_preds_
            # Just return None here and do the computation in the scoring function
            return X_test, None, y_pred

        super().get_test_data(self, dataset_train, dataset_valid)


class MyCheckpoint(Checkpoint):
    def __init__(
        self,
        dirname,
        monitor,
        log_losses=["train_loss", "valid_loss"],
        log_scores=["IOU", "MCC"],
        f_params="model.pth",
        f_optimizer="optimizer.pt",
        f_history="history.json",
        f_pickle=None,
        fn_prefix="",
        event_name="event_cp",
        sink=noop,
    ):
        super().__init__(
            monitor=monitor,
            f_params=f_params,
            f_optimizer=f_optimizer,
            f_history=f_history,
            f_pickle=f_pickle,
            fn_prefix=fn_prefix,
            dirname=dirname,
            event_name=event_name,
            sink=sink,
        )
        self.log_losses = log_losses
        self.log_scores = log_scores

    def save_model(self, net):
        # Save the json file normally but also include a log.txt for easy viewing
        super().save_model(net)
        if self.f_history is not None:
            f = path.join(self.dirname, "log.txt")
            losses_history = {}
            scores_history = {}
            if self.monitor:
                # Only for specific models where the best score is needed
                file = open(f, "w")
                for key in self.log_losses:
                    try:
                        losses_history[key] = net.history[:, "batches", :, key]
                    except (KeyError):
                        pass
                for key in self.log_scores:
                    try:
                        scores_history[key] = net.history[:, key]
                    except (KeyError):
                        pass
                epochs = net.history[-1, "epoch"]
            else:
                # Latest model gets all the data stored / appended
                file = open(f, "a")
                for key in self.log_losses:
                    try:
                        losses_history[key] = [net.history[-1, "batches", :, key]]
                    except (KeyError):
                        pass
                for key in self.log_scores:
                    try:
                        scores_history[key] = [net.history[-1, key]]
                    except (KeyError):
                        pass
                epochs = 1
            # Each key in history will be a dictionary of lists of size [epochs, len_of_key]
            for epoch in range(epochs):
                for key in losses_history:
                    file.writelines(
                        "%s: %s\n" % (key, str(num))
                        for num in losses_history[key][epoch]
                    )
                for key in scores_history:
                    file.write("%s: %s\n" % (key, str(scores_history[key][epoch])))
            file.close()
