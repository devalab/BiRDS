import torch
from skorch.callbacks import Checkpoint, EpochScoring

SMOOTH = 1e-6


# Make sure scoring outputs a float number and not tensor
# Or else it won't be logged
def IOU(net, X, y):
    # y will be None
    iterator = net.get_iterator(X, training=False)
    intersection = 0
    union = 0
    for data in iterator:
        Xi, yt = data
        yp = net.evaluation_step(Xi, training=False)
        yp = torch.sigmoid(yp)
        yp = (yp > 0.5).int()
        yt = yt.int()
        lengths = Xi["lengths"]
        batch_size = len(lengths)
        for i in range(batch_size):
            intersection += (yp[i, : lengths[i]] & yt[i, : lengths[i]]).float().sum()
            union += (yp[i, : lengths[i]] | yt[i, : lengths[i]]).float().sum()
    iou = (intersection + SMOOTH) / (union + SMOOTH)

    return 100.0 * iou.item()


class MyEpochScoring(EpochScoring):
    def __init__(self, scoring, lower_is_better=True):
        super().__init__(scoring, lower_is_better=lower_is_better)

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
        log_scores=["IOU"],
        f_params="model.pth",
        f_optimizer="optimizer.pth",
        f_history="history.json",
    ):
        super().__init__(
            monitor=monitor,
            f_params=f_params,
            f_optimizer=f_optimizer,
            f_history=f_history,
            dirname=dirname,
        )
        self.log_losses = ["train_loss", "valid_loss"]
        self.log_scores = log_scores

    def save_model(self, net):
        # Save the json file normally but also include a log.txt for easy viewing
        super().save_model(net)
        if self.f_history is not None:
            f = self.dirname + "log.txt"
            losses_history = {}
            scores_history = {}
            if self.monitor:
                file = open(f, "w")
                for key in self.log_losses:
                    losses_history[key] = net.history[:, "batches", :, key]
                for key in self.log_scores:
                    scores_history[key] = net.history[:, key]
                epochs = net.history[-1, "epoch"]
            else:
                file = open(f, "a")
                for key in self.log_losses:
                    losses_history[key] = [net.history[-1, "batches", :, key]]
                for key in self.log_scores:
                    scores_history[key] = [net.history[-1, key]]
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
