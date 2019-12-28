import skorch
import torch

SMOOTH = 1e-6


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
        lengths = Xi["lengths"]
        batch_size = len(lengths)
        for i in range(batch_size):
            intersection += (yp & yt.int()).float().sum()
            union += (yp | yt.int()).float().sum()
    iou = (intersection + SMOOTH) / (union + SMOOTH)

    return 100.0 * iou


class MyEpochScoring(skorch.callbacks.EpochScoring):
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
