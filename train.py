#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Some constants that will be required
# ALWAYS RUN THIS CODE CELL
import os
from glob import glob

import torch

data_dir = os.path.abspath("./data/scPDB")
splits_dir = os.path.join(data_dir, "splits")
preprocessed_dir = os.path.join(data_dir, "preprocessed")
torch.manual_seed(42)
device = torch.device("cpu")
if torch.cuda.is_available():
    print("Using available GPU")
    device = torch.device("cuda")


# In[ ]:


# Assuming that preprocessing has been completed from preprocessing folder
# We define a dataset based on the preprocessed data
from collections import defaultdict

import numpy as np
from torch.utils.data import Dataset


class Kalasanty(Dataset):
    def __init__(self, precompute_class_weights=False, **kwargs):
        super().__init__()
        self.train_folds = []
        self.valid_folds = []
        for i in range(10):
            with open(os.path.join(splits_dir, "train_ids_fold" + str(i))) as f:
                self.train_folds.append([line.strip() for line in f.readlines()])
        self.dataset_list = set(self.train_folds[0]).union(set(self.train_folds[1]))
        for i in range(10):
            self.valid_folds.append(list(self.dataset_list - set(self.train_folds[i])))
        self.dataset = self.get_dataset()
        self.dataset_list = sorted(list(self.dataset_list))
        self.dataset_id_to_index = defaultdict(int)
        for i, val in enumerate(self.dataset_list):
            self.dataset_id_to_index[val] = i
        if precompute_class_weights:
            # self.pos_weight = self.compute_class_weights()
            # Already computed the weight from above
            self.pos_weight = [11238357 / 356850]

    def get_dataset(self):
        available = defaultdict(list)
        for file in os.listdir(preprocessed_dir):
            available[file[:4]].append(file)

        extras = ["scPDB_blacklist.txt", "scPDB_leakage.txt"]
        for file in extras:
            with open(os.path.join(splits_dir, file)) as f:
                for line in f.readlines():
                    line = line.strip()
                    if line in available[line[:4]]:
                        available[line[:4]].remove(line)
                    if available[line[:4]] == list():
                        del available[line[:4]]

        for key in set(available.keys()) - self.dataset_list:
            del available[key]

        return available
    
    def compute_class_weights(self):
        print("Precomputing class weights...")
        zeros = 0
        ones = 0
        for i, pdb_id in enumerate(self.dataset_list):
            pdb_id_struct = self.dataset[pdb_id][0]
            y = np.load(os.path.join(preprocessed_dir, pdb_id_struct, "labels.npy"))
            one = np.count_nonzero(y)
            ones += one
            zeros += len(y) - one
        print(zeros, ones)
        pos_weight = [zeros / ones]
        print(pos_weight)
        print("Done")
        return pos_weight

    def custom_cv(self):
        for i in range(10):
            train_indices = [self.dataset_id_to_index[el] for el in self.train_folds[i]]
            valid_indices = [self.dataset_id_to_index[el] for el in self.valid_folds[i]]
            yield train_indices[:24], valid_indices[:24]
#             yield train_indices, valid_indices

    def __getitem__(self, index):
        pdb_id = self.dataset_list[index]
        # Just taking the first available structure for a pdb #TODO
        pdb_id_struct = self.dataset[pdb_id][0]
        # print(pdb_id_struct)
        X = torch.from_numpy(
            np.load(os.path.join(preprocessed_dir, pdb_id_struct, "features.npy"))
        )
        y = torch.from_numpy(
            np.load(os.path.join(preprocessed_dir, pdb_id_struct, "labels.npy"))
        )
        # print(X.shape, y.shape)
        return X, y


# A collate function to merge samples into a minibatch, will be used by DataLoader
def collate_fn(samples):
    # samples is a list of (X, y) of size MINIBATCH_SIZE
    # Sort the samples in decreasing order of their length
    # x[1] will be y of each sample
    samples.sort(key=lambda x: len(x[1]), reverse=True)
    batch_size = len(samples)
    lengths = [0] * batch_size
    max_len = samples[0][0].shape[1]
    X = torch.zeros(batch_size, feat_vec_len, max_len, device=device)
    y = torch.zeros(batch_size, max_len, device=device)
    for i, sample in enumerate(samples):
        lengths[i] = len(sample[1])
        X[i, :, : lengths[i]] = sample[0]
        y[i, : lengths[i]] = sample[1]
    return X, y, lengths


# In[ ]:


from tqdm.auto import tqdm
from metrics import batch_metrics, batch_loss

# Define the main training loop
def train_loop(model, dl, pr=50):
    model.train()
    running_loss = 0.0
    pbar = tqdm(dl, leave=False)
    for i, batch_el in enumerate(pbar):
        X, y, lengths = batch_el
        optimizer.zero_grad()
        y_pred = model(X, lengths)
        loss = batch_loss(y_pred, y, lengths, criterion=criterion)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % pr == pr - 1:
            pbar.set_postfix({"train_loss": loss.item()})
    print(f"Train --- %.8f" % (running_loss / len(dl)))


# Define the main validation loop
def valid_loop(model, dl):
    model.eval()
    with torch.no_grad():
        pbar = tqdm(dl, leave=False)
        for i, batch_el in enumerate(pbar):
            X, y, lengths = batch_el
            y_pred = model(X, lengths)
            metrics = batch_metrics(y_pred, y, lengths)
            metrics["loss"] = batch_loss(y_pred, y, lengths, criterion=criterion).item()
            pbar.set_postfix(metrics)
            if i == 0:
                running_metrics = metrics
                continue
            for key in metrics:
                running_metrics[key] += metrics[key]
        print("Validation --- ", end="")
        for key in metrics:
            print(f"%s: %.5f" % (key, (running_metrics[key] / len(dl))), end=" ")
        print()


# In[ ]:


from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.nn import BCEWithLogitsLoss
from models import *

max_epochs = 50
learning_rate = 0.02
dataset = Kalasanty(precompute_class_weights=True)
criterion = BCEWithLogitsLoss(pos_weight=torch.Tensor(dataset.pos_weight)).to(device)
feat_vec_len = dataset[0][0].shape[0]
models = []
optimizers = []

for i, (train_indices, valid_indices) in enumerate(dataset.custom_cv()):
    # model = StackedNN(feat_vec_len).to(device)
    model = ResNet(feat_vec_len, layers=[2, 2, 2, 2], kernel_sizes=[7, 7]).to(device)
    #     print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    models.append(model)
    optimizers.append(optimizer)
    print()
    print("Model #" + str(i + 1), "--------------------------------------------")
    for epoch in range(max_epochs):
        print("Epoch", str(epoch), end=" ")
        # Don't use multiprocessing here since our dataloading is I/O bound and not CPU
        train_dl = DataLoader(
            dataset,
            batch_size=1,
            sampler=SubsetRandomSampler(train_indices),
            collate_fn=collate_fn,
        )
        train_loop(model, train_dl)
        valid_dl = DataLoader(
            dataset,
            batch_size=1,
            sampler=SubsetRandomSampler(valid_indices),
            collate_fn=collate_fn,
        )
        valid_loop(model, valid_dl)

