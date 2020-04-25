# Assuming that preprocessing has been completed from preprocessing folder
# We define a dataset based on the preprocessed data
import torch
from collections import defaultdict
import os

import numpy as np
from torch.utils.data import Dataset

data_dir = os.path.abspath("./data/scPDB")
splits_dir = os.path.join(data_dir, "splits")
preprocessed_dir = os.path.join(data_dir, "preprocessed")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# A collate function to merge samples into a minibatch, will be used by DataLoader
def collate_fn(samples):
    # samples is a list of (X, y) of size MINIBATCH_SIZE
    # Sort the samples in decreasing order of their length
    # x[1] will be y of each sample
    samples.sort(key=lambda x: len(x[1]), reverse=True)
    batch_size = len(samples)
    lengths = [0] * batch_size
    feat_vec_len, max_len = samples[0][0].shape
    X = torch.zeros(batch_size, feat_vec_len, max_len, device=device)
    y = torch.zeros(batch_size, max_len, device=device)
    for i, sample in enumerate(samples):
        lengths[i] = len(sample[1])
        X[i, :, : lengths[i]] = sample[0]
        y[i, : lengths[i]] = sample[1]
    return X, y, lengths


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
        for file in sorted(os.listdir(preprocessed_dir)):
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
#             yield train_indices[:24], valid_indices[:24]
            yield train_indices, valid_indices

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


class KalasantyChains(Dataset):
    def __init__(self, precompute_class_weights=False, **kwargs):
        super().__init__()
        self.train_folds = []
        self.valid_folds = []
        self.dataset = self.get_dataset()
        for i in range(10):
            with open(os.path.join(splits_dir, "train_ids_fold" + str(i))) as f:
                tmp = []
                for line in f.readlines():
                    tmp += self.dataset[line.strip()]
                self.train_folds.append(tmp)
        self.dataset_list = set(self.train_folds[0]).union(set(self.train_folds[1]))
        for i in range(10):
            self.valid_folds.append(list(self.dataset_list - set(self.train_folds[i])))
        self.dataset_list = sorted(list(self.dataset_list))
        self.dataset_id_to_index = defaultdict(int)
        for i, val in enumerate(self.dataset_list):
            self.dataset_id_to_index[val] = i
        if precompute_class_weights:
            # Already computed the weight from above
            self.pos_weight = [11238357 / 356850]
    
    def get_dataset(self):
        chains = defaultdict(list)
        for folder in sorted(os.listdir(preprocessed_dir)):
            if folder[:4] in chains:
                continue
            for file in sorted(os.listdir(os.path.join(preprocessed_dir, folder))):
                if file.startswith("feat"):
                    chains[folder[:4]].append(folder + "/" + file[8:9])
        extras = ["scPDB_blacklist.txt", "scPDB_leakage.txt"]
        for file in extras:
            with open(os.path.join(splits_dir, file)) as f:
                for line in f.readlines():
                    line = line.strip()
                    if line[:4] in chains and chains[line[:4]][0][:-2] == line:
                        del chains[line[:4]]
        return chains

    def custom_cv(self):
        for i in range(10):
            train_indices = [self.dataset_id_to_index[el] for el in self.train_folds[i]]
            valid_indices = [self.dataset_id_to_index[el] for el in self.valid_folds[i]]
#             yield train_indices[:24], valid_indices[:24]
            yield train_indices, valid_indices

    def __getitem__(self, index):
        pdb_id_struct, chain_id = self.dataset_list[index].split("/")
#         print(pdb_id_struct, chain_id)
        X = torch.from_numpy(
            np.load(os.path.join(preprocessed_dir, pdb_id_struct, "features" + chain_id + ".npy"))
        )
        y = torch.from_numpy(
            np.load(os.path.join(preprocessed_dir, pdb_id_struct, "labels" + chain_id + ".npy"))
        )
        # print(X.shape, y.shape)
        return X, y