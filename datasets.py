# Assuming that preprocessing has been completed from preprocessing folder
# We define a dataset based on the preprocessed data
import os
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

data_dir = os.path.abspath("./data/scPDB")
splits_dir = os.path.join(data_dir, "splits")
preprocessed_dir = os.path.join(data_dir, "preprocessed")


# A collate function to merge samples into a minibatch, will be used by DataLoader
def collate_fn(samples):
    # samples is a list of (X, y) of size MINIBATCH_SIZE
    # Sort the samples in decreasing order of their length
    # x[1] will be y of each sample
    samples.sort(key=lambda x: len(x[1]), reverse=True)
    batch_size = len(samples)
    lengths = [0] * batch_size
    feat_vec_len, max_len = samples[0][0].shape
    X = torch.zeros(batch_size, feat_vec_len, max_len)
    y = torch.zeros(batch_size, max_len)
    for i, sample in enumerate(samples):
        lengths[i] = len(sample[1])
        X[i, :, : lengths[i]] = sample[0]
        y[i, : lengths[i]] = sample[1]
    return X, y, lengths


class KalasantyBase(Dataset):
    def __init__(self):
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
        self.class_counts = self.get_class_counts()
        self.pos_weight = self.class_counts[0] / self.class_counts[1]

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

    def get_class_counts(self):
        print("Computing class sample counts...")
        zeros = 0
        ones = 0
        # NOTE: Using just the first fold for now
        for i, pdb_id in tqdm(enumerate(self.train_folds[0]), leave=False):
            pdb_id_struct = self.dataset[pdb_id][0]
            y = np.load(os.path.join(preprocessed_dir, pdb_id_struct, "labels.npy"))
            one = np.count_nonzero(y)
            ones += one
            zeros += len(y) - one
        print(zeros, ones)
        print("Done")
        return [zeros, ones]

    def custom_cv(self):
        for i in range(10):
            train_indices = [self.dataset_id_to_index[el] for el in self.train_folds[i]]
            valid_indices = [self.dataset_id_to_index[el] for el in self.valid_folds[i]]
            #             yield train_indices[:24], valid_indices[:24]
            yield train_indices, valid_indices


class Kalasanty(KalasantyBase):
    def __init__(self):
        super().__init__()

    def __getitem__(self, index):
        pdb_id = self.dataset_list[index]
        # Just taking the first available structure for a pdb #TODO
        pdb_id_struct = self.dataset[pdb_id][0]
        X = torch.from_numpy(
            np.load(os.path.join(preprocessed_dir, pdb_id_struct, "features.npy"))
        )
        y = torch.from_numpy(
            np.load(os.path.join(preprocessed_dir, pdb_id_struct, "labels.npy"))
        )
        return X, y


class KalasantyChains(KalasantyBase):
    def __init__(self):
        super().__init__()

    def __getitem__(self, index):
        pdb_id_struct, chain_id = self.dataset_list[index].split("/")
        X = torch.from_numpy(
            np.load(
                os.path.join(
                    preprocessed_dir, pdb_id_struct, "features_" + chain_id + ".npy"
                )
            )
        )
        y = torch.from_numpy(
            np.load(
                os.path.join(
                    preprocessed_dir, pdb_id_struct, "labels_" + chain_id + ".npy"
                )
            )
        )
        return X, y
