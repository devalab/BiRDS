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


def fl_collate_fn(samples):
    samples.sort(key=lambda x: len(x[1]), reverse=True)
    batch_size = len(samples)
    lengths = [0] * batch_size
    feat_vec_len, fixed_len = samples[0][0].shape
    X = torch.zeros(batch_size, feat_vec_len, fixed_len)
    y = torch.zeros(batch_size, fixed_len)
    for i, sample in enumerate(samples):
        lengths[i] = len(sample[1])
        X[i] = sample[0]
        y[i, : lengths[i]] = sample[1]
    return X, y, lengths


class Kalasanty(Dataset):
    def __init__(self, precompute_class_weights=False, fixed_length=False):
        super().__init__()
        self.id_to_id_struct = self.get_mapping()
        self.train_folds = self.get_train_folds()
        self.dataset_list = self.train_folds[0].union(self.train_folds[1])
        self.fixed_length = fixed_length
        if fixed_length:
            self.remove_extras()
        self.valid_folds = self.make_valid_folds()
        self.dataset_list = sorted(list(self.dataset_list))
        self.pdb_id_to_index = defaultdict(int)
        for i, val in enumerate(self.dataset_list):
            self.pdb_id_to_index[val] = i
        if precompute_class_weights:
            # self.pos_weight = self.compute_class_weights()
            self.pos_weight = [6505272 / 475452]

    @staticmethod
    def get_mapping():
        available = defaultdict(list)
        for file in sorted(os.listdir(preprocessed_dir)):
            available[file[:4]].append(file)
        return available

    @staticmethod
    def get_train_folds():
        train_folds = []
        for i in range(10):
            with open(os.path.join(splits_dir, "train_ids_fold" + str(i))) as f:
                train_folds.append(set([line.strip() for line in f.readlines()]))
        return train_folds

    def remove_extras(self):
        extras = set()
        for i, pdb_id in tqdm(enumerate(self.dataset_list), leave=False):
            pdb_id_struct = self.id_to_id_struct[pdb_id][0]
            if not os.path.exists(
                os.path.join(preprocessed_dir, pdb_id_struct, "features.npy")
            ):
                extras.add(pdb_id)
        for i in range(10):
            self.train_folds[i] -= extras
        self.dataset_list -= extras

    def make_valid_folds(self):
        valid_folds = []
        for i in range(10):
            valid_folds.append(self.dataset_list - self.train_folds[i])
        return valid_folds

    def compute_class_weights(self):
        print("Precomputing class weights...")
        zeros = 0
        ones = 0
        # NOTE: Using just the first fold for now
        for i, pdb_id in tqdm(enumerate(self.train_folds[0]), leave=False):
            pdb_id_struct = self.id_to_id_struct[pdb_id][0]
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
            train_indices = [self.pdb_id_to_index[el] for el in self.train_folds[i]]
            valid_indices = [self.pdb_id_to_index[el] for el in self.valid_folds[i]]
            # yield train_indices[:24], valid_indices[:24]
            yield train_indices, valid_indices

    def __getitem__(self, index):
        pdb_id = self.dataset_list[index]
        # Just taking the first available structure for a pdb #TODO
        pdb_id_struct = self.id_to_id_struct[pdb_id][0]
        X = torch.from_numpy(
            np.load(os.path.join(preprocessed_dir, pdb_id_struct, "features.npy"))
        )
        y = torch.from_numpy(
            np.load(os.path.join(preprocessed_dir, pdb_id_struct, "labels.npy"))
        )
        return X, y


class KalasantyChains(Dataset):
    def __init__(self, precompute_class_weights=False, fixed_length=False):
        super().__init__()
        self.train_folds = []
        self.valid_folds = []
        self.id_to_id_struct = self.get_dataset()
        for i in range(10):
            with open(os.path.join(splits_dir, "train_ids_fold" + str(i))) as f:
                tmp = []
                for line in f.readlines():
                    tmp += self.id_to_id_struct[line.strip()]
                self.train_folds.append(tmp)
        self.dataset_list = set(self.train_folds[0]).union(set(self.train_folds[1]))
        for i in range(10):
            self.valid_folds.append(list(self.dataset_list - set(self.train_folds[i])))
        self.dataset_list = sorted(list(self.dataset_list))
        self.pdb_to_index = defaultdict(int)
        for i, val in enumerate(self.dataset_list):
            self.pdb_to_index[val] = i
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
        return chains

    def custom_cv(self):
        for i in range(10):
            train_indices = [self.pdb_to_index[el] for el in self.train_folds[i]]
            valid_indices = [self.pdb_to_index[el] for el in self.valid_folds[i]]
            #             yield train_indices[:24], valid_indices[:24]
            yield train_indices, valid_indices

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
