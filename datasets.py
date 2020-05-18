# Assuming that preprocessing has been completed from preprocessing folder
# We define a dataset based on the preprocessed data
import os
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


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
    if max_len == len(samples[0][1]):
        for i, (tX, ty) in enumerate(samples):
            lengths[i] = len(ty)
            X[i, :, : lengths[i]] = tX
            y[i, : lengths[i]] = ty
    else:
        for i, (tX, ty) in enumerate(samples):
            lengths[i] = len(ty)
            X[i] = tX
            y[i, : lengths[i]] = ty
    return X, y, lengths


class Chen(Dataset):
    def __init__(self):
        super().__init__()
        self.data_dir = os.path.abspath("./data/chen")
        self.preprocessed_dir = os.path.join(self.data_dir, "preprocessed")
        self.dataset_list = self.get_dataset_list()
        self.pdb_id_to_index = defaultdict(int)
        for i, val in enumerate(self.dataset_list):
            self.pdb_id_to_index[val] = i

    def get_dataset_list(self):
        available = []
        for file in sorted(os.listdir(self.preprocessed_dir)):
            available.append(file)
        return sorted(available)

    def __getitem__(self, index):
        pdb_id = self.dataset_list[index]
        X = torch.from_numpy(
            np.load(os.path.join(self.preprocessed_dir, pdb_id, "features.npy"))
        )
        y = torch.from_numpy(
            np.load(os.path.join(self.preprocessed_dir, pdb_id, "labels.npy"))
        )
        return X, y

    def __len__(self):
        return len(self.dataset_list)


class Kalasanty(Dataset):
    def __init__(
        self, precompute_class_weights=False, fixed_length=False, get_input_size=True
    ):
        super().__init__()
        self.data_dir = os.path.abspath("./data/scPDB")
        self.splits_dir = os.path.join(self.data_dir, "splits")
        self.preprocessed_dir = os.path.join(self.data_dir, "preprocessed")
        self.pdb_id_to_pdb_id_struct = self.get_mapping()
        self.train_folds = self.get_train_folds()
        self.dataset_list = self.train_folds[0].union(self.train_folds[1])
        if fixed_length:
            self.remove_extras()
        self.valid_folds = self.make_valid_folds()
        self.dataset_list = sorted(list(self.dataset_list))
        self.pdb_id_struct_to_index = defaultdict(int)
        for i, val in enumerate(self.dataset_list):
            self.pdb_id_struct_to_index[val] = i
        if precompute_class_weights:
            # self.pos_weight = self.compute_class_weights()
            self.pos_weight = [6505272 / 475452]
        self.train_indices, self.valid_indices = self.custom_cv()
        if get_input_size:
            self.feat_vec_len = self[0][0].shape[0]

    def get_mapping(self):
        # There are multiple structures for a particular pdb_id, taking the first one
        available = defaultdict(str)
        for file in sorted(os.listdir(self.preprocessed_dir)):
            if file[:4] not in available:
                available[file[:4]] = file
        return available

    def get_train_folds(self):
        train_folds = []
        for i in range(10):
            with open(os.path.join(self.splits_dir, "train_ids_fold" + str(i))) as f:
                tmp = set()
                for line in f.readlines():
                    tmp.add(self.pdb_id_to_pdb_id_struct[line.strip()])
                train_folds.append(tmp)
        return train_folds

    def remove_extras(self):
        extras = set()
        for pdb_id_struct in tqdm(self.dataset_list, leave=False):
            if not os.path.exists(
                os.path.join(self.preprocessed_dir, pdb_id_struct, "features.npy")
            ):
                extras.add(pdb_id_struct)
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
        for pdb_id_struct in tqdm(self.dataset_list, leave=False):
            y = np.load(
                os.path.join(self.preprocessed_dir, pdb_id_struct, "labels.npy")
            )
            one = np.count_nonzero(y)
            ones += one
            zeros += len(y) - one
        pos_weight = [zeros / ones]
        print("Done")
        return pos_weight

    def custom_cv(self):
        train_indices = []
        valid_indices = []
        for i in range(10):
            train_indices.append(
                [self.pdb_id_struct_to_index[el] for el in self.train_folds[i]]
            )
            valid_indices.append(
                [self.pdb_id_struct_to_index[el] for el in self.valid_folds[i]]
            )
        return train_indices, valid_indices

    def __getitem__(self, index):
        pdb_id_struct = self.dataset_list[index]
        X = torch.from_numpy(
            np.load(os.path.join(self.preprocessed_dir, pdb_id_struct, "features.npy"))
        )
        y = torch.from_numpy(
            np.load(os.path.join(self.preprocessed_dir, pdb_id_struct, "labels.npy"))
        )
        return X, y

    def __len__(self):
        return len(self.dataset_list)


class KalasantyChains(Kalasanty):
    def __init__(self, precompute_class_weights=False, fixed_length=False):
        super().__init__(
            precompute_class_weights=precompute_class_weights, fixed_length=fixed_length
        )

    def get_train_folds(self):
        train_folds = []
        for i in range(10):
            with open(os.path.join(self.splits_dir, "train_ids_fold" + str(i))) as f:
                tmp = set()
                for line in f.readlines():
                    pdb_id_struct = self.pdb_id_to_pdb_id_struct[line.strip()]
                    for file in sorted(
                        os.listdir(os.path.join(self.preprocessed_dir, pdb_id_struct))
                    ):
                        if file.startswith("feat"):
                            tmp.add(pdb_id_struct + "/" + file[9:10])
                train_folds.append(tmp)
        return train_folds

    def __getitem__(self, index):
        pdb_id_struct, chain_id = self.dataset_list[index].split("/")
        X = torch.from_numpy(
            np.load(
                os.path.join(
                    self.preprocessed_dir,
                    pdb_id_struct,
                    "features_" + chain_id + ".npy",
                )
            )
        )
        y = torch.from_numpy(
            np.load(
                os.path.join(
                    self.preprocessed_dir, pdb_id_struct, "labels_" + chain_id + ".npy"
                )
            )
        )
        return X, y


def collate_fn_dict(samples):
    # samples is a list of (X, y) of size MINIBATCH_SIZE
    # Sort the samples in decreasing order of their length
    # x[1] will be y of each sample
    samples.sort(key=lambda x: len(x[1]["y"]), reverse=True)
    batch_size = len(samples)

    lengths = [0] * batch_size
    for i, (tX, ty) in enumerate(samples):
        lengths[i] = len(ty["y"])

    X = {}
    for key in samples[0][0].keys():
        feat_vec_len, max_len = samples[0][0][key].shape
        X[key] = torch.zeros(batch_size, feat_vec_len, max_len)
        for i, (tX, ty) in enumerate(samples):
            X[key][i, :, : lengths[i]] = tX[key]

    y = {}
    for key in samples[0][1].keys():
        max_len = samples[0][1][key].shape[0]
        y[key] = torch.zeros(batch_size, max_len)
        for i, (tX, ty) in enumerate(samples):
            y[key][i, : lengths[i]] = ty[key]
    return X, y, lengths


class KalasantyDict(Kalasanty):
    def __init__(
        self,
        precompute_class_weights=False,
        fixed_length=False,
        use_dist_map=False,
        use_pl_dist=False,
    ):
        self.use_dist_map = use_dist_map
        self.use_pl_dist = use_pl_dist
        super().__init__(
            precompute_class_weights=precompute_class_weights,
            fixed_length=fixed_length,
            get_input_size=False,
        )
        self.feat_vec_len = self[0][0]["X"].shape[0]

    def __getitem__(self, index):
        pdb_id_struct = self.dataset_list[index]
        X = {}
        y = {}
        X["X"] = torch.from_numpy(
            np.load(os.path.join(self.preprocessed_dir, pdb_id_struct, "features.npy"))
        )
        if self.use_dist_map:
            X["dist_map"] = torch.from_numpy(
                np.load(
                    os.path.join(self.preprocessed_dir, pdb_id_struct, "dist_map.npy")
                )
            )
        y["y"] = torch.from_numpy(
            np.load(os.path.join(self.preprocessed_dir, pdb_id_struct, "labels.npy"))
        )
        if self.use_pl_dist:
            y["pl_dist"] = torch.from_numpy(
                np.load(
                    os.path.join(self.preprocessed_dir, pdb_id_struct, "pl_dist.npy")
                )
            )
        return X, y
