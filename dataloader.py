from os import listdir, path

import numpy as np
import torch

# import torch.nn as nn
from torch.utils.data import Dataset

from constants import AA_ID_DICT, DEVICE


def generate_input(data):
    """
    Generate input for each minibatch. Pad the input feature vectors
    so that the final input shape is [MINIBATCH_SIZE, 2, Max_length]
    """
    lengths = data["length"]
    sequences = data["sequence"]
    batch_size = len(lengths)
    transformed_sequence = torch.zeros(
        batch_size, 2, lengths[0], device=DEVICE, dtype=torch.float32
    )
    for i in range(batch_size):
        transformed_sequence[i, 0, : lengths[i]] = torch.from_numpy(sequences[i])
        transformed_sequence[i, 1, : lengths[i]] = (
            torch.arange(0, lengths[i], dtype=torch.float32) / lengths[i]
        )

    return transformed_sequence


def generate_target(data):
    lengths = data["length"]
    labels = data["labels"]
    batch_size = len(lengths)
    target = torch.zeros(batch_size, lengths[0], device=DEVICE, dtype=torch.float32)
    for i in range(batch_size):
        target[i, : lengths[i]] = torch.from_numpy(labels[i])
    return target


def collate_fn(samples):
    # samples is a list of dictionaries and is of size MINIBATCH_SIZE
    # The dicts are the one returned from __getitem__ function
    # Sort the samples in decreasing order of their length
    samples.sort(key=lambda x: x["length"], reverse=True)
    # Convert a list of dictionaries into a dictionary of lists
    batch = {k: [dic[k] for dic in samples] for k in samples[0]}
    X = {}
    X["X"] = generate_input(batch)
    X["lengths"] = batch["length"]
    # print(batch["pdb_id"][0])
    y = generate_target(batch)
    return X, y


class PDBbind(Dataset):
    def __init__(self, foldername):
        super(PDBbind, self).__init__()
        self.foldername = foldername
        self.filenames = listdir(foldername)

    def __getitem__(self, index):
        data = np.load(
            path.join(self.foldername, self.filenames[index]), allow_pickle=True
        )
        sample = data["protein"].item()
        sample.update(data["metadata"].item())
        sample["sequence"] = np.array([AA_ID_DICT[el] for el in sample["sequence"]])
        return sample

    def __len__(self):
        return len(self.filenames)


class DeepCSeqSite(Dataset):
    def __init__(self, filename):
        super(DeepCSeqSite, self).__init__()
        self.data = []
        with open(filename, "r") as file:
            lines = file.readlines()
        for i, line in enumerate(lines):
            line = line.strip()
            if i % 4 == 0:
                uniprot_id = line
            elif i % 4 == 1:
                neg_log_k = float(line)
            elif i % 4 == 2:
                sequence = np.array([AA_ID_DICT[el] for el in line])
            else:
                labels = np.array([int(num) for num in line])
                self.data.append(
                    {
                        "labels": labels,
                        "sequence": sequence,
                        "pdb_id": uniprot_id,
                        "neg_log_k": neg_log_k,
                        "length": len(sequence),
                    }
                )

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
