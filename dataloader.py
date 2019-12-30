from os import listdir

import numpy as np
import torch
from torch.utils.data import Dataset

from constants import AA_ID_DICT, DEVICE


class PDBbind(Dataset):
    def __init__(self, foldername, data_needed):
        super(PDBbind, self).__init__()
        self.foldername = foldername
        self.data_needed = data_needed
        self.filenames = listdir(foldername)

    def __getitem__(self, index):
        data = np.load(self.foldername + self.filenames[index], allow_pickle=True)
        protein = data["protein"].item()
        protein.update(data["metadata"].item())
        sample = dict((k, protein[k]) for k in self.data_needed if k in protein)
        return sample

    def __len__(self):
        return len(self.filenames)


def generate_input(data):
    """
    Generate input for each minibatch. Pad the input feature vectors
    so that the final input shape is [MINIBATCH_SIZE, 21, Max_length]
    """
    lengths = data["length"]
    sequence = data["sequence"]
    batch_size = len(lengths)
    transformed_sequence = torch.zeros(
        batch_size, 21, lengths[0], device=DEVICE, dtype=torch.float32
    )

    # TODO: Use pythonic way
    for i in range(batch_size):
        for j in range(lengths[i]):
            residue = AA_ID_DICT[sequence[i][j]]
            transformed_sequence[i][residue][j] = 1.0

    return transformed_sequence


def generate_target(data):
    lengths = data["length"]
    labels = data["labels"]
    batch_size = len(lengths)
    target = torch.zeros(batch_size, lengths[0], device=DEVICE, dtype=torch.float32)
    for i in range(batch_size):
        target[i, : lengths[i]] = torch.from_numpy(labels[i])
    return target


def PDBbind_collate_fn(samples):
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
