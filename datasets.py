# Assuming that preprocessing has been completed from preprocessing folder
# We define a dataset based on the preprocessed data
import os
from collections import defaultdict
from glob import glob

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

AMINO_ACIDS = "XACDEFGHIKLMNPQRSTVWY"
AA_DICT = defaultdict(lambda: 0, {aa: idx for idx, aa in enumerate(AMINO_ACIDS)})


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
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.data_dir = os.path.abspath("./data/scPDB")
        self.preprocessed_dir = os.path.join(self.data_dir, "preprocessed")
        self.splits_dir = os.path.join(self.data_dir, "splits")
        self.fold = str(hparams.fold)

        # Get features and labels that are small in size
        self.sequences, self.labels = self.get_sequences_and_labels()
        self.aap = self.get_npy("aap", hparams.use_aap)
        self.pssm = self.get_npy("pssm", hparams.use_pssm)
        self.ss2 = self.get_npy("ss2", hparams.use_ss2)
        self.solv = self.get_npy("solv", hparams.use_solv)

        self.available_data = sorted(list(self.labels.keys()))
        self.train_fold, self.valid_fold = self.get_fold()
        self.dataset = sorted(self.train_fold + self.valid_fold)

        # ------------MAPPINGS------------
        # pdbID to pdbID_structure mapping
        self.pi_to_pis = defaultdict(list)
        for key in self.available_data:
            pis = key.split("/")[0]
            if pis not in self.pi_to_pis[key[:4]]:
                self.pi_to_pis[key[:4]].append(pis)

        # pdbID_structure TO pdbID_structure/chain mapping
        self.pis_to_pisc = defaultdict(list)
        for key in self.available_data:
            self.pis_to_pisc[key.split("/")[0]].append(key)

        # pdbID_structure/chain to available MSA pdbID_structure/chain sequence
        self.pisc_to_mpisc = {}
        with open(os.path.join(self.data_dir, "unique"), "r") as f:
            for line in f.readlines():
                line = line.strip().split()
                mpisc = line[0][:-1]
                self.pisc_to_mpisc[mpisc] = mpisc
                for pisc in line[1:]:
                    self.pisc_to_mpisc[pisc[:-1]] = mpisc

        # Dataset pdbID to index mapping
        self.pi_to_index = {val: key for key, val in enumerate(self.dataset)}

        if hparams.compute_pos_weight:
            # self.pos_weight = self.compute_pos_weight()
            self.pos_weight = [6507812 / 475440]

        self.train_indices = [self.pi_to_index[pi] for pi in self.train_fold]
        self.valid_indices = [self.pi_to_index[pi] for pi in self.valid_fold]
        self.input_size = self[0][0]["X"].shape[0]

    def get_sequences_and_labels(self):
        sequences = {}
        labels = {}
        with open(os.path.join(self.data_dir, "info.txt")) as f:
            f.readline()
            line = f.readline()
            while line != "":
                line = line.strip().split("\t")
                key = line[0] + "_" + line[1] + "/" + line[2]
                sequences[key] = line[3]
                labels[key] = np.array([True if aa == "1" else False for aa in line[4]])
                line = f.readline()
        return sequences, labels

    def get_npy(self, name, flag=True):
        if not flag:
            return None
        mapping = {}
        print("Loading", name)
        for file in tqdm(
            sorted(glob(os.path.join(self.preprocessed_dir, "*", name + "_?.npy")))
        ):
            pis, chain = file.split("/")[-2:]
            chain = chain[-5:-4]
            mapping[pis + "/" + chain] = np.load(file)
        return mapping

    def get_fold(self):
        with open(os.path.join(self.splits_dir, "train_ids_fold" + self.fold)) as f:
            train = sorted([line.strip() for line in f.readlines()])
        with open(os.path.join(self.splits_dir, "test_ids_fold" + self.fold)) as f:
            valid = sorted([line.strip() for line in f.readlines()])
        return train, valid

    def compute_pos_weight(self):
        print("Computing positional weights...")
        zeros = 0
        ones = 0
        for pi in tqdm(self.train_fold, leave=False):
            pis = self.pi_to_pis[pi][0]
            for pisc in self.pis_to_pisc[pis]:
                try:
                    y = self.labels[pisc]
                except KeyError:
                    print(pi, pisc)
                one = np.count_nonzero(y)
                ones += one
                zeros += len(y) - one
        pos_weight = [zeros / ones]
        print(zeros, ones, "Done")
        return pos_weight

    def __getitem__(self, index):
        pi = self.dataset[index]
        # Taking the first structure available
        pis = self.pi_to_pis[pi][0]
        # For all available chains
        X = {}
        y = {}
        for i, pisc in enumerate(self.pis_to_pisc[pis]):
            mpisc = self.pisc_to_mpisc[pisc]
            sequence = self.sequences[pisc]
            if self.hparams.use_tape_embeddings:
                tape = np.load(
                    os.path.join(
                        self.preprocessed_dir, mpisc[:-2], "tape_" + mpisc[-1] + ".npy"
                    )
                )
            oh = np.array([np.eye(len(AMINO_ACIDS))[AA_DICT[aa]] for aa in sequence]).T
            pe = np.arange(1, len(sequence) + 1).reshape((1, -1)) / len(sequence)
            inputs = [oh, pe]
            if self.hparams.use_aap:
                inputs.append(self.aap[mpisc])
            if self.hparams.use_pssm:
                inputs.append(self.pssm[mpisc])
            if self.hparams.use_ss2:
                inputs.append(self.ss2[mpisc])
            if self.hparams.use_solv:
                inputs.append(self.solv[mpisc])
            feature = np.vstack(inputs)
            label = self.labels[pisc]
            # Add 0s at start and 1s at end to give an idea of different chains
            ln = feature.shape[0]
            feature = np.hstack((np.zeros((ln, 1)), feature, np.ones((ln, 1)))).astype(
                np.float32
            )
            label = np.hstack(([0], label, [0])).astype(np.float32)
            if i == 0:
                if self.hparams.use_tape_embeddings:
                    X["seq"] = tape
                X["X"] = feature
                y["y"] = label
            else:
                if self.hparams.use_tape_embeddings:
                    X["seq"] = np.hstack((X["seq"], tape))
                X["X"] = np.hstack((X["X"], feature))
                y["y"] = np.hstack((y["y"], label))
        return X, y

    def __len__(self):
        return len(self.dataset_list)

    @staticmethod
    # A collate function to merge samples into a minibatch, will be used by DataLoader
    def collate_fn(samples):
        # samples is a list of (X, y) of size MINIBATCH_SIZE
        # Sort the samples in decreasing order of their length
        # x[1] will be y of each sample
        samples.sort(key=lambda x: len(x[1]["y"]), reverse=True)
        batch_size = len(samples)

        lengths = [0] * batch_size
        for i, (tX, ty) in enumerate(samples):
            lengths[i] = len(ty["y"])

        X = {}
        fX = samples[0][0]
        for key in fX.keys():
            X[key] = np.zeros((batch_size, *fX[key].shape), dtype=fX[key].dtype)
            for i, (tX, ty) in enumerate(samples):
                X[key][i, ..., : lengths[i]] = tX[key]
            X[key] = torch.from_numpy(X[key])

        y = {}
        fy = samples[0][1]
        for key in fy.keys():
            y[key] = np.zeros((batch_size, *fy[key].shape), dtype=fy[key].dtype)
            for i, (tX, ty) in enumerate(samples):
                y[key][i, ..., : lengths[i]] = ty[key]
            y[key] = torch.from_numpy(y[key])
        return X, y, lengths

    @staticmethod
    def add_class_specific_args(parser):
        parser.add_argument(
            "--fold",
            metavar="NUMBER",
            type=int,
            default=0,
            help="Cross Validation fold number to train on. Default: %(default)d",
        )
        parser.add_argument(
            "--pos-weight",
            dest="compute_pos_weight",
            action="store_true",
            help="Compute the positional weight of the binding residue class. Default: %(default)s",
        )
        parser.add_argument(
            "--no-pos-weight", dest="compute_pos_weight", action="store_false",
        )
        parser.set_defaults(compute_pos_weight=True)

        parser.add_argument(
            "--fixed-length",
            metavar="LENGTH",
            type=int,
            help="Use a fixed length input instead of variable lengths",
        )

        parser.add_argument(
            "--amino-acid-probabilities",
            dest="use_aap",
            action="store_true",
            help="Use amino acid probabilities in input features for the model. Default: %(default)s",
        )
        parser.add_argument(
            "--no-amino-acid-probabilities", dest="use_aap", action="store_false"
        )
        parser.set_defaults(use_aap=False)

        parser.add_argument(
            "--pssm",
            dest="use_pssm",
            action="store_true",
            help="Use Position Specific Scoring Matrix in input features for the model. Default: %(default)s",
        )
        parser.add_argument("--no-pssm", dest="use_pssm", action="store_false")
        parser.set_defaults(use_pssm=True)

        parser.add_argument(
            "--secondary-structure",
            dest="use_ss2",
            action="store_true",
            help="Use predicted secondary structure in input features for the model. Default: %(default)s",
        )
        parser.add_argument(
            "--no-secondary-structure", dest="use_ss2", action="store_false"
        )
        parser.set_defaults(use_ss2=True)

        parser.add_argument(
            "--solvent-accessibility",
            dest="use_solv",
            action="store_true",
            help="Use predicted solvent accessibilities in input features for the model. Default: %(default)s",
        )
        parser.add_argument(
            "--no-solvent-accessibility", dest="use_solv", action="store_false"
        )
        parser.set_defaults(use_solv=True)

        parser.add_argument(
            "--protein-ligand-distance",
            dest="use_pl_dist",
            action="store_true",
            help="Use distance between amino acid residues and ligand to improve loss function. Default: %(default)s",
        )
        parser.add_argument(
            "--no-protein-ligand-distance", dest="use_pl_dist", action="store_false"
        )
        parser.set_defaults(use_pl_dist=False)

        parser.add_argument(
            "--tape-embeddings",
            dest="use_tape_embeddings",
            action="store_true",
            help="Use a BERT model on sequence to understand the language of proteins. Default: %(default)s",
        )
        parser.add_argument(
            "--no-tape-embeddings", dest="use_tape_embeddings", action="store_false"
        )
        parser.set_defaults(use_tape_embeddings=False)

        return parser
