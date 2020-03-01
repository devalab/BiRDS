from collections import defaultdict
from datetime import timedelta
from os import listdir, makedirs, path
from shutil import rmtree
from time import time
import csv

import numpy as np
import torch
from Bio.PDB import PDBParser, PPBuilder

# from biopandas.mol2 import PandasMol2
from rdkit import Chem
from torch.utils.data import Dataset
from skorch import dataset

from constants import AA_ID_DICT, DEVICE, PROJECT_FOLDER

parser = PDBParser()
ppb = PPBuilder()
RCSB_SEQUENCES = path.join(PROJECT_FOLDER, "data/pdb_seqres.txt")


def get_features(csv_file):
    feats = {}
    with open(csv_file) as f:
        records = csv.reader(f)
        for i, row in enumerate(records):
            if i == 0:
                length = len(row) - 1
                continue
            feats[row[0]] = np.array(
                [float(el) if el != "" else 0.0 for el in row[1:]], dtype=np.float32
            )
        feats["X"] = np.zeros(length)
    return feats


AA_all_feats = get_features(path.join(PROJECT_FOLDER, "data/all_features.csv"))
AA_sel_feats = get_features(path.join(PROJECT_FOLDER, "data/selected_features.csv"))
feat_vec_len = 22
# feat_vec_len += len(AA_all_feats["X"])
feat_vec_len += len(AA_sel_feats["X"])


def generate_input(sample):
    """
    Generate input for a single sample which is a dictionary containing required items
    """
    X = {}
    X["length"] = sample["length"]
    X["X"] = torch.zeros(feat_vec_len, X["length"], device=DEVICE)

    # One-hot encoding
    sequence = np.array([np.eye(21)[AA_ID_DICT[el]] for el in sample["sequence"]]).T
    X["X"][:21] = torch.from_numpy(sequence)

    # Positional encoding
    X["X"][21] = torch.arange(1, X["length"] + 1, dtype=torch.float32) / X["length"]

    # AA Properties
    X["X"][22:] = torch.from_numpy(
        np.array([AA_sel_feats[aa] for aa in sample["sequence"]]).T
    )

    return X


def collate_fn(samples):
    # samples is a list of (X, y) of size MINIBATCH_SIZE
    # Sort the samples in decreasing order of their length
    samples.sort(key=lambda x: len(x[1]), reverse=True)
    batch_size = len(samples)
    max_len = samples[0][0]["length"]
    X = {}
    X["lengths"] = []
    X["X"] = torch.zeros(batch_size, feat_vec_len, max_len, device=DEVICE)
    y = torch.zeros(batch_size, max_len, device=DEVICE)
    for i in range(batch_size):
        length = samples[i][0]["length"]
        X["lengths"].append(length)
        X["X"][i, :, :length] = samples[i][0]["X"]
        y[i, :length] = samples[i][1]
    return X, y


class TupleDataset(dataset.Dataset):
    def __init__(self, X, y=None, length=None):
        if type(X[0]) is tuple and y is None:
            self.tuple_dataset = True
            self.y = [el[1] for el in X]
            self.X = [el[0] for el in X]
            self._len = len(self.y)
        else:
            self.tuple_dataset = False
            super().__init__(X, y=y, length=length)

    def __getitem__(self, i):
        if self.tuple_dataset:
            return self.X[i], self.y[i]
        else:
            return super().__getitem__(i)


class PDBbindRefined(Dataset):
    def __init__(self, overwrite=False):
        super(PDBbindRefined, self).__init__()
        self.data_dir = path.join(PROJECT_FOLDER, "data/PDBbind")
        self.save_dir = path.join(self.data_dir, type(self).__name__ + "_preprocessed")
        if overwrite:
            rmtree(self.save_dir)
        if not path.exists(self.save_dir):
            makedirs(self.save_dir)
            self.refined_dir = path.join(
                self.data_dir, "pdbbind_v2018_refined/refined-set"
            )
            self.index_dir = path.join(
                self.data_dir, "PDBbind_2018_plain_text_index/index"
            )
            self.index_file = path.join(self.index_dir, "INDEX_refined_data.2018")
            self.process_time = 0
            self.write_time = 0
            self.dataset = self.initialize_dataset_from_index_file()
            self.protein_sequences = self.get_sequences_from_rcsb()
        else:
            print("Using available preprocessed data")
        self.filenames = listdir(self.save_dir)

    def initialize_dataset_from_index_file(self):
        dataset = []
        with open(self.index_file) as f:
            line = f.readline()
            while line:
                if line[0] != "#":
                    dataset.append(line.strip().split())
                line = f.readline()
        return dataset

    def get_sequences_from_rcsb(self):
        sequences = defaultdict(str)
        with open(RCSB_SEQUENCES) as file:
            pdb_id = file.readline()[1:5]
            for data in sorted(self.dataset):
                while pdb_id != data[0]:
                    file.readline()
                    pdb_id = file.readline()[1:5]
                # Each id can have multiple chains
                while pdb_id == data[0]:
                    seq = file.readline().strip()
                    sequences[pdb_id] += seq
                    pdb_id = file.readline()[1:5]
        print(len(sequences))
        return sequences

    def init_complex_info(self, pdb_id):
        pdb_prefix = path.join(self.refined_dir, pdb_id) + pdb_id + "_"

        self.protein_structure = parser.get_structure(
            pdb_id, pdb_prefix + "protein.pdb"
        )
        self.protein_residues = []
        # Including non-standard amino acids
        # The letter in the sequence will be their standard amino acid counterpart
        # For example, MSE (Selenomethionines) are shown as M (Methionine)
        for seq in ppb.build_peptides(self.protein_structure, aa_only=False):
            for res in seq:
                self.residues.append(res)
        self.sequence = self.get_sequence_from_rcsb(pdb_id)

        self.ligand_supplier = Chem.SDMolSupplier(
            pdb_prefix + "ligand.sdf", sanitize=False
        )
        assert len(self.ligand_supplier) == 1
        self.ligand_supplier = self.ligand_supplier[0]
        assert self.ligand_supplier.GetNumConformers() == 1
        self.ligand_coords = self.ligand_supplier.GetConformer().GetPositions()
        self.ligand_num_atoms = self.ligand_supplier.GetNumAtoms()
        assert self.ligand_num_atoms == len(self.ligand_coords)
        self.ligand_atom_types = np.array(
            [atom.GetSymbol() for atom in self.ligand_supplier.GetAtoms()]
        )

    def get_binding_affinity(self, string):
        prefixes = {"m": 1e-3, "u": 1e-6, "n": 1e-9, "p": 1e-12, "f": 1e-15}
        for i in prefixes.keys():
            if i in string:
                prefix = i
                break
        value = float(string.split("=")[1].split(prefix)[0]) * prefixes[prefix]
        return value

    def get_sequence_from_structure(self):
        sequences = [
            str(seq.get_sequence())
            for seq in ppb.build_peptides(self.protein_structure, aa_only=False)
        ]
        return "".join(sequences)

    def find_residues_in_contact(self):
        """
        Returns a numpy 1D array where a 1 represents that the amino acid is in
        contact with the ligand
        """
        labels = np.zeros(len(self.sequence))
        for ind, residue in enumerate(self.residues):
            for atom in residue.get_atoms():
                if atom.get_fullname()[1] == "H":
                    continue
                for i in range(self.ligand_num_atoms):
                    if self.ligand_atom_types[i] == "H":
                        continue
                    # We are considering the ligand to be in contact with the AA
                    # if the distance between them is within 4.5A
                    if np.linalg.norm(atom.get_coord() - self.ligand_coords[i]) < 4.5:
                        labels[ind] = 1
                        break
                # We know that the residue is in contact with ligand
                # So go to the next residue
                if labels[ind]:
                    break
        return labels

    def get_distance_map(self):
        pass

    def make_dictionary_for_storage(self, data):
        metadata = {}
        metadata["pdb_id"] = data[0]
        metadata["resolution"] = float(data[1])
        metadata["release_year"] = int(data[2])
        metadata["neg_log_k"] = float(data[3])
        metadata["k"] = self.get_binding_affinity(data[4])
        metadata["ligand_name"] = data[7].split("(")[1].split(")")[0]

        protein = {}
        protein["sequence"] = self.protein_sequences[data[0]]
        protein["length"] = len(protein["sequence"])
        protein["labels"] = self.find_residues_in_contact()
        protein["distance_map"] = self.get_distance_map()

        assert len(protein["sequence"]) == len(protein["labels"])
        return metadata, protein

    def log_info(self):
        print(
            "Wrote output of %d proteins to %s folder"
            % (len(self.dataset), self.save_dir)
        )
        print(
            "Total Process time %s | Write time %s"
            % (
                str(timedelta(seconds=self.process_time)),
                str(timedelta(seconds=self.write_time)),
            )
        )

    def preprocess(self):
        for element in sorted(self.dataset):
            pdb_id = element[0]
            print(pdb_id)

            process_time_start = time()
            self.init_complex_info(pdb_id)
            metadata, protein = self.make_dictionary_for_storage(element)
            self.process_time += time() - process_time_start

            write_time_start = time()
            np.savez(
                self.save_dir + pdb_id + ".npz", metadata=metadata, protein=protein,
            )
            self.write_time += time() - write_time_start
        self.log_info()

    def __getitem__(self, index):
        data = np.load(
            path.join(self.save_dir, self.filenames[index]), allow_pickle=True
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


class scPDB(Dataset):
    def __init__(self, overwrite=False, continue_processing=False):
        # overwrite: If true, the whole preprocessed data will be overwritten
        # continue_preprocessing: Continue preprocessing data from where it stopped
        super(scPDB, self).__init__()
        self.data_dir = path.join(PROJECT_FOLDER, "data/scPDB")
        self.raw_dir = path.join(self.data_dir, "raw")
        self.splits_dir = path.join(self.data_dir, "splits")
        self.save_dir = path.join(self.data_dir, "preprocessed")
        self.process_time = 0
        self.write_time = 0
        # self.protein_sequences = self.get_sequences_from_rcsb()
        if overwrite:
            rmtree(self.save_dir)
        if not path.exists(self.save_dir):
            makedirs(self.save_dir)
            continue_processing = True
        if continue_processing:
            self.dataset = self.initialize_dataset_pdb_ids(self.raw_dir)
            self.blacklisted = self.get_blacklisted_proteins()
            self.preprocess_pdb()

    def initialize_dataset_pdb_ids(self):
        available = defaultdict(list)
        for file in listdir(self.raw_dir):
            available[file[:4]].add(file)
        return available

    def get_blacklisted_proteins(self):
        blacklisted = defaultdict(bool)
        with open(path.join(self.splits_dir, "scPDB_blacklist.txt")) as f:
            for line in f.readlines():
                blacklisted[line.strip()] = True
        return blacklisted

    # def get_sequences_from_rcsb(self):
    #     sequences = defaultdict(str)
    #     with open(RCSB_SEQUENCES) as file:
    #         pdb_id = file.readline()[1:5]
    #         for data in sorted(self.dataset):
    #             while pdb_id != data[0]:
    #                 file.readline()
    #                 pdb_id = file.readline()[1:5]
    #             # Each id can have multiple chains
    #             while pdb_id == data[0]:
    #                 seq = file.readline().strip()
    #                 sequences[pdb_id] += seq
    #                 pdb_id = file.readline()[1:5]
    #     print(len(sequences))
    #     return sequences

    def init_complex_info(self, pdb_id_struct):
        pdb_prefix = path.join(self.raw_dir, pdb_id_struct)
        self.protein_structure = parser.get_structure(
            pdb_id_struct, path.join(pdb_prefix, "converted_protein.pdb")
        )
        # Including non-standard amino acids
        # The letter in the sequence will be their standard amino acid counterpart
        # For example, MSE (Selenomethionines) are shown as M (Methionine)
        self.sequence = ""
        self.residues = []
        for seq in ppb.build_peptides(self.protein_structure, aa_only=False):
            self.sequence += str(seq.get_sequence())
            for res in seq:
                self.residues.append(res)
        # self.sequence = self.get_sequence_from_rcsb(pdb_id_struct)

        self.ligand_supplier = Chem.SDMolSupplier(
            path.join(pdb_prefix, "ligand.sdf"), sanitize=False
        )
        assert len(self.ligand_supplier) == 1
        self.ligand_supplier = self.ligand_supplier[0]
        assert self.ligand_supplier.GetNumConformers() == 1
        self.ligand_coords = self.ligand_supplier.GetConformer().GetPositions()
        self.ligand_num_atoms = self.ligand_supplier.GetNumAtoms()
        assert self.ligand_num_atoms == len(self.ligand_coords)
        self.ligand_atom_types = np.array(
            [atom.GetSymbol() for atom in self.ligand_supplier.GetAtoms()]
        )

    def find_residues_in_contact(self):
        """
        Returns a numpy 1D array where a 1 represents that the amino acid is in
        contact with the ligand
        """
        labels = np.zeros(len(self.sequence))
        for ind, residue in enumerate(self.residues):
            for atom in residue.get_atoms():
                if atom.get_fullname()[1] == "H":
                    continue
                for i in range(self.ligand_num_atoms):
                    if self.ligand_atom_types[i] == "H":
                        continue
                    # We are considering the ligand to be in contact with the AA
                    # if the distance between them is within 4.5A
                    if np.linalg.norm(atom.get_coord() - self.ligand_coords[i]) <= 4.5:
                        labels[ind] = 1
                        break
                # We know that the residue is in contact with ligand
                # So go to the next residue
                if labels[ind]:
                    break
        return labels

    def make_dictionary_for_storage(self, pdb_id):
        data = {}
        data["pdb_id"] = pdb_id
        data["sequence"] = self.sequence
        data["length"] = len(data["sequence"])
        data["labels"] = self.find_residues_in_contact()

        assert len(data["sequence"]) == len(data["labels"])
        return data

    def log_info(self):
        print(
            "Wrote output of %d proteins to %s folder"
            % (len(self.dataset), self.save_dir)
        )
        print(
            "Total Process time %s | Write time %s"
            % (
                str(timedelta(seconds=self.process_time)),
                str(timedelta(seconds=self.write_time)),
            )
        )

    def preprocess_pdb(self):
        for pdb_id in sorted(self.dataset):
            for pdb_id_struct in self.dataset[pdb_id]:
                if not path.exists(
                    path.join(self.raw_dir, pdb_id_struct, "converted_protein.pdb")
                ):
                    print("Converted PDB does not exist for %s" % pdb_id_struct)
                    continue
                if self.blacklisted[pdb_id_struct]:
                    continue
                # print(pdb_id_struct)
                if path.exists(path.join(self.save_dir, pdb_id_struct + ".npz")):
                    continue
                process_time_start = time()
                self.init_complex_info(pdb_id_struct)
                data = self.make_dictionary_for_storage(pdb_id_struct)
                self.process_time += time() - process_time_start

                write_time_start = time()
                np.savez(path.join(self.save_dir, pdb_id_struct + ".npz"), **data)
                self.write_time += time() - write_time_start
        self.log_info()

    # def preprocess_mol2(self):
    #     # available = np.unique([el[:4] for el in listdir(self.raw_dir)])
    #     # self.id_to_rcsb_seq = get_sequences_from_rcsb(available)
    #     # print(len(self.id_to_rcsb_seq))
    #     makedirs(self.save_dir)

    #     for i, pdb_id in enumerate(sorted(listdir(self.raw_dir))):
    #         pmol = PandasMol2().read_mol2(
    #             path.join(self.raw_dir, pdb_id, "protein.mol2")
    #         )
    #         lmol = PandasMol2().read_mol2(
    #             path.join(self.raw_dir, pdb_id, "ligand.mol2")
    #         )
    #         ligand_coords = lmol.df[lmol.df["atom_type"] != "H"][["x", "y", "z"]]
    #         protein_heavy = pmol.df[pmol.df["atom_type"] != "H"]
    #         binding_site = {}
    #         for j, atom_coord in enumerate(ligand_coords.values):
    #             pmol.df["distances"] = pmol.distance_df(protein_heavy, atom_coord)
    #             cutoff = pmol.df[pmol.df["distances"] <= 4.5]
    #             for k, aa in enumerate(cutoff.values):
    #                 binding_site[aa[7]] = aa[6]
    #         # sequence = self.id_to_rcsb_seq[pdb_id]
    #         # TODO Get the sequence here
    #         sequence = None
    #         length = len(sequence)
    #         labels = np.zeros(length)
    #         for key, val in binding_site.items():
    #             aa = THREE_TO_ONE[key[:3]]
    #             offset = int(key[3:]) - int(val) + 1
    #             pos = int(key[3:]) - offset
    #             assert pos < length
    #             assert sequence[pos] == aa
    #             labels[pos] = 1
    #         sequence = np.array([AA_ID_DICT[el] for el in sequence])
    #         np.savez(
    #             path.join(self.save_dir, pdb_id + ".npz"),
    #             sequence=sequence,
    #             length=length,
    #             labels=labels,
    #         )


class Kalasanty(scPDB):
    def __init__(self):
        super().__init__()
        self.train_folds = []
        self.valid_folds = []
        for i in range(10):
            with open(path.join(self.splits_dir, "train_ids_fold" + str(i))) as f:
                self.train_folds.append([line.strip() for line in f.readlines()])
        self.dataset_list = set(self.train_folds[0]).union(set(self.train_folds[1]))
        for i in range(10):
            self.valid_folds.append(list(self.dataset_list - set(self.train_folds[i])))
        self.dataset = self.get_dataset()
        self.dataset_list = sorted(list(self.dataset_list))
        self.dataset_id_to_index = defaultdict(int)
        for i, val in enumerate(self.dataset_list):
            self.dataset_id_to_index[val] = i

    def get_dataset(self):
        available = defaultdict(list)
        for file in listdir(self.save_dir):
            available[file[:4]].append(file[:-4])

        extras = ["scPDB_blacklist.txt", "scPDB_leakage.txt"]
        for file in extras:
            with open(path.join(self.splits_dir, file)) as f:
                for line in f.readlines():
                    line = line.strip()
                    if line in available[line[:4]]:
                        available[line[:4]].remove(line)
                    if available[line[:4]] == list():
                        del available[line[:4]]

        for key in set(available.keys()) - self.dataset_list:
            del available[key]

        return available

    def custom_cv(self):
        for i in range(10):
            train_indices = [self.dataset_id_to_index[el] for el in self.train_folds[i]]
            valid_indices = [self.dataset_id_to_index[el] for el in self.valid_folds[i]]
            # yield train_indices[:100], valid_indices[:12]
            yield train_indices, valid_indices

    def __getitem__(self, index):
        pdb_id = self.dataset_list[index]
        # Just taking the first available structure for a pdb
        pdb_id_struct = self.dataset[pdb_id][0]
        sample = np.load(path.join(self.save_dir, pdb_id_struct + ".npz"))
        sample = {
            key: sample[key].item() if sample[key].shape is () else sample[key]
            for key in sample
        }
        X = generate_input(sample)
        y = torch.from_numpy(sample["labels"]).to(DEVICE)
        return X, y

    def __len__(self):
        return len(self.dataset_list)


if __name__ == "__main__":
    data = Kalasanty()
