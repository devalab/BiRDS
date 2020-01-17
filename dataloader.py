from collections import defaultdict
from os import listdir, makedirs, path
from shutil import rmtree
from time import time

import numpy as np
import torch
from Bio.PDB import PDBParser, PPBuilder
from biopandas.mol2 import PandasMol2
from rdkit import Chem
from torch.utils.data import Dataset

from constants import AA_ID_DICT, DEVICE, THREE_TO_ONE, PROJECT_FOLDER
from datetime import timedelta

parser = PDBParser()
ppb = PPBuilder()
RCSB_SEQUENCES = path.join(PROJECT_FOLDER, "data/pdb_seqres.txt")


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
    def __init__(self, overwrite=False):
        super(scPDB, self).__init__()
        self.data_dir = path.join(PROJECT_FOLDER, "data/scPDB")
        self.raw_dir = path.join(self.data_dir, "raw")
        self.splits_dir = path.join(self.data_dir, "splits")
        self.save_dir = path.join(self.data_dir, "preprocessed")
        self.process_time = 0
        self.write_time = 0
        self.dataset = self.initialize_dataset_pdb_ids()
        self.blacklisted = self.get_blacklisted_proteins()
        # self.protein_sequences = self.get_sequences_from_rcsb()
        if overwrite:
            rmtree(self.save_dir)
        if not path.exists(self.save_dir):
            makedirs(self.save_dir)
        self.preprocess_pdb()

    def initialize_dataset_pdb_ids(self):
        available = defaultdict(set)
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
                if (
                    self.blacklisted[pdb_id_struct]
                    or pdb_id_struct == "3acl_1"
                    or pdb_id_struct == "3uf9_1"
                ):
                    continue
                print(pdb_id_struct)
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

    def preprocess_mol2(self):
        # available = np.unique([el[:4] for el in listdir(self.raw_dir)])
        # self.id_to_rcsb_seq = get_sequences_from_rcsb(available)
        # print(len(self.id_to_rcsb_seq))
        makedirs(self.save_dir)

        for i, pdb_id in enumerate(sorted(listdir(self.raw_dir))):
            pmol = PandasMol2().read_mol2(
                path.join(self.raw_dir, pdb_id, "protein.mol2")
            )
            lmol = PandasMol2().read_mol2(
                path.join(self.raw_dir, pdb_id, "ligand.mol2")
            )
            ligand_coords = lmol.df[lmol.df["atom_type"] != "H"][["x", "y", "z"]]
            protein_heavy = pmol.df[pmol.df["atom_type"] != "H"]
            binding_site = {}
            for j, atom_coord in enumerate(ligand_coords.values):
                pmol.df["distances"] = pmol.distance_df(protein_heavy, atom_coord)
                cutoff = pmol.df[pmol.df["distances"] <= 4.5]
                for k, aa in enumerate(cutoff.values):
                    binding_site[aa[7]] = aa[6]
            # sequence = self.id_to_rcsb_seq[pdb_id]
            # TODO Get the sequence here
            sequence = None
            length = len(sequence)
            labels = np.zeros(length)
            for key, val in binding_site.items():
                aa = THREE_TO_ONE[key[:3]]
                offset = int(key[3:]) - int(val) + 1
                pos = int(key[3:]) - offset
                assert pos < length
                assert sequence[pos] == aa
                labels[pos] = 1
            sequence = np.array([AA_ID_DICT[el] for el in sequence])
            np.savez(
                path.join(self.save_dir, pdb_id + ".npz"),
                sequence=sequence,
                length=length,
                labels=labels,
            )


class Kalasanty(scPDB):
    def __init__(self, folder, validation=False):
        super().__init__(folder)
        self.curr_fold = 0
        self.folds = []
        for i in range(10):
            with open(path.join(self.folder, "splits", "train_ids_fold" + str(i))) as f:
                self.folds.append(set([line.strip() for line in f.readlines()]))
        self.dataset = self.get_dataset()

    def get_dataset(self):
        all_samples = self.folds[0].union(self.folds[1])

        available = defaultdict(set)
        for file in listdir(path.join(self.folder, "raw")):
            available[file[:4]].add(file)

        with open(path.join(self.folder, "splits", "scPDB_blacklist.txt")) as f:
            for line in f.readlines():
                line = line.strip()
                available[line[:4]].remove(line)
                if available[line[:4]] == set():
                    del available[line[:4]]

        with open(path.join(self.folder, "splits", "scPDB_leakage.txt")) as f:
            for line in f.readlines():
                line = line.strip()
                available[line[:4]].remove(line)
                if available[line[:4]] == []:
                    del available[line[:4]]

        for key in set(available.keys()) - all_samples:
            del available[key]

        return available


if __name__ == "__main__":
    data = scPDB()
