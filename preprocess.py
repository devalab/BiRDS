import logging
from datetime import timedelta
from os import makedirs, path
from shutil import rmtree
from time import time
from collections import defaultdict

import numpy as np
from Bio.PDB import PDBParser, PPBuilder
from rdkit import Chem

from constants import PROJECT_FOLDER

parser = PDBParser()
ppb = PPBuilder()
RCSB_SEQUENCES = path.join(PROJECT_FOLDER, "data/pdb_seqres.txt")


class PDBbindRefined:
    def __init__(self, overwrite=False, combine_same_protein=True):
        self.data_dir = path.join(PROJECT_FOLDER, "data/PDBbind")
        self.refined_dir = path.join(self.data_dir, "pdbbind_v2018_refined/refined-set")
        self.index_dir = path.join(self.data_dir, "PDBbind_2018_plain_text_index/index")
        self.index_file = path.join(self.index_dir, "INDEX_refined_data.2018")
        self.save_dir = path.join(self.data_dir, type(self).__name__ + "_preprocessed")
        self.process_time = 0
        self.write_time = 0
        self.dataset = self.initialize_dataset_from_index_file()
        self.protein_sequences = self.get_sequences_from_rcsb()
        self.overwrite = overwrite
        self.combine_same_protein = combine_same_protein

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
        logging.info(
            "Wrote output of %d proteins to %s folder",
            len(self.dataset),
            self.save_dir,
        )
        logging.info(
            "Total Process time %s | Write time %s",
            str(timedelta(seconds=self.process_time)),
            str(timedelta(seconds=self.write_time)),
        )

    def preprocess(self):
        if self.overwrite:
            rmtree(self.save_dir)
        if not path.exists(self.save_dir):
            makedirs(self.save_dir)
            for element in self.dataset:
                pdb_id = element[0]
                logging.info(pdb_id)

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
        else:
            logging.info("Preprocessed files already available in %s", self.save_dir)


if __name__ == "__main__":
    data = PDBbindRefined()
    data.preprocess()
