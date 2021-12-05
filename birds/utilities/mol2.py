from collections import OrderedDict

import pandas as pd
from birds.utilities.NWalign import (
    calcualte_score_gotoh,
    code_with_modified_residues as aa3to1,
    trace_back_gotoh,
)


class Mol2:
    def __init__(self, path):
        """
        path: Path to the mol2 file to be read
        """
        self.path = path
        self.text = self.read_mol2()
        self.headers = self.get_headers()
        self.atom_df = self.get_atom_df()
        self.subst_df = self.get_subst_df()

    def read_mol2(self):
        """
        Returns: The full text of the mol2 file
        """
        with open(self.path, "r") as f:
            lines = [line.strip() for line in f.readlines()]
        return lines

    def get_records(self, records_type):
        """
        records_type: The type of record such as @<TRIPOS>MOLECULE, @<TRIPOS>ATOM, etc.

        Returns: The full text of the particual record
        """
        started = False
        for idx, line in enumerate(self.text):
            if line.startswith(records_type):
                first_idx = idx + 1
                started = True
            elif started and line.startswith("@<TRIPOS>"):
                last_idx = idx
                break
        # If we reach the end of the file, last_idx will not be set
        try:
            last_idx
        except NameError:
            last_idx = len(self.text)
        return self.text[first_idx:last_idx]

    def get_headers(self):
        """
        Returns: Headers contained in @<TRIPOS>MOLECULE
        """
        headers = {}
        records = self.get_records("@<TRIPOS>MOLECULE")
        headers["mol_name"] = records[0]
        (
            headers["num_atoms"],
            headers["num_bonds"],
            headers["num_subst"],
            headers["num_feat"],
            headers["num_sets"],
        ) = [int(el) for el in records[1].split()]
        headers["mol_type"] = records[2]
        headers["charge_type"] = records[3]
        return headers

    def convert_to_df(self, records, columns):
        """
        records: The text present in the particual record
        columns: The format in which each line is. It should be an OrderedDict with keys
                representing the name of the entry and the value representing the type
                of the entry

        Returns: A Pandas Dataframe that contains columns.keys() as the name of the columns
        """
        col_len = len(columns)
        df = []
        for line in records:
            line = line.split()
            df.append(line[:col_len] + [" ".join(line[col_len:])])
        # Anything extra at the end of the record will be stored as a comment
        columns["comment"] = str
        col_names = list(columns.keys())
        col_types = [columns[el] for el in col_names]
        df = pd.DataFrame(df, columns=col_names)
        for i in range(df.shape[1]):
            df[col_names[i]] = df[col_names[i]].astype(col_types[i])
        return df

    def get_atom_df(self):
        columns = [
            ("atom_id", int),
            ("atom_name", str),
            ("x", float),
            ("y", float),
            ("z", float),
            ("atom_type", str),
            ("subst_id", int),
            ("subst_name", str),
            ("charge", float),
        ]
        columns = OrderedDict(columns)
        records = self.get_records("@<TRIPOS>ATOM")
        df = self.convert_to_df(records, columns)
        # NOTE: Remove alternate locations of the same residue
        df = df.drop_duplicates(
            ["atom_name", "atom_type", "subst_id", "subst_name", "charge"]
        )
        return df

    def get_subst_df(self):
        columns = [
            ("subst_id", int),
            ("subst_name", str),
            ("root_atom", int),
            ("subst_type", str),
            ("dict_type", int),
            ("chain", str),
            ("sub_type", str),
            ("inter_bonds", int),
        ]
        columns = OrderedDict(columns)
        records = self.get_records("@<TRIPOS>SUBSTRUCTURE")
        return self.convert_to_df(records, columns)

    def to_fasta(self):
        sequences = {}
        # NOTE: atom_df works better than subst_df
        # Using subst_df for finding the sequence
        # for chain in self.subst_df["chain"].unique():
        #     seq = ""
        #     for aa in self.subst_df[self.subst_df["chain"] == chain]["sub_type"]:
        #         seq += aa3to1[aa] if aa in aa3to1 else "X"
        #     if len(set(seq)) > 1:
        #         sequences[chain] = seq

        # Let us use atom_df for finding CA atom and then consider them for the sequence
        df = self.atom_df[self.atom_df["atom_name"] == "CA"]
        df = df.merge(self.subst_df, "left", ["subst_id", "subst_name"])
        for chain in df["chain"].unique():
            seq = ""
            for aa in df[df["chain"] == chain]["sub_type"].values:
                if len(aa) != 3:
                    continue
                seq += aa3to1[aa] if aa in aa3to1 else "X"
            if seq != "" and set(seq) != set("X"):
                sequences[chain] = seq
        self.sequences = sequences
        return sequences

    @staticmethod
    def read_fasta(rcsb_fasta):
        sequences = {}
        with open(rcsb_fasta, "r") as f:
            header = f.readline()
            while 1:
                chain_id = header[6:7]
                sequence = ""
                line = f.readline()
                while line != "" and line is not None and line[0] != ">":
                    sequence += line.strip()
                    line = f.readline()
                sequences[chain_id] = sequence.replace("U", "X").replace("O", "X")
                if line == "" or line is None:
                    break
                header = line
        return sequences

    @staticmethod
    def align_sequences(mol2_sequences, rcsb_sequences):
        for chain in mol2_sequences:
            # idir (DP path); jpV (Horizontal jump number); jpH (Vertical jump number)
            idir, jpV, jpH = calcualte_score_gotoh(
                rcsb_sequences[chain], mol2_sequences[chain]
            )
            # sequenceA (aligned f1); sequenceB (aligned f2)
            rcsb_sequences[chain], mol2_sequences[chain] = trace_back_gotoh(
                idir, jpV, jpH, rcsb_sequences[chain], mol2_sequences[chain]
            )
        return mol2_sequences, rcsb_sequences

    def reindex(self, rcsb_fasta):
        """
        rcsb_fasta: The actual fasta file containing the full sequence of amino acids

        Creates a new ["reindex_id"] column in the subst_df dataframe
        """
        mol2_sequences = self.to_fasta()
        rcsb_sequences = self.read_fasta(rcsb_fasta)
        mol2_sequences, rcsb_sequences = self.align_sequences(
            mol2_sequences, rcsb_sequences
        )
        # Create a new column that contains the reindexed id
        self.subst_df["reindex_id"] = 0
        for chain in mol2_sequences:
            df_idxs = self.subst_df[self.subst_df["chain"] == chain].index.tolist()
            jdx = 0
            for idx, aa in enumerate(mol2_sequences[chain]):
                if aa == "-":
                    continue

                subst_id, sub_type = self.subst_df.iloc[df_idxs[jdx]][
                    ["subst_id", "sub_type"]
                ]
                # If our mol2 file has no amino acid with subst_id and "CA" atom
                # then go to the next amino acid
                while self.atom_df[
                    (self.atom_df["subst_id"] == subst_id)
                    & (self.atom_df["atom_name"] == "CA")
                ].empty:
                    jdx += 1
                    subst_id, sub_type = self.subst_df.iloc[df_idxs[jdx]][
                        ["subst_id", "sub_type"]
                    ]
                if sub_type in aa3to1:
                    assert aa3to1[sub_type] == aa
                elif len(sub_type) == 3:
                    assert aa == "X"
                else:
                    continue
                self.subst_df.at[df_idxs[jdx], "reindex_id"] = idx + 1
                jdx += 1

    @staticmethod
    def pdb_line(record, reindex_format):
        if reindex_format and record[4] == 0:
            return ""
        space = " "
        serial = str(record[0]).rjust(5)
        name = record[1][:4].center(4)
        altLoc = space
        resName = record[2][:3].rjust(3)
        chainID = record[3]
        resSeq = str(record[4]).rjust(4)
        iCode = space
        x = "%8.3f" % record[5]
        y = "%8.3f" % record[6]
        z = "%8.3f" % record[7]
        occupancy = "%6.2f" % 1.0
        tempFactor = "%6.2f" % 0.0
        element = record[1][0].rjust(2)
        charge = str(int(record[8]))[::-1].ljust(2)
        return (
            "ATOM  "
            + serial
            + space
            + name
            + altLoc
            + resName
            + space
            + chainID
            + resSeq
            + iCode
            + 3 * space
            + x
            + y
            + z
            + occupancy
            + tempFactor
            + 10 * space
            + element
            + charge
            + "\n"
        )

    def write_pdb(self, pdb_file, reindex_format=True):
        df = self.atom_df.merge(self.subst_df, "left", ["subst_id", "subst_name"])
        if "reindex_id" not in df.columns:
            df["reindex_id"] = df["subst_id"]
        pdb_text = [
            self.pdb_line(record, reindex_format)
            for record in df[
                [
                    "atom_id",
                    "atom_name",
                    "sub_type",
                    "chain",
                    "reindex_id",
                    "x",
                    "y",
                    "z",
                    "charge",
                ]
            ].values
        ]
        with open(pdb_file, "w") as f:
            f.writelines(pdb_text)
        return pdb_text
