import pandas as pd
from collections import OrderedDict
from NWalign import (
    code_with_modified_residues as aa3to1,
    calcualte_score_gotoh,
    trace_back_gotoh,
)


class Mol2:
    def __init__(self, path):
        self.path = path
        self.text = self.read_mol2()
        self.headers = self.get_headers()
        self.atom_df = self.get_atom_df()
        self.subst_df = self.get_subst_df()

    def read_mol2(self):
        with open(self.path, "r") as f:
            lines = [line.strip() for line in f.readlines()]
        return lines

    def get_records(self, records_type):
        started = False
        for idx, line in enumerate(self.text):
            if line.startswith(records_type):
                first_idx = idx + 1
                started = True
            elif started and line.startswith("@<TRIPOS>"):
                last_idx = idx
                break
        return self.text[first_idx:last_idx]

    def get_headers(self):
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
        col_len = len(columns)
        df = []
        for line in records:
            line = line.split()
            df.append(line[:col_len] + [" ".join(line[col_len:])])
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
        # NOTE: Remove alternate locations of the same residue
        df = self.convert_to_df(records, columns)
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
        # Using subst_df for finding the sequence
        # for chain in self.subst_df["chain"].unique():
        #     seq = ""
        #     for aa in self.subst_df[self.subst_df["chain"] == chain]["sub_type"]:
        #         seq += aa3to1[aa] if aa in aa3to1 else "X"
        #     if len(set(seq)) != 1:
        #         sequences[chain] = seq

        # Let us use atom_df for finding CA atom and then consider them for the sequence
        df = self.atom_df[self.atom_df["atom_name"] == "CA"]
        df = df.merge(self.subst_df, "left", ["subst_id", "subst_name"])
        for chain in df["chain"].unique():
            seq = ""
            for aa in df[df["chain"] == chain]["sub_type"].values:
                seq += aa3to1[aa] if aa in aa3to1 else "X"
            if len(set(seq)) != 1:
                sequences[chain] = seq
        self.sequences = sequences
        return sequences

    def reindex(self, rcsb_fasta):
        def read_fasta():
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

        def align_sequences():
            for chain in mol2_sequences:
                # idir (DP path); jpV (Horizontal jump number); jpH (Vertical jump number)
                idir, jpV, jpH = calcualte_score_gotoh(
                    rcsb_sequences[chain], mol2_sequences[chain]
                )
                # sequenceA (aligned f1); sequenceB (aligned f2)
                rcsb_sequences[chain], mol2_sequences[chain] = trace_back_gotoh(
                    idir, jpV, jpH, rcsb_sequences[chain], mol2_sequences[chain]
                )

        mol2_sequences = self.to_fasta()
        # print(mol2_sequences)
        rcsb_sequences = read_fasta()
        # print(rcsb_sequences)
        align_sequences()
        # self.aligned_sequences = mol2_sequences
        # self.rcsb_sequences = rcsb_sequences
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
                if sub_type == "PLP":
                    pass
                if self.atom_df[
                    (self.atom_df["subst_id"] == subst_id)
                    & (self.atom_df["atom_name"] == "CA")
                ].empty:
                    continue
                # print(sub_type, aa)
                if sub_type in aa3to1:
                    assert aa3to1[sub_type] == aa
                else:
                    assert aa == "X"
                self.subst_df.at[df_idxs[jdx], "reindex_id"] = idx + 1
                jdx += 1

    def write_pdb(self, pdb_file, reindex_format=True):
        def pdb_line(record):
            if reindex_format and record[4] == 0:
                ignored.add(record[2])
                return ""
            space = " "
            serial = str(record[0]).rjust(5)
            name = record[1].center(4)
            altLoc = space
            resName = record[2].rjust(3)
            chainID = record[3]
            resSeq = str(record[4]).rjust(4)
            iCode = space
            x = "%8.3f" % record[5]
            y = "%8.3f" % record[6]
            z = "%8.3f" % record[7]
            occupancy = "%6.2f" % 1.0
            tempFactor = "%6.2f" % 0.0
            element = record[1][0].ljust(2)
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

        ignored = set()
        df = self.atom_df.merge(self.subst_df, "left", ["subst_id", "subst_name"])
        if "reindex_id" not in df.columns:
            df["reindex_id"] = df["subst_id"]
        pdb_text = [
            pdb_line(record)
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
        print("Ignored", ignored)
        with open(pdb_file, "w") as f:
            f.writelines(pdb_text)
        return pdb_text


# pdb_id = "12gs_1"
# test = Mol2("../data/scPDB/raw/" + pdb_id + "/protein.mol2")
# print(test.to_fasta())
# test.reindex("../data/scPDB/raw/" + pdb_id + "/sequence.fasta")
# test.write_pdb("tmp.pdb")
# test
