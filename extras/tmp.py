from Bio.PDB import *
from rdkit import Chem
import os
from reindex_pdb import reindex_pdb

# import nglview as nv
import numpy as np

raw_dir = "./data/scPDB/raw"
parser = PDBParser()
pdb_id = "3mvq_7"
chain_id = "F"

pre = os.path.join(raw_dir, pdb_id)
dest = os.path.join(pre, "tmp.pdb")
PDBtxt_reindex = reindex_pdb(
    os.path.join(pre, chain_id + ".fasta"),
    os.path.join(pre, "downloaded.pdb"),
    True,
)
with open(dest, "w") as fp:
    fp.write(PDBtxt_reindex)
# Protein structure
structure = parser.get_structure(
    pdb_id, dest
)

# os.remove(dest)


# Run a regex search on the generated fasta files to ensure that we don't have any DNA/RNA sequences
# Will have to manually check the files to ensure that they are protein sequences
# All of them are protein sequences
import re


def match(strg, search=re.compile(r"[^ACGTURYKMSWBDHVN\-\.]").search):
    return not bool(search(strg))


with open(os.path.join(data_dir, "unique"), "r") as f:
    lines = f.readlines()

for line in lines:
    pdb_id_struct, chain_id = line.strip().split()[0].split("/")
    chain_id = chain_id[0]
    with open(os.path.join(raw_dir, pdb_id_struct, chain_id + ".fasta"), "r") as f:
        f.readline()
        seq = f.readline().strip()
    if match(seq):
        print(pdb_id_struct, chain_id)
