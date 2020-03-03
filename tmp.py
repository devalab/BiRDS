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
