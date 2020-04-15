# MAKING OF .npz FILES FROM HERE
# Download NWalign.py, pdb2fasta.py and reindex_pdb.py
# https://zhanglab.ccmb.med.umich.edu/NW-align/NWalign.py (Make small changes for Python3 compatibility)
# https://zhanglab.ccmb.med.umich.edu/reindex_pdb/reindex_pdb.py (Make changes to allow for reindexing specific chains and also to replace celenocysteine, "U" with "X")
# https://zhanglab.ccmb.med.umich.edu/reindex_pdb/pdb2fasta.py (Almost the same)
# Let us preprocess ALL the available data and create .npz files which contain pdb_id, chain_id, sequence, length, labels
# %debug
# from IPython.core.debugger import set_trace
import numpy as np
from reindex_pdb import reindex_pdb
from time import time
from Bio.PDB import PDBParser, is_aa
from Bio import BiopythonWarning
import warnings
from rdkit import Chem

warnings.simplefilter("ignore", BiopythonWarning)
parser = PDBParser()


def initialize_protein_info(pdb_id_struct, chain_id):
    pre = os.path.join(raw_dir, pdb_id_struct)
    protein = {}
    protein["structure"] = parser.get_structure(
        pdb_id_struct, os.path.join(pre, "tmp.pdb")
    )

    protein["residues"] = []
    for res in protein["structure"][0][chain_id]:
        id = res.get_id()
        if is_aa(res, standard=False) and id[0] == " ":
            protein["residues"].append(res)

    protein["sequence"] = ""
    with open(os.path.join(pre, chain_id + ".fasta")) as f:
        line = f.readline()
        line = f.readline()
        while line != "" and line is not None:
            protein["sequence"] += line.strip()
            line = f.readline()
    return protein


def initialize_ligand_info(pdb_id_struct, chain_id):
    pre = os.path.join(raw_dir, pdb_id_struct)
    ligand = {}
    ligand["supplier"] = Chem.SDMolSupplier(
        os.path.join(pre, "ligand.sdf"), sanitize=False
    )
    assert len(ligand["supplier"]) == 1
    ligand["supplier"] = ligand["supplier"][0]
    assert ligand["supplier"].GetNumConformers() == 1
    ligand["coords"] = ligand["supplier"].GetConformer().GetPositions()
    ligand["num_atoms"] = ligand["supplier"].GetNumAtoms()
    assert ligand["num_atoms"] == len(ligand["coords"])
    ligand["atom_types"] = np.array(
        [atom.GetSymbol() for atom in ligand["supplier"].GetAtoms()]
    )
    return ligand


def find_residues_in_contact(protein, ligand):
    """
    Returns a numpy 1D array where a 1 represents that the amino acid is in
    contact with the ligand
    """
    labels = np.zeros(len(protein["sequence"]))
    for residue in protein["residues"]:
        res_ind = residue.get_id()[1] - 1
        for atom in residue.get_atoms():
            if atom.get_fullname()[1] == "H":
                continue
            for i in range(ligand["num_atoms"]):
                if ligand["atom_types"][i] == "H":
                    continue
                # We are considering the ligand to be in contact with the AA
                # if the distance between them is within 5A
                if np.linalg.norm(atom.get_coord() - ligand["coords"][i]) <= 5.0:
                    labels[res_ind] = 1
                    break
            # We know that the residue is in contact with ligand
            # So go to the next residue
            if labels[res_ind]:
                break
    return labels


def get_distance_map_true(protein, atom_type):
    length = len(protein["sequence"])
    distance_map = np.full((length, length), np.inf)  # Initialize to infinite distance
    for res1 in protein["residues"]:
        res1_ind = res1.get_id()[1] - 1
        for res2 in protein["residues"]:
            res2_ind = res2.get_id()[1] - 1
            distance_map[res1_ind][res2_ind] = np.linalg.norm(
                res1[atom_type].get_coord() - res2[atom_type].get_coord()
            )
    np.fill_diagonal(distance_map, 0.0)
    return distance_map


process_time = 0
write_time = 0
for pdb_id_struct in sorted(os.listdir(raw_dir)):
    pre = os.path.join(raw_dir, pdb_id_struct)
    if not os.path.exists(os.path.join(pre, "downloaded.pdb")):
        print("Downloaded PDB does not exist for %s" % pdb_id_struct)
        continue
    process_time_start = time()
    for file in os.listdir(pre):
        # Get only the chain fasta sequences
        if file[2:] != "fasta":
            continue
        chain_id = file[0]

        # If our preprocessed file exists, continue
        if os.path.exists(
            os.path.join(preprocessed_dir, pdb_id_struct, chain_id + ".npz")
        ):
            continue

        print(pdb_id_struct, chain_id)
        # Reindex the chain and write to tmp.pdb
        dest = os.path.join(pre, "tmp.pdb")
        PDBtxt_reindex = reindex_pdb(
            os.path.join(pre, chain_id + ".fasta"),
            os.path.join(pre, "downloaded.pdb"),
            True,
        )
        # set_trace()
        if PDBtxt_reindex is None:
            print(pdb_id_struct, chain_id, "failed")
            continue

        with open(dest, "w") as fp:
            fp.write(PDBtxt_reindex)

        # Initialize information required for the complex
        protein = initialize_protein_info(pdb_id_struct, chain_id)
        ligand = initialize_ligand_info(pdb_id_struct, chain_id)

        # Make the dictionary for storage
        data = {}
        data["pdb_id_struct"] = pdb_id_struct
        data["chain_id"] = chain_id
        data["sequence"] = protein["sequence"]
        data["length"] = len(data["sequence"])
        data["labels"] = find_residues_in_contact(protein, ligand)
        assert len(data["sequence"]) == len(data["labels"])

        # Remove the tmp.pdb file
        os.remove(dest)
        process_time += time() - process_time_start

        # Write the data to a numpy .npz file
        write_time_start = time()
        folder = os.path.join(preprocessed_dir, pdb_id_struct)
        if not os.path.exists(folder):
            os.mkdir(folder)
        np.savez(os.path.join(folder, chain_id + ".npz"), **data)
        write_time += time() - write_time_start

print("Processing time:", process_time)
print("Write time:", write_time)
