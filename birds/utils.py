import os
import re
import numpy as np
from collections import defaultdict


def is_dna_rna_sequence(sequence):
    def match(strg, search=re.compile(r"[^ACGTURYKMSWBDHVN\-\.]").search):
        return not bool(search(strg))

    if match(sequence):
        return True
    return False


def create_chains_from_sequence_fasta(pre, sequence_fasta, do_not_include=None):
    with open(sequence_fasta, "r") as f:
        header = f.readline()
        while 1:
            chain_id = header[6:7]
            sequence = ""
            line = f.readline()
            while line != "" and line is not None and line[0] != ">":
                sequence += line.strip()
                line = f.readline()
            if (
                not do_not_include or chain_id not in do_not_include
            ) and not is_dna_rna_sequence(sequence):
                with open(os.path.join(pre, chain_id + ".fasta"), "w") as hlp:
                    hlp.write(header)
                    hlp.write(sequence + "\n")
            if line == "" or line is None:
                break
            header = line


def make_unique_file(raw_dir, splits_dir):
    sequences = defaultdict(list)
    for file in sorted(os.listdir(raw_dir)):
        pre = os.path.join(raw_dir, file.strip())
        for fasta in sorted(os.listdir(pre)):
            if fasta[2:] != "fasta":
                continue
            chain_id = fasta[0]
            with open(os.path.join(pre, fasta)) as f:
                f.readline()
                sequence = f.readline().strip()
                # This choice was made so that rsync would work much better and easier
                sequences[sequence].append(file + "/" + chain_id + "*")
    keys = list(sequences.keys())

    if not os.path.exists(splits_dir):
        os.makedirs(splits_dir)
    with open(os.path.join(splits_dir, "unique"), "w") as f:
        for key in keys:
            line = ""
            for chain_id in sequences[key]:
                line += chain_id + " "
            f.write(line[:-1] + "\n")


def split_unique_file(splits_dir, num_seq_per_file=100):
    with open(os.path.join(splits_dir, "unique"), "r") as f:
        keys = [line.strip().split()[0] for line in f.readlines()]
    unique_ln = len(keys)

    for i in range(0, unique_ln // num_seq_per_file + 1):
        with open(os.path.join(splits_dir, str(i)), "w") as f:
            for key in keys[
                i * num_seq_per_file : min(unique_ln, (i + 1) * num_seq_per_file)
            ]:
                f.write(key + "\n")


def convert_feature_to_numpy_array(raw_dir, pis, chain, length, feat_type):
    with open(os.path.join(raw_dir, pis, chain + "." + feat_type), "r") as f:
        lines = f.readlines()
    if feat_type == "pssm":
        feature = np.array([line.strip().split() for line in lines], dtype=np.float32)
    elif feat_type == "aap":
        feature = np.array(
            [line.strip().split()[1:] for line in lines[2:-1]], dtype=np.float32
        ).T
    elif feat_type == "ss2":
        feature = np.array(
            [line.strip().split()[3:] for line in lines[2:]], dtype=np.float32
        ).T
    elif feat_type == "solv":
        feature = np.array(
            [line.strip().split()[2] for line in lines], dtype=np.float32
        )[np.newaxis, :]
    assert feature.shape[1] == length
    return feature


def store_features_as_numpy(dataset_dir, file):
    raw_dir = os.path.join(dataset_dir, "raw")
    prep_dir = os.path.join(dataset_dir, "preprocessed")
    if not os.path.exists(prep_dir):
        os.makedirs(prep_dir)
    with open(file, "r") as f:
        lines = [line.strip().split()[0] for line in f.readlines()]

    for line in lines:
        pis, chain = line.split("/")
        chain = chain[:-1]
        with open(os.path.join(raw_dir, pis, chain + ".fasta"), "r") as f:
            f.readline()
            sequence = f.readline().strip()
        pssm = convert_feature_to_numpy_array(
            raw_dir, pis, chain, len(sequence), "pssm"
        )
        aap = convert_feature_to_numpy_array(raw_dir, pis, chain, len(sequence), "aap")
        ss2 = convert_feature_to_numpy_array(raw_dir, pis, chain, len(sequence), "ss2")
        solv = convert_feature_to_numpy_array(
            raw_dir, pis, chain, len(sequence), "solv"
        )
        pre = os.path.join(prep_dir, pis)

        if not os.path.exists(pre):
            os.makedirs(pre)
        np.save(os.path.join(pre, "pssm_" + chain + ".npy"), pssm)
        np.save(os.path.join(pre, "aap_" + chain + ".npy"), aap)
        np.save(os.path.join(pre, "ss2_" + chain + ".npy"), ss2)
        np.save(os.path.join(pre, "solv_" + chain + ".npy"), solv)


def create_info_file(dataset_dir):
    raw_dir = os.path.join(dataset_dir, "raw")
    lines = []
    for pis in sorted(os.listdir(raw_dir)):
        pre = os.path.join(raw_dir, pis.strip())
        for fasta in sorted(os.listdir(pre)):
            if fasta[2:] != "fasta":
                continue
            chain_id = fasta[0]
            with open(os.path.join(pre, fasta)) as f:
                f.readline()
                sequence = f.readline().strip()
            lines.append(pis.split("_") + [chain_id, sequence, "0" * len(sequence)])
    lines = ["\t".join(line) + "\n" for line in lines]

    with open(os.path.join(dataset_dir, "info.txt"), "w") as f:
        f.write(
            "\t".join(
                ["pdb_id", "structure", "chain", "sequence", "binding_residues\n"]
            )
        )
        f.writelines(lines)
