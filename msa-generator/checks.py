import sys
import os
import numpy as np
import re

raw_dir = os.path.abspath("data/scPDB")


def aln_exists():
    splits_dir = os.path.abspath("splits")
    try:
        arg = sys.argv[1]
    except IndexError:
        arg = "unique"

    with open(os.path.join(splits_dir, arg), "r") as f:
        lines = f.readlines()

    cnt = 0

    for file in lines:
        if arg == "unique":
            file, chain = file.strip().split()[0].split("/")
        else:
            file, chain = file.strip().split("/")
        chain = chain[:-1]
        pre = os.path.join(raw_dir, file)
        if os.path.exists(os.path.join(pre, chain + ".aln")):
            # print(file + "_" + chain + " alignment exists")
            cnt += 1

    print(arg, cnt)


def a3m_format():
    pdb_id_struct = sys.argv[1]
    chain = sys.argv[2]
    a3m = os.path.join(raw_dir, pdb_id_struct, chain + ".a3m")
    with open(a3m, "r") as f:
        lines = f.readlines()
    ln = len(lines[1])
    for i in range(3, len(lines), 2):
        assert ln == len(lines[i])


def check_pssm(pdb_id_struct, chain_id):
    pssm = os.path.join(raw_dir, pdb_id_struct, chain_id + ".pssm")
    if not os.path.exists(pssm):
        return
    with open(pssm, "r") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        line = np.array(line.strip().split(), dtype=np.float32)
        if np.any(np.isnan(line)) is np.True_:
            print(pdb_id_struct, chain_id)
            break


def check_dna_rna_seqs():
    def match(strg, search=re.compile(r"[^ACGTURYKMSWBDHVN\-\.]").search):
        return not bool(search(strg))

    for pdb_id_struct in sorted(os.listdir(raw_dir)):
        pre = os.path.join(raw_dir, pdb_id_struct)
        files = os.listdir(pre)
        for file in files:
            if file[2:] != "aln":
                continue
            chain = file[0]
            with open(os.path.join(pre, file), "r") as f:
                seq = f.readline().strip()
            if match(seq):
                print(pdb_id_struct, chain)


def check_aln(pdb_id_struct, chain_id):
    aln = os.path.join(raw_dir, pdb_id_struct, chain_id + ".aln")
    if not os.path.exists(aln) or os.stat(aln).st_size == 0:
        print(pdb_id_struct, chain_id)


def check_a3m(pdb_id_struct, chain_id):
    a3m = os.path.join(raw_dir, pdb_id_struct, chain_id + ".a3m")
    if not os.path.exists(a3m) or os.stat(a3m).st_size == 0:
        print(pdb_id_struct, chain_id)


with open("./splits/unique", "r") as f:
    lines = f.readlines()

for line in lines:
    line = line.strip().split()
    pdb_id_struct, chain_id = line[0].split("/")
    chain_id = chain_id[:-1]
    # check_aln(pdb_id_struct, chain_id)
    check_a3m(pdb_id_struct, chain_id)
#     check_pssm(pdb_id_struct, chain_id)

# check_dna_rna_seqs()
# a3m_format()
