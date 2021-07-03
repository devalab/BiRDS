import os
import re
import sys

import numpy as np

raw_dir = os.path.abspath("data/scPDB")


def delete_two_lines(original_file, line_number):
    """ Delete a line from a file at the given line number """
    is_skipped = False
    current_index = 0
    dummy_file = original_file + ".bak"
    # Open original file in read only mode and dummy file in write mode
    with open(original_file, "r") as read_obj, open(dummy_file, "w") as write_obj:
        # Line by line copy data from original file to dummy file
        for line in read_obj:
            # If current line number matches the given line number then skip copying
            if current_index != line_number and current_index != line_number + 1:
                write_obj.write(line)
            else:
                is_skipped = True
            current_index += 1

    # If any line is skipped then rename dummy file as original file
    if is_skipped:
        os.remove(original_file)
        os.rename(dummy_file, original_file)
    else:
        os.remove(dummy_file)


def fix_a3m(file, error):
    line_no = int(error.strip().split(" ")[-1]) - 2
    delete_two_lines(file, line_no)


def make_clean_a3m():
    from string import ascii_lowercase

    remove_lowercase = str.maketrans(dict.fromkeys(ascii_lowercase))

    unique_file = os.path.abspath("splits/unique")
    with open(unique_file, "r") as f:
        lines = f.readlines()

    for line in lines:
        pdb_id_struct, chain_id = line.strip().split()[0].split("/")
        chain_id = chain_id[:-1]
        pre = os.path.join(raw_dir, pdb_id_struct)
        if os.path.exists(os.path.join(pre, chain_id + ".a3m")):
            continue

        hhba3m = os.path.join(pre, chain_id + ".hhba3m")
        jaca3m = os.path.join(pre, chain_id + ".jaca3m")
        with open(os.path.join(pre, chain_id + ".aln"), "r") as f:
            aln_lines = f.readlines()

        sigh = {}
        if os.path.exists(hhba3m):
            with open(hhba3m, "r") as f:
                a3m_lines = f.readlines()
            for i in range(0, len(a3m_lines), 2):
                a3m_lines[i + 1] = a3m_lines[i + 1].translate(remove_lowercase)
                sigh[a3m_lines[i + 1]] = a3m_lines[i]

        if os.path.exists(jaca3m):
            with open(jaca3m, "r") as f:
                a3m_lines = f.readlines()
            for i in range(0, len(a3m_lines), 2):
                a3m_lines[i + 1] = a3m_lines[i + 1].translate(remove_lowercase)
                sigh[a3m_lines[i + 1]] = a3m_lines[i]

        new_a3m_lines = []
        for aln_line in aln_lines:
            new_a3m_lines.append(sigh[aln_line])
            new_a3m_lines.append(aln_line)
        with open(os.path.join(pre, chain_id + ".a3m"), "w") as f:
            f.writelines(new_a3m_lines)


def remove_extra_files(pre, chain_id, rm_fasta=False):
    fasta = os.path.join(pre, chain_id + ".fasta")
    aln = os.path.join(pre, chain_id + ".aln")
    jaca3m = os.path.join(pre, chain_id + ".jaca3m")
    hhba3m = os.path.join(pre, chain_id + ".hhba3m")
    a3m = os.path.join(pre, chain_id + ".a3m")
    if rm_fasta and os.path.exists(fasta):
        os.remove(fasta)
    if os.path.exists(aln):
        os.remove(aln)
    if os.path.exists(jaca3m):
        os.remove(jaca3m)
    if os.path.exists(hhba3m):
        os.remove(hhba3m)
    if os.path.exists(a3m):
        os.remove(a3m)


def remove_dna_rna_msas():
    import re

    def match(strg, search=re.compile(r"[^AGTCU]").search):
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
                remove_extra_files(pre, chain, True)


def remove_non_unique_files():
    unique_file = os.path.abspath("splits/unique")
    with open(unique_file, "r") as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip().split()
        for extras in line[1:]:
            pdb_id_struct, chain = extras.split("/")
            chain = chain[:-1]
            pre = os.path.join(raw_dir, pdb_id_struct)
            remove_extra_files(pre, chain)


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


# with open("./splits/unique", "r") as f:
#     lines = f.readlines()

# for line in lines:
#     line = line.strip().split()
#     pdb_id_struct, chain_id = line[0].split("/")
#     chain_id = chain_id[:-1]
#     # check_aln(pdb_id_struct, chain_id)
#     check_a3m(pdb_id_struct, chain_id)
#     check_pssm(pdb_id_struct, chain_id)

# check_dna_rna_seqs()
# a3m_format()
